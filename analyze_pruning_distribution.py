"""Which layers does global Taylor pruning remove, and how much of each?

The Taylor scores |w * dL/dw| do not depend on the pruning rate, and the mask of every rate is
"the lowest rate * N scores over all Linear layers" (main_measure.global_mask_from_taylor_scores).
So the scores are computed once per model (same calibration images as compare_acc_by_each_pruning.py)
and the masks of all pruning rates are rebuilt from them without pruning and evaluating again.

Outputs (per model, in ./prun_acc_figure/<model>/layer_distribution/):
    taylor_scores.pt   cache of the scores and the layer execution order (delete or --recompute to redo)
    layer_stats.csv    one row per (pruning rate, Linear layer)
    summary.csv        one row per pruning rate (head / body pruned ratio)
    heatmap_*.png      pruned ratio of each layer type (row) x execution order (column)
    share_*.png        breakdown of the pruned weights by layer type against the pruning rate
Comparison of the models (in ./prun_acc_figure/layer_distribution/):
    depth_*.png        pruned ratio against relative depth, one subplot per pruning rate
    head_body_*.png    pruned ratio of the classifier head and of the other layers

Usage examples:
    python analyze_pruning_distribution.py --models sret_s vit_small_patch16_224
    python analyze_pruning_distribution.py --models sret_s sret_t vit_small_patch16_224 vit_tiny_patch16_224 --plot-rates 0.3 0.5 0.7
    python analyze_pruning_distribution.py --models sret_s --verify-rates 0.3 0.5   # re-evaluate with the rebuilt masks
"""

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import os
from pathlib import Path
import pickle

from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn.utils import prune

from compare_acc_by_each_pruning import Config, build_loaders, load_model, make_calibration_batches
from main import evaluate_topk, set_seed
from main_measure import collect_taylor_scores, global_mask_from_taylor_scores, iter_prunable_linear_layers
from show_prun_acc_graph import DATA_ROOT, get_model_style

PRUNING_RATES = np.arange(0.0, 1.0, 0.05)  # same rates as compare_acc_by_each_pruning.py
DEFAULT_PLOT_RATES = [0.3, 0.5, 0.7]
INPUT_SIZE = (1, 3, 224, 224)  # dummy input used to record the execution order of the layers
OUTPUT_DIR_NAME = "layer_distribution"
COMPARE_DIR = DATA_ROOT / OUTPUT_DIR_NAME

# Categorical colors of the layer types (fixed order, assigned by first appearance in the model).
CATEGORY_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
HEATMAP_CMAP = "Blues"  # sequential: 0 (light) -> 1 (dark)
MISSING_COLOR = "#f2f2f2"  # cells where the layer type does not exist (hatched)
DPI = 200


# * ===== Layer information =====

@dataclass
class LayerInfo:
    name: str          # module name, e.g. "transformers.1.blocks.0.attn.qkv"
    block_name: str    # e.g. "transformers.1.blocks.0" (the layer itself for the head)
    block_kind: str    # class name of the block, e.g. "Transformer_Block", "Non_proj", "Block"
    layer_type: str    # e.g. "attn.qkv", "mlp.fc1", "head"
    exec_steps: list[int]  # block steps (execution order) in which the layer is used
    is_head: bool      # the last executed Linear layer = classifier

    @property
    def uses(self) -> int:
        """Number of times the layer is used in one forward pass (> 1 for the shared blocks of SReT)."""
        return len(self.exec_steps)

    @property
    def row_key(self) -> str:
        return "head" if self.is_head else f"{self.block_kind}/{self.layer_type}"


def split_layer_name(name: str) -> tuple[str, str]:
    """"blocks.0.attn.qkv" -> ("blocks.0", "attn.qkv"); a layer without a parent block is its own block."""
    parts = name.split(".")
    if len(parts) < 3:
        return name, name
    return ".".join(parts[:-2]), ".".join(parts[-2:])


def record_layer_info(model: nn.Module, device: torch.device) -> list[LayerInfo]:
    """Run one forward pass and record in which block step each Linear layer is used.
    A block step increases every time the executed block changes, so a shared SReT block gets several steps."""
    layers = dict(iter_prunable_linear_layers(model))  # named_modules() gives a shared module only once
    calls: list[str] = []
    handles = [
        module.register_forward_hook(lambda _m, _i, _o, name=name: calls.append(name))
        for name, module in layers.items()
    ]
    model.eval()
    with torch.no_grad():
        model(torch.zeros(INPUT_SIZE, device=device))
    for handle in handles:
        handle.remove()

    steps: dict[str, list[int]] = {name: [] for name in layers}
    step, previous_block = -1, None
    for name in calls:
        block_name, _ = split_layer_name(name)
        if block_name != previous_block:
            step += 1
            previous_block = block_name
        if not steps[name] or steps[name][-1] != step:
            steps[name].append(step)

    head_name = calls[-1]
    infos = []
    for name in layers:
        block_name, layer_type = split_layer_name(name)
        block_kind = type(model.get_submodule(block_name)).__name__ if block_name != name else "-"
        infos.append(LayerInfo(name, block_name, block_kind, "head" if name == head_name else layer_type,
                               steps[name], name == head_name))
    return infos


# * ===== Taylor scores (cached) =====

def cache_path(model_name: str) -> Path:
    return DATA_ROOT / model_name / OUTPUT_DIR_NAME / "taylor_scores.pt"


def compute_scores(model_name: str, device: torch.device) -> dict:
    """Taylor scores with the same calibration images as compare_acc_by_each_pruning.py, and the layer info."""
    set_seed(42)
    model = load_model(model_name, device)
    train_loader, _ = build_loaders(model, model_name)
    scores = collect_taylor_scores(
        model,
        make_calibration_batches(train_loader),
        nn.CrossEntropyLoss(),
        device,
        prune_bias=Config.prune_bias,
    )
    return {
        "scores": {key: score.cpu() for key, score in scores.items()},
        "layers": [asdict(info) for info in record_layer_info(model, device)],
        "config": {**asdict(Config()), "model_name": model_name},
    }


def load_or_compute_scores(model_name: str, device: torch.device, recompute: bool) -> dict:
    path = cache_path(model_name)
    if path.exists() and not recompute:
        print(f"[{model_name}] load cached scores: {path}")
        return torch.load(path, weights_only=False)
    print(f"[{model_name}] compute Taylor scores")
    data = compute_scores(model_name, device)
    os.makedirs(path.parent, exist_ok=True)
    torch.save(data, path)
    return data


# * ===== Distribution =====

def compute_layer_stats(model_name: str, data: dict) -> list[dict]:
    """One row per (pruning rate, layer). The masks are made by the same function as the pruning experiment."""
    infos = [LayerInfo(**info) for info in data["layers"]]
    scores = data["scores"]
    n_steps = max(max(info.exec_steps) for info in infos if not info.is_head)
    rows = []
    for rate in PRUNING_RATES:
        masks = global_mask_from_taylor_scores(scores, float(rate))
        total_pruned = sum(mask.numel() - int(mask.sum().item()) for mask in masks.values())
        for info in infos:
            mask = masks[(info.name, "weight")]
            pruned = mask.numel() - int(mask.sum().item())
            rows.append({
                "model": model_name,
                "pruning_rate": round(float(rate), 2),
                "layer_name": info.name,
                "block_name": info.block_name,
                "block_kind": info.block_kind,
                "layer_type": info.layer_type,
                "row_key": info.row_key,
                "is_head": info.is_head,
                "numel": mask.numel(),
                "pruned": pruned,
                "layer_prune_ratio": pruned / mask.numel(),
                "share_of_pruned": pruned / total_pruned if total_pruned else 0.0,
                "uses": info.uses,
                "exec_steps": ";".join(map(str, info.exec_steps)),
                "rel_depth": float(np.mean(info.exec_steps)) / n_steps if n_steps else 0.0,
            })
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    """One row per pruning rate: pruned ratio of the whole model, the head and the other layers."""
    summary = []
    for rate in sorted({row["pruning_rate"] for row in rows}):
        rate_rows = [row for row in rows if row["pruning_rate"] == rate]
        head = [row for row in rate_rows if row["is_head"]]
        body = [row for row in rate_rows if not row["is_head"]]
        summary.append({
            "model": rate_rows[0]["model"],
            "pruning_rate": rate,
            "total_ratio": sum(r["pruned"] for r in rate_rows) / sum(r["numel"] for r in rate_rows),
            "head_ratio": sum(r["pruned"] for r in head) / sum(r["numel"] for r in head) if head else 0.0,
            "body_ratio": sum(r["pruned"] for r in body) / sum(r["numel"] for r in body),
            "head_share_of_pruned": sum(r["share_of_pruned"] for r in head),
        })
    return summary


def save_csv(rows: list[dict], path: Path) -> None:
    os.makedirs(path.parent, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved: {path}")


def rows_at(rows: list[dict], rate: float) -> list[dict]:
    return [row for row in rows if np.isclose(row["pruning_rate"], rate)]


def row_keys_in_order(rows: list[dict]) -> list[str]:
    """Layer types in order of first appearance in the model, the head last."""
    keys = list(dict.fromkeys(row["row_key"] for row in rows if not row["is_head"]))
    return keys + (["head"] if any(row["is_head"] for row in rows) else [])


def ratio_per_step(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """(relative depth, pruned ratio) of each block step, without the head.
    A shared layer counts in every step that uses it."""
    pruned: dict[int, int] = {}
    numel: dict[int, int] = {}
    for row in rows:
        if row["is_head"]:
            continue
        for step in map(int, row["exec_steps"].split(";")):
            pruned[step] = pruned.get(step, 0) + row["pruned"]
            numel[step] = numel.get(step, 0) + row["numel"]
    steps = np.array(sorted(numel))
    return steps / steps.max(), np.array([pruned[s] / numel[s] for s in steps])


# * ===== Plot (per model) =====

def short_block_name(block_name: str) -> str:
    return block_name.replace("transformers.", "s").replace("blocks.", "b")


def plot_heatmap(rows: list[dict], plot_rates: list[float], model_name: str, out_dir: Path) -> None:
    """Rows = layer types, columns = block steps (execution order), color = pruned ratio of the layer.
    A dot marks a layer shared by several steps (the same mask in every step)."""
    keys = row_keys_in_order(rows)
    n_steps = max(int(s) for row in rows for s in row["exec_steps"].split(";")) + 1
    step_labels = [""] * n_steps
    for row in rows_at(rows, plot_rates[0]):
        for step in map(int, row["exec_steps"].split(";")):
            step_labels[step] = "head" if row["is_head"] else short_block_name(row["block_name"])

    fig, axes = plt.subplots(len(plot_rates), 1, figsize=(max(8, n_steps * 0.3 + 3), 1 + len(plot_rates) * (0.45 * len(keys) + 0.8)),
                             squeeze=False)
    for ax, rate in zip(axes[:, 0], plot_rates):
        grid = np.full((len(keys), n_steps), np.nan)
        for row in rows_at(rows, rate):
            for step in map(int, row["exec_steps"].split(";")):
                grid[keys.index(row["row_key"]), step] = row["layer_prune_ratio"]
                if row["uses"] > 1:
                    ax.plot(step, keys.index(row["row_key"]), "o", color="white", markersize=2.5,
                            markeredgecolor="black", markeredgewidth=0.4)
        ax.set_facecolor(MISSING_COLOR)
        ax.patch.set_hatch("///")
        ax.patch.set_edgecolor("#d0d0d0")
        image = ax.imshow(grid, cmap=HEATMAP_CMAP, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        ax.set_yticks(range(len(keys)), keys, fontsize=8)
        ax.set_xticks(range(n_steps), step_labels, rotation=90, fontsize=7)
        ax.set_title(f"pruning rate {rate:.2f}", fontsize=10, loc="left")
        fig.colorbar(image, ax=ax, label="pruned ratio in the layer", fraction=0.03, pad=0.01)
    fig.suptitle(f"{get_model_style(model_name, 0)['label']}: pruned ratio per layer (columns = execution order, "
                 f"dot = shared layer)", fontsize=11)
    fig.tight_layout()
    save_figure(fig, out_dir / f"heatmap_{rates_tag(plot_rates)}.png")


def plot_share(rows: list[dict], model_name: str, out_dir: Path) -> None:
    """Breakdown of the pruned weights by layer type, against the pruning rate."""
    keys = row_keys_in_order(rows)
    rates = sorted({row["pruning_rate"] for row in rows})[1:]  # rate 0 prunes nothing
    shares = np.array([
        [sum(r["share_of_pruned"] for r in rows_at(rows, rate) if r["row_key"] == key) for rate in rates]
        for key in keys
    ])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.stackplot(rates, shares * 100, labels=keys, colors=CATEGORY_COLORS[:len(keys)],
                 edgecolor="white", linewidth=1.0)
    ax.set_xlabel("Pruning Rate")
    ax.set_ylabel("Share of the pruned weights [%]")
    ax.set_xlim(rates[0], rates[-1])
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, title="layer type", reverse=True)
    ax.set_title(f"{get_model_style(model_name, 0)['label']}: which layer types are pruned", loc="left")
    fig.tight_layout()
    save_figure(fig, out_dir / "share.png")


# * ===== Plot (comparison of the models) =====

def plot_depth(all_rows: dict[str, list[dict]], plot_rates: list[float]) -> None:
    """Pruned ratio of each block step against the relative depth (0 = first block, 1 = last block)."""
    fig, axes = plt.subplots(1, len(plot_rates), figsize=(5 * len(plot_rates), 4.5), sharey=True, squeeze=False)
    for ax, rate in zip(axes[0], plot_rates):
        for i, (model_name, rows) in enumerate(all_rows.items()):
            depth, ratio = ratio_per_step(rows_at(rows, rate))
            ax.plot(depth, ratio * 100, linewidth=1.5, markersize=4, **get_model_style(model_name, i))
        ax.axhline(rate * 100, color="gray", linewidth=0.8, linestyle=":")
        ax.set_title(f"pruning rate {rate:.2f}", loc="left")
        ax.set_xlabel("Relative Depth (execution order)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle=":", alpha=0.6)
    axes[0, 0].set_ylabel("Pruned ratio in the block [%]\n(head excluded, dotted = pruning rate)")
    axes[0, -1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), title="Model")
    fig.tight_layout()
    save_figure(fig, COMPARE_DIR / f"depth_{rates_tag(plot_rates)}_{models_tag(all_rows)}.png")


def plot_head_body(all_summary: dict[str, list[dict]]) -> None:
    """Pruned ratio of the classifier head (left) and of the other layers (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for i, (model_name, summary) in enumerate(all_summary.items()):
        style = get_model_style(model_name, i)
        rates = [s["pruning_rate"] for s in summary]
        axes[0].plot(rates, [s["head_ratio"] * 100 for s in summary], markersize=4, **style)
        axes[1].plot(rates, [s["body_ratio"] * 100 for s in summary], markersize=4, **style)
    for ax, title in zip(axes, ["classifier head", "all layers except the head"]):
        ax.plot([0, 1], [0, 100], color="gray", linewidth=0.8, linestyle=":")
        ax.set_title(title, loc="left")
        ax.set_xlabel("Pruning Rate (global)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle=":", alpha=0.6)
    axes[0].set_ylabel("Pruned ratio [%] (dotted = pruning rate)")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), title="Model")
    fig.tight_layout()
    save_figure(fig, COMPARE_DIR / f"head_body_{models_tag(all_summary)}.png")


def rates_tag(rates: list[float]) -> str:
    return "rate" + "_".join(f"{r:.2f}".replace(".", "") for r in rates)


def models_tag(by_model: dict) -> str:
    return "-".join(by_model) + datetime.now().strftime("_%Y%m%d_%H%M%S")


def save_figure(fig: plt.Figure, path: Path) -> None:
    os.makedirs(path.parent, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"saved: {path}")


# * ===== Verify =====

def verify(model_name: str, data: dict, rates: list[float], device: torch.device, max_batches: int | None) -> None:
    """Apply the rebuilt masks and evaluate; compare with the accuracy saved in data.pkl.
    The scores are not bit-identical to the original run (drop path / random grouping in SReT in train mode),
    so the accuracy should be close but not exactly the same."""
    with open(DATA_ROOT / model_name / "data.pkl", "rb") as f:
        saved = pickle.load(f)
    saved_rates = np.asarray(saved["x"]["pruning_rate"])
    for rate in rates:
        model = load_model(model_name, device)
        _, val_loader = build_loaders(model, model_name)
        modules = dict(iter_prunable_linear_layers(model))
        for (module_name, parameter_name), mask in global_mask_from_taylor_scores(data["scores"], rate).items():
            module = modules[module_name]
            prune.custom_from_mask(module, parameter_name, mask=mask.to(module.weight.device))
        top1 = evaluate_topk(model, val_loader, device, max_batches=max_batches)["top1"]
        index = int(np.argmin(np.abs(saved_rates - rate)))
        print(f"[{model_name}] rate {rate:.2f}: rebuilt mask top-1 {top1:.2%} / "
              f"data.pkl (rate {saved_rates[index]:.2f}) top-1 {saved['y']['imagenet_top1'][index]:.2%}")


# * ===== Main =====

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Layer-wise distribution of the global Taylor pruning.")
    parser.add_argument("--models", nargs="+", required=True, help="model names (see compare_acc_by_each_pruning.load_model)")
    parser.add_argument("--plot-rates", nargs="+", type=float, default=DEFAULT_PLOT_RATES,
                        help=f"pruning rates drawn in the heatmap / depth figures (default: {DEFAULT_PLOT_RATES})")
    parser.add_argument("--recompute", action="store_true", help="ignore the cached Taylor scores")
    parser.add_argument("--verify-rates", nargs="+", type=float, default=[],
                        help="re-evaluate with the rebuilt masks at these rates and compare with data.pkl")
    parser.add_argument("--verify-batches", type=int, default=None, help="number of val batches for --verify-rates (default: all)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_rows: dict[str, list[dict]] = {}
    all_summary: dict[str, list[dict]] = {}

    for model_name in args.models:
        data = load_or_compute_scores(model_name, device, args.recompute)
        out_dir = DATA_ROOT / model_name / OUTPUT_DIR_NAME
        rows = compute_layer_stats(model_name, data)
        summary = summarize(rows)
        save_csv(rows, out_dir / "layer_stats.csv")
        save_csv(summary, out_dir / "summary.csv")
        plot_heatmap(rows, args.plot_rates, model_name, out_dir)
        plot_share(rows, model_name, out_dir)
        all_rows[model_name] = rows
        all_summary[model_name] = summary
        if args.verify_rates:
            verify(model_name, data, args.verify_rates, device, args.verify_batches)

    plot_depth(all_rows, args.plot_rates)
    plot_head_body(all_summary)


if __name__ == "__main__":
    main()
