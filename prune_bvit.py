"""Step-conditioned pruning experiments on a RecurrentViT checkpoint (paper Sec. 4.2.3).

For every target group and method, prune at each ratio WITHOUT retraining and
evaluate on val. Importance scores are computed once per (group, unit, scorer)
on train images with the deterministic eval transform ([A] the paper says
"ImageNet-1K training data"; the number of images is not given).

Target groups: aliases Q K V PROJ FFN_IN FFN_OUT, or glob patterns on module
names; "+" prunes several targets together.
    --groups Q V FFN_IN                       three experiments (paper Fig. 6)
    --groups "V+FFN_IN" "blocks.0.attn.[qk]"  two experiments

Methods: see pruning/pruner.py METHODS
    paper: spectral_step spectral_global spectral_permuted spectral_sigma

Usage:
    uv run python prune_bvit.py --checkpoint outputs/bvit/<run>/best.pth
    uv run python prune_bvit.py --checkpoint ... --groups V --methods spectral_step taylor_step --ratios 0 0.3 0.6

Outputs (./outputs/bvit_pruning/<run>/): results.json / .csv, scores_<group>_<unit>_<scorer>.pt,
similarity_<group>.json and figures (accuracy vs ratio, persistence, step similarity, CLS vs patch).
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import islice
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch import nn

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from dl_dataset.registry import build_loaders, get_spec  # noqa: E402
from pruning import METHODS, Pruner, ScoringContext, parse_groups  # noqa: E402
from pruning.analysis import (  # noqa: E402
    chance_persistence, cls_patch_similarity, persistence, step_similarity)
from run_logging import add_log_args, link_run_log, run_timestamp, start_run_log  # noqa: E402
from train_bvit import Config, evaluate, load_model, set_seed  # noqa: E402

PAPER_METHODS = ["spectral_step", "spectral_global", "spectral_permuted", "spectral_sigma"]
DEFAULT_RATIOS = [round(0.1 * i, 1) for i in range(11)]


def group_label(group: list[str]) -> str:
    return "+".join(group).replace("*", "x").replace("/", "_").replace("[", "").replace("]", "")


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    model, checkpoint = load_model(args.checkpoint, use_ema=not args.no_ema, device=device)
    train_config = Config(**checkpoint["config"])
    spec = get_spec(checkpoint["dataset"], args.data_root or train_config.data_root)
    calib_loader, val_loader, _ = build_loaders(spec, batch_size=args.batch_size, num_workers=args.num_workers,
                                                train_eval_transform=True)
    calib_batches = list(islice(calib_loader, max(1, args.calib_images // args.batch_size)))
    loss_fn = nn.CrossEntropyLoss()

    out_dir = Path(args.output_dir) / (args.run_name or f"{Path(args.checkpoint).parent.name}_{run_timestamp()}")
    out_dir.mkdir(parents=True, exist_ok=True)
    link_run_log(out_dir)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))
    print(f"model loop={model.schedule} dataset={spec.name} calib_images={len(calib_batches) * args.batch_size}")
    print(f"output: {out_dir}")

    base = evaluate(model, val_loader, device, max_batches=args.eval_batches)
    print(f"unpruned: top1={base['top1']:.4f}")
    records: list[dict] = []
    for group in parse_groups(args.groups):
        label = group_label(group)
        score_cache: dict[tuple[str, str], dict] = {}
        for method_name in args.methods:
            method = METHODS[method_name]
            with Pruner(model, group, method, seed=args.seed) as pruner:
                key = (method.unit, method.scorer)
                if key not in score_cache:
                    score_cache[key] = pruner.compute_scores(
                        ScoringContext(model, calib_batches, loss_fn, device, amp=args.amp, seed=args.seed))
                    torch.save(score_cache[key], out_dir / f"scores_{label}_{method.unit}_{method.scorer}.pt")
                scores = score_cache[key]
                for ratio in args.ratios:
                    masks = pruner.make_masks(scores, ratio)
                    pruner.apply(masks)
                    metrics = evaluate(model, val_loader, device, max_batches=args.eval_batches)
                    record = {"group": label, "method": method_name, "ratio": ratio,
                              "top1": metrics["top1"], "top5": metrics["top5"],
                              "kept_fraction": sum(m.sum().item() for m in masks.values())
                              / sum(m.numel() for m in masks.values())}
                    if method.step_mode == "per_use":
                        record["persistence"] = {name: persistence(mask) for name, mask in masks.items()
                                                 if mask.shape[0] > 1}
                    records.append(record)
                    print(f"[{label}] {method_name:18s} ratio={ratio:.2f} top1={metrics['top1']:.4f}", flush=True)
        save_similarity(score_cache, label, out_dir)
        plot_group(records, label, out_dir)

    (out_dir / "results.json").write_text(json.dumps({"unpruned": base, "records": records}, indent=2))
    with open(out_dir / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "method", "ratio", "top1", "top5", "kept_fraction"],
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(f"saved: {out_dir}")


# * ===== Similarity / figures =====

def save_similarity(score_cache: dict, label: str, out_dir: Path) -> None:
    """Fig. 4 (step x step Spearman for CLS / patch) and Fig. 5 (CLS vs patch per step)."""
    scores = score_cache.get(("spectral", "grad_gate"))
    if scores is None:
        return
    result = {}
    for name, parts in scores.items():
        if parts["total"].shape[0] < 2:
            continue  # one use per layer (e.g. ViT): no recurrence to compare
        result[name] = {
            "cls_steps": step_similarity(parts["cls"]).tolist(),
            "patch_steps": step_similarity(parts["patch"]).tolist(),
            "cls_vs_patch": cls_patch_similarity(parts["cls"], parts["patch"]),
        }
    if not result:
        return
    (out_dir / f"similarity_{label}.json").write_text(json.dumps(result, indent=2, allow_nan=True))
    fig, axes = plt.subplots(len(result), 3, figsize=(13, 4 * len(result)), squeeze=False)
    for row, (name, sim) in enumerate(result.items()):
        for col, key in enumerate(("cls_steps", "patch_steps")):
            ax = axes[row, col]
            image = ax.imshow(np.array(sim[key], dtype=float), cmap="coolwarm", vmin=0, vmax=1)
            ticks = range(len(sim[key]))
            ax.set_xticks(ticks, [t + 1 for t in ticks])
            ax.set_yticks(ticks, [t + 1 for t in ticks])
            ax.set_title(f"{key.split('_')[0].upper()}, {name}")
            ax.set_xlabel("Recurrence step")
            ax.set_ylabel("Recurrence step")
            fig.colorbar(image, ax=ax, label="Spearman")
        ax = axes[row, 2]
        ax.plot(range(1, len(sim["cls_vs_patch"]) + 1), sim["cls_vs_patch"], marker="o", color="black")
        ax.set_ylim(0, 1)
        ax.set_title(f"CLS vs PATCH, {name}")
        ax.set_xlabel("Recurrence step")
        ax.set_ylabel("Spearman")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"similarity_{label}.png", dpi=150)
    plt.close(fig)


def plot_group(records: list[dict], label: str, out_dir: Path) -> None:
    """Fig. 6: accuracy vs pruning ratio, and persistence of step-conditioned masks."""
    group = [r for r in records if r["group"] == label]
    fig, ax = plt.subplots(figsize=(5, 4))
    for method in dict.fromkeys(r["method"] for r in group):
        rows = [r for r in group if r["method"] == method]
        ax.plot([100 * r["ratio"] for r in rows], [r["top1"] for r in rows], marker="o", label=method)
    ax.set_xlabel("Pruning ratio (%)")
    ax.set_ylabel("Accuracy")
    ax.set_title(label)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"accuracy_{label}.png", dpi=150)
    plt.close(fig)

    rows = [r for r in group if r.get("persistence")]
    if not rows:
        return
    for method in dict.fromkeys(r["method"] for r in rows):
        method_rows = [r for r in rows if r["method"] == method]
        layers = list(method_rows[0]["persistence"])
        fig, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), 4), squeeze=False)
        for ax, layer in zip(axes[0], layers):
            ratios = [r["ratio"] for r in method_rows]
            bins = list(method_rows[0]["persistence"][layer])
            num_uses = int(bins[-1].split("-")[-1])
            for b in bins:
                (line,) = ax.plot([100 * x for x in ratios], [100 * r["persistence"][layer][b] for r in method_rows],
                                  marker="o", label=b.replace("steps_", "Steps "))
                chance = [100 * chance_persistence(num_uses, 1 - x)[b] for x in ratios]
                ax.plot([100 * x for x in ratios], chance, linestyle=":", color=line.get_color())
            ax.set_xlabel("Pruning ratio (%)")
            ax.set_ylabel("Selected directions (%)")
            ax.set_title(f"{method}: {layer} (dotted: chance)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"persistence_{label}_{method}.png", dpi=150)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="best.pth / last.pth from train_bvit.py")
    parser.add_argument("--groups", nargs="+", default=["Q", "V", "FFN_IN"])
    parser.add_argument("--methods", nargs="+", default=PAPER_METHODS, choices=list(METHODS))
    parser.add_argument("--ratios", nargs="+", type=float, default=DEFAULT_RATIOS)
    parser.add_argument("--calib-images", type=int, default=10240)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batches", type=int, default=None, help="limit val batches (default: all)")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--no-ema", action="store_true", help="use raw weights instead of EMA weights")
    parser.add_argument("--amp", action="store_true", help="fp16 while scoring (default fp32)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="./outputs/bvit_pruning")
    parser.add_argument("--run-name", default=None)
    add_log_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    start_run_log(__file__, tag="+".join(args.groups), log_dir=args.log_dir, config=vars(args),
                  enabled=not args.no_log)
    run(args)
