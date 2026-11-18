"""Tables and figures for the multigrid loop ViT experiments (mgloop_experiments.py).

Reads every finished run, measures FLOPs / throughput of its architecture, and writes
    outputs/mgloop/results.json      all numbers
    outputs/mgloop/results.md        markdown table
    docs/mgloop/figures/*.png        figures used by docs/mgloop/multigrid_loop_vit.md

Usage:
    uv run python mgloop_report.py                    # needs a free GPU for the throughput numbers
    uv run python mgloop_report.py --skip-throughput  # reuse / skip throughput
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from dl_dataset.registry import get_spec  # noqa: E402
from mgloop_experiments import DATASET, EXPERIMENTS, OUTPUT_DIR, Experiment  # noqa: E402
from model_profile import count_flops, throughput  # noqa: E402
from models.bViT.bViT import build_recurrent_vit  # noqa: E402
from models.bViT.schedule import LoopSchedule  # noqa: E402

FIGURE_DIR = Path("docs/mgloop/figures")

# colors: baselines are gray, multigrid models take the categorical slots in a fixed order
SURFACE, TEXT, TEXT_SECONDARY, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
GRAY = "#8a8985"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
LEVEL_NAMES = ["F (8x8)", "C (4x4)", "CC (2x2)"]


def build(experiment: Experiment) -> torch.nn.Module:
    spec = get_spec(DATASET)
    extra = {"use_level_embed": False} if "--no-use-level-embed" in experiment.extra else {}
    return build_recurrent_vit(experiment.model, img_size=spec.img_size, patch_size=spec.patch_size,
                               num_classes=spec.num_classes, loop=experiment.loop, levels=experiment.levels,
                               **extra)


def step_levels(experiment: Experiment) -> list[int]:
    if experiment.levels is None:
        return [0] * LoopSchedule.parse(experiment.loop).num_steps
    return [step.block_id for step in LoopSchedule.parse(experiment.levels).unroll()]


def collect(skip_throughput: bool, previous: dict) -> dict[str, dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = get_spec(DATASET)
    results: dict[str, dict] = {}
    for name, experiment in EXPERIMENTS.items():
        final_path = experiment.run_dir / "final_eval.json"
        if not final_path.exists():
            continue
        final = json.loads(final_path.read_text())
        history = json.loads((experiment.run_dir / "history.json").read_text())
        model = build(experiment).to(device)
        levels = step_levels(experiment)
        row = {
            "label": experiment.label, "loop": experiment.loop, "levels": experiment.levels,
            "note": experiment.note, "step_levels": levels,
            "steps_per_level": [levels.count(level) for level in range(max(levels) + 1)],
            "top1": final["top1"], "top5": final["top5"], "top1_per_step": final["top1_per_step"],
            "best_ema_top1": max(r.get("ema_val_top1", 0.0) for r in history),
            "final_train_loss": history[-1]["train_loss"],
            "curve": [r.get("ema_val_top1") for r in history],
            "epoch_time_s": sum(r["epoch_time_s"] for r in history) / len(history),
            "params": sum(p.numel() for p in model.parameters()),
            **count_flops(model, spec.img_size, device),
        }
        # FLOPs spent up to and including each step (blocks only; embedding and head are negligible)
        per_level = []
        for level in range(max(levels) + 1):
            tokens = (model.grid_size >> level) ** 2 + model.num_prefix_tokens
            block = model.blocks[0]
            with torch.utils.flop_counter.FlopCounterMode(display=False) as counter:
                block(torch.randn(1, tokens, model.embed_dim, device=device))
            per_level.append(counter.get_total_flops() / 1e9)
        cumulative, total = [], 0.0
        for level in levels:
            total += per_level[level]
            cumulative.append(total)
        row["gflops_cumulative"] = cumulative
        if skip_throughput:
            for key in ("infer.img_per_s", "train.img_per_s", "infer.peak_mem_gb", "train.peak_mem_gb"):
                if key in previous.get(name, {}):
                    row[key] = previous[name][key]
        else:
            row.update(throughput(model, spec.img_size, device, train=False, batch_size=512, iters=50, warmup=10))
            row.update(throughput(model, spec.img_size, device, train=True, batch_size=128, iters=50, warmup=10))
        results[name] = row
        del model
    return results


# * ===== Table =====

def schedule_text(row: dict) -> str:
    return " + ".join(f"{count}{name.split()[0]}" for count, name in zip(row["steps_per_level"], LEVEL_NAMES)
                      if count)


def markdown_table(results: dict[str, dict]) -> str:
    reference = results.get("bvit_12")
    lines = ["| モデル | ステップ構成 | levels | Params (M) | GFLOPs | 推論 img/s | 学習 img/s | Top-1 (%) | Top-5 (%) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for row in results.values():
        flops = f"{row['gflops.total']:.2f}"
        if reference is not None:
            flops += f" ({row['gflops.total'] / reference['gflops.total']:.0%})"
        infer = f"{row['infer.img_per_s']:,.0f}" if "infer.img_per_s" in row else "-"
        train = f"{row['train.img_per_s']:,.0f}" if "train.img_per_s" in row else "-"
        levels = f"`{row['levels']}`" if row["levels"] else "-"
        lines.append(f"| {row['label']} | {schedule_text(row)} | {levels} | {row['params'] / 1e6:.2f} | {flops} | "
                     f"{infer} | {train} | {100 * row['top1']:.2f} | {100 * row['top5']:.2f} |")
    return "\n".join(lines)


# * ===== Figures =====

def new_axes(width: float = 7.2, height: float = 4.4):
    fig, ax = plt.subplots(figsize=(width, height), dpi=160)
    style_axes(fig, ax)
    return fig, ax


def style_axes(fig, ax) -> None:
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(TEXT_SECONDARY)
    ax.yaxis.label.set_color(TEXT_SECONDARY)


def save(fig, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / name, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {FIGURE_DIR / name}")


def is_multigrid(row: dict) -> bool:
    return row["levels"] is not None


# label offset in points (dx, dy) and horizontal alignment, chosen so that labels do not collide
LABEL_OFFSETS = {
    "bvit_4": (0, 9, "center"), "bvit_6": (-9, 7, "right"), "bvit_12": (0, 9, "center"),
    "fine_to_coarse": (9, 7, "left"), "w_cycle": (9, 3, "left"), "v_cycle": (9, -5, "left"),
    "coarse_to_fine": (9, 0, "left"), "w_cycle_deep": (0, -16, "center"),
}


def figure_accuracy_vs_flops(results: dict[str, dict]) -> None:
    """Accuracy against compute: multigrid models (blue) against the fixed-grid baselines (gray)."""
    fig, ax = new_axes()
    rows = {name: row for name, row in results.items() if row["params"] < 2e6}  # single shared block only
    baseline = sorted((row for row in rows.values() if not is_multigrid(row)), key=lambda row: row["gflops.total"])
    if len(baseline) > 1:
        ax.plot([row["gflops.total"] for row in baseline], [100 * row["top1"] for row in baseline],
                color=GRAY, linewidth=1.5, linestyle=(0, (4, 3)), zorder=1)
    for name, row in rows.items():
        multigrid = is_multigrid(row)
        ax.scatter(row["gflops.total"], 100 * row["top1"], s=60 if multigrid else 150,
                   color=BLUE if multigrid else GRAY, edgecolor=SURFACE, linewidth=2, zorder=3 if multigrid else 2)
        dx, dy, align = LABEL_OFFSETS.get(name, (0, 9, "center"))
        ax.annotate(f"{row['label'].replace('MG ', '')}  {100 * row['top1']:.2f}", (row["gflops.total"], 100 * row["top1"]),
                    xytext=(dx, dy), textcoords="offset points", ha=align, va="center", fontsize=8.5, color=TEXT)
    ax.scatter([], [], s=50, color=BLUE, label="multigrid loop (this work)")
    ax.scatter([], [], s=90, color=GRAY, label="bViT-S, full grid only")
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, loc="lower right")
    ax.set_xlabel("GFLOPs per image")
    ax.set_ylabel("CIFAR-100 top-1 (%)")
    ax.set_title("Accuracy vs. compute (all models: 1 shared block, 1.86M params)", color=TEXT, fontsize=11,
                 loc="left")
    ax.margins(x=0.12, y=0.12)
    save(fig, "accuracy_vs_flops.png")


def figure_schedule_order(results: dict[str, dict]) -> None:
    """Same budget (4F + 8C), different orders, with the same-FLOPs bViT as reference."""
    names = [name for name in ("bvit_6", "coarse_to_fine", "fine_to_coarse", "v_cycle", "w_cycle",
                               "w_cycle_no_level_embed") if name in results]
    if len(names) < 2:
        return
    fig, ax = new_axes(7.2, 0.55 * len(names) + 1.4)
    names = names[::-1]
    values = [100 * results[name]["top1"] for name in names]
    colors = [BLUE if is_multigrid(results[name]) else GRAY for name in names]
    bars = ax.barh([results[name]["label"] for name in names], values, color=colors, height=0.62)
    for bar, value in zip(bars, values):
        ax.text(value + 0.08, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", fontsize=9,
                color=TEXT)
    low = min(values) - 1.5
    ax.set_xlim(low, max(values) + 0.8)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_xlabel(f"CIFAR-100 top-1 (%)  -  axis starts at {low:.1f}")
    ax.set_title("Same compute (about 1.43 GFLOPs), different loop structure", color=TEXT, fontsize=11, loc="left")
    save(fig, "schedule_order.png")


def figure_per_step(results: dict[str, dict]) -> None:
    """Accuracy when inference stops after step t, against the compute spent so far."""
    chosen = [(name, color) for name, color in (("bvit_6", GRAY), ("fine_to_coarse", BLUE), ("v_cycle", ORANGE),
                                                ("w_cycle", AQUA)) if name in results]
    if len(chosen) < 2:
        return
    fig, ax = new_axes()
    for name, color in chosen:
        row = results[name]
        x, y = row["gflops_cumulative"], [100 * v for v in row["top1_per_step"]]
        ax.plot(x, y, color=color, linewidth=2, label=f"{row['label']}  (final {y[-1]:.1f})", zorder=2)
        coarse = [i for i, level in enumerate(row["step_levels"]) if level > 0]
        fine = [i for i, level in enumerate(row["step_levels"]) if level == 0]
        ax.scatter([x[i] for i in fine], [y[i] for i in fine], s=42, color=color, edgecolor=SURFACE, linewidth=1.5,
                   zorder=3)
        ax.scatter([x[i] for i in coarse], [y[i] for i in coarse], s=30, facecolor=SURFACE, edgecolor=color,
                   linewidth=1.5, zorder=3)
    ax.scatter([], [], s=42, color=TEXT_SECONDARY, label="step on the full grid (F)")
    ax.scatter([], [], s=30, facecolor=SURFACE, edgecolor=TEXT_SECONDARY, linewidth=1.5, label="step on a coarse grid (C)")
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, loc="lower right")
    ax.set_xlabel("GFLOPs spent before the classifier is applied")
    ax.set_ylabel("CIFAR-100 top-1 (%)")
    ax.set_title("Accuracy if the loop is stopped after each step", color=TEXT, fontsize=11, loc="left")
    save(fig, "accuracy_per_step.png")


def figure_training_curves(results: dict[str, dict]) -> None:
    chosen = [(name, color) for name, color in (("bvit_6", GRAY), ("fine_to_coarse", BLUE), ("w_cycle", ORANGE),
                                                ("coarse_to_fine", AQUA)) if name in results]
    if len(chosen) < 2:
        return
    fig, ax = new_axes()
    for name, color in chosen:
        row = results[name]
        curve = [100 * v for v in row["curve"]]
        ax.plot(range(1, len(curve) + 1), curve, color=color, linewidth=2,
                linestyle=(0, (4, 3)) if color == GRAY else "-", label=f"{row['label']}  (final {curve[-1]:.1f})")
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, loc="lower right")
    ax.set_xlabel("epoch")
    ax.set_ylabel("CIFAR-100 top-1 of the EMA model (%)")
    ax.set_title("Validation accuracy during training", color=TEXT, fontsize=11, loc="left")
    ax.margins(x=0.0)
    save(fig, "training_curves.png")


def figure_schedules() -> None:
    """Which grid every step runs on (does not need any training result)."""
    names = ["bvit_12", "bvit_6", "fine_to_coarse", "coarse_to_fine", "v_cycle", "w_cycle", "w_cycle_deep",
             "w_cycle_3level"]
    fig, axes = plt.subplots(len(names), 1, figsize=(7.2, 1.2 * len(names)), dpi=160, sharex=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, name in zip(axes, names):
        experiment = EXPERIMENTS[name]
        levels = step_levels(experiment)
        style_axes(fig, ax)
        ax.grid(visible=False)
        color = BLUE if experiment.levels else GRAY
        ax.step(range(len(levels) + 1), levels + levels[-1:], where="post", color=color, linewidth=2)
        ax.set_yticks(range(3), ["F 8x8", "C 4x4", "CC 2x2"], fontsize=7)
        ax.set_ylim(2.35, -0.35)  # fine grid on top, like a multigrid cycle diagram
        ax.set_xlim(0, 25)
        ax.set_title(f"{experiment.label}   (levels = {experiment.levels or 'none'})", loc="left", fontsize=8.5,
                     color=TEXT, pad=3)
    axes[-1].set_xlabel("step (one application of the shared block)")
    fig.subplots_adjust(hspace=0.75)
    save(fig, "schedules.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-throughput", action="store_true")
    args = parser.parse_args()

    results_path = OUTPUT_DIR / "results.json"
    previous = json.loads(results_path.read_text()) if results_path.exists() else {}
    results = collect(args.skip_throughput, previous)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2))
    table = markdown_table(results)
    (OUTPUT_DIR / "results.md").write_text(table + "\n")
    print(table)

    figure_schedules()
    figure_accuracy_vs_flops(results)
    figure_schedule_order(results)
    figure_per_step(results)
    figure_training_curves(results)


if __name__ == "__main__":
    main()
