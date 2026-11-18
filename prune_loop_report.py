"""Figures and tables for prune_loop_bvit.py results.

    uv run python prune_loop_report.py --run outputs/bvit_pruning_loop/<run> --figure-dir docs/loop_pruning/figures/<run>

Optional context from earlier experiments of the same checkpoint:
    --compute-run outputs/bvit_pruning_compute/<run>   the paper's spectral (low-rank) step-specific pruning as published

Writes <figure-dir>/*.png and <run>/summary.md.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from prune_compute_report import GRAY, SLOTS, SURFACE, legend_below, save  # noqa: E402  (also sets the rcParams)
from wandb_results import add_wandb_args, upload_results  # noqa: E402

# label, setting, method, color, marker, linestyle, hollow
MAIN = [
    ("Proposed: loop-aware importance against the step mean", "rot", "joint_mc", SLOTS[0], "o", "-", False),
    ("Proposed without mean replacement (plain removal)", "rot", "joint", SLOTS[0], "o", "--", True),
    ("bViT step-specific score, same units, equal ratio per step", "raw", "paper_step", SLOTS[1], "s", "-", False),
    ("bViT step-specific score + step-specific bias correction (rotated basis)", "rot", "paper_step+bc", SLOTS[1], "s", "--", True),
    ("Loop-unaware: one mask for all steps (best shared ranking)", "rot", "shared_joint", SLOTS[3], "v", "-", False),
    ("Random units, allocation of the proposed method", "rot", "random_alloc", SLOTS[4], "D", "-", False),
]
COMPUTE = [MAIN[0], MAIN[2], MAIN[4]]  # MAIN[0] needs rel_throughput (extra speed run)
ABLATION = [
    ("bViT score, equal ratio per step", "paper_step", SLOTS[1], "s", "-", False),
    ("Fisher, equal ratio per step", "fisher_step", SLOTS[7], "X", "-", False),
    ("+ one ranking over steps and layers (alloc)", "alloc", SLOTS[6], "^", "-", False),
    ("+ cross-step terms (joint)", "joint", SLOTS[2], "D", "-", False),
    ("+ importance against the step mean (joint_mc = proposed)", "joint_mc", SLOTS[0], "o", "-", False),
    ("variant: joint + re-scoring every 5% (joint_iter)", "joint_iter", SLOTS[5], "P", "--", True),
    ("variant: proposed + re-scoring (joint_iter_mc)", "joint_iter_mc", SLOTS[0], "o", "--", True),
]
# mean replacement: (panel title, setting, [(label, method, color, marker, linestyle, hollow)])
COMPENSATION = [
    ("Proposed ranking (rotated basis)", "rot", [
        ("Plain removal (joint_iter)", "joint_iter", SLOTS[2], "D", "--", True),
        ("Same masks + bias correction afterwards (joint_iter+bc)", "joint_iter+bc", SLOTS[6], "^", "-", False),
        ("Importance against the mean (joint_mc = proposed)", "joint_mc", SLOTS[0], "o", "-", False),
        ("Importance against the mean, re-scored (joint_iter_mc)", "joint_iter_mc", SLOTS[0], "o", "--", True)]),
    ("bViT step-specific score (channels as trained)", "raw", [
        ("Plain removal (paper_step)", "paper_step", SLOTS[1], "s", "-", False),
        ("Same masks + bias correction afterwards", "paper_step+bc", SLOTS[1], "s", "--", True),
        ("Score against the mean (paper_step_mc)", "paper_step_mc", SLOTS[7], "X", "-", False)]),
]
LAYERS = [("fc1", "FFN neurons"), ("q", "QK channels"), ("v", "V channels")]
WEIGHT = {"fc1": 4, "q": 1, "v": 1}  # units per layer relative to dim (mlp_ratio 4)


def rows_of(records: list[dict], setting: str, method: str) -> list[dict]:
    rows = [r for r in records if r["setting"] == setting and r["method"] == method]
    return sorted(rows, key=lambda r: -r["gflops.total"])


def reduction(row: dict, base: dict) -> float:
    return 100 * (1 - row["gflops.total"] / base["gflops.total"])


def pruning_rate(row: dict) -> float:
    """Removed fraction of all (step, unit) pairs."""
    kept = sum(WEIGHT[k] * statistics.mean(v) for k, v in row["kept_per_step"].items())
    return 100 * (1 - kept / sum(WEIGHT[k] for k in row["kept_per_step"]))


def style(color: str, marker: str, linestyle: str, hollow: bool) -> dict:
    extra = {"markerfacecolor": SURFACE, "markeredgecolor": color} if hollow else {}
    return {"color": color, "marker": marker, "linestyle": linestyle, **extra}


def unpruned_line(ax: plt.Axes, value: float, x: float) -> None:
    ax.axhline(value, color=GRAY, linewidth=1, linestyle=":")
    ax.annotate("unpruned", (x, value), xytext=(-2, 4), textcoords="offset points", ha="right", color=GRAY, fontsize=8)


def accuracy_figure(records, base, steps, spectral, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for label, setting, method, color, marker, linestyle, hollow in MAIN:
        rows = rows_of(records, setting, method)
        if rows:
            ax.plot([reduction(r, base) for r in rows], [100 * r["top1"] for r in rows],
                    label=label, **style(color, marker, linestyle, hollow))
    if spectral:
        ax.plot([reduction(r, base) for r in spectral], [100 * r["top1"] for r in spectral],
                label="bViT spectral step-specific as published (low-rank layers)", **style(SLOTS[2], "^", "-", False))
    if steps:
        ax.plot([reduction(r, base) for r in steps], [100 * r["top1"] for r in steps],
                label="Fewer loop steps (reference)", **style(SLOTS[5], "P", ":", False))
    unpruned_line(ax, 100 * base["top1"], 72)
    ax.set(xlabel="FLOPs reduction (%)", ylabel="Top-1 accuracy (%)", title=title, xlim=(-2, 74), ylim=(0, None))
    legend_below(fig, ax, columns=1)
    save(fig, path)


def ablation_figure(records, base, path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    for ax, (setting, name) in zip(axes, (("raw", "Channels as trained"), ("rot", "Heads rotated to the spectral basis"))):
        for label, method, color, marker, linestyle, hollow in ABLATION:
            rows = rows_of(records, setting, method)
            if rows:
                ax.plot([reduction(r, base) for r in rows], [100 * r["top1"] for r in rows],
                        label=label, **style(color, marker, linestyle, hollow))
        unpruned_line(ax, 100 * base["top1"], 72)
        ax.set(xlabel="FLOPs reduction (%)", title=name, xlim=(-2, 74), ylim=(0, None))
    axes[0].set_ylabel("Top-1 accuracy (%)")
    fig.suptitle(title, fontweight="bold")
    legend_below(fig, axes[0], columns=2)
    save(fig, path)


def compensation_figure(records, base, path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    for ax, (name, setting, series) in zip(axes, COMPENSATION):
        for label, method, color, marker, linestyle, hollow in series:
            rows = rows_of(records, setting, method)
            if rows:
                ax.plot([reduction(r, base) for r in rows], [100 * r["top1"] for r in rows],
                        label=label, **style(color, marker, linestyle, hollow))
        unpruned_line(ax, 100 * base["top1"], 72)
        ax.set(xlabel="FLOPs reduction (%)", title=name, xlim=(-2, 74), ylim=(0, None))
        ax.legend(loc="lower left", fontsize=8)
    axes[0].set_ylabel("Top-1 accuracy (%)")
    fig.suptitle(title, fontweight="bold")
    save(fig, path)


def compute_figure(records, base, steps, path: Path, title: str) -> None:
    """FLOPs and measured throughput against the pruning rate (removed fraction of (step, unit) pairs)."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    series = [(label, rows_of(records, setting, method), color, marker)
              for label, setting, method, color, marker, _, _ in COMPUTE]
    for label, rows, color, marker in series:
        rows = [r for r in rows if "rel_throughput" in r]
        if not rows:
            continue
        x = [pruning_rate(r) for r in rows]
        axes[0].plot(x, [r["gflops.total"] for r in rows], color=color, marker=marker, label=label)
        axes[1].plot(x, [r["rel_throughput"] for r in rows], color=color, marker=marker, label=label)
    if steps and "rel_throughput" in steps[0]:
        total = max(r["steps"] for r in steps) + 1
        x = [100 * (1 - r["steps"] / total) for r in steps]
        axes[0].plot(x, [r["gflops.total"] for r in steps], color=SLOTS[5], marker="P", linestyle=":",
                     label="Fewer loop steps (reference; rate = removed steps)")
        axes[1].plot(x, [r["rel_throughput"] for r in steps], color=SLOTS[5], marker="P", linestyle=":")
    unpruned_line(axes[0], base["gflops.total"], 75)
    unpruned_line(axes[1], 1.0, 75)
    axes[0].set(xlabel="Pruning rate: removed (step, unit) pairs (%)", ylabel="GFLOPs", title="FLOPs",
                xlim=(-2, 77), ylim=(0, None))
    axes[1].set(xlabel="Pruning rate: removed (step, unit) pairs (%)", ylabel="Throughput relative to unpruned",
                title="Measured throughput (batch 256, fp16)", xlim=(-2, 77))
    fig.suptitle(title, fontweight="bold")
    legend_below(fig, axes[0], columns=2)
    save(fig, path)


def param_reduction(row: dict, base_params: int) -> float:
    return 100 * (1 - row["stored_params"] / base_params)


def params_figure(records, base, steps, spectral, base_params: int, path: Path, title: str) -> None:
    """Stored parameters (prune_loop_params.py): accuracy against them, and how they follow the FLOPs."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    series = [(label, [r for r in rows_of(records, setting, method) if "stored_params" in r],
               style(color, marker, linestyle, hollow)) for label, setting, method, color, marker, linestyle, hollow in MAIN]
    series.append(("bViT spectral step-specific as published (low-rank layers)", spectral,
                   style(SLOTS[2], "^", "-", False)))
    series.append(("Fewer loop steps (reference)", [{**r, "stored_params": base_params} for r in steps],
                   style(SLOTS[5], "P", ":", False)))
    for label, rows, kwargs in series:
        if not rows:
            continue
        axes[0].plot([param_reduction(r, base_params) for r in rows], [100 * r["top1"] for r in rows],
                     label=label, **kwargs)
        axes[1].plot([reduction(r, base) for r in rows], [param_reduction(r, base_params) for r in rows], **kwargs)
    unpruned_line(axes[0], 100 * base["top1"], 72)
    axes[1].plot([0, 72], [0, 72], color=GRAY, linewidth=1, linestyle=":")
    axes[1].annotate("parameters fall as fast as FLOPs", (36, 36), xytext=(6, -14), textcoords="offset points",
                     rotation=33, color=GRAY, fontsize=8)
    axes[0].set(xlabel="Stored-parameter reduction (%)", ylabel="Top-1 accuracy (%)",
                title=f"Accuracy at equal parameters (unpruned {base_params / 1e6:.2f} M)", xlim=(-2, 74), ylim=(0, None))
    axes[1].set(xlabel="FLOPs reduction (%)", ylabel="Stored-parameter reduction (%)",
                title="Parameters against FLOPs", xlim=(-2, 74), ylim=(-2, 74))
    fig.suptitle(title, fontweight="bold")
    legend_below(fig, axes[0], columns=2)
    save(fig, path)


def params_table(records, base_params: int, targets: list[float]) -> str:
    out = (f"\nStored parameters in millions (unpruned {base_params / 1e6:.3f} M; fewer loop steps keep all of them; "
           "params = what the current implementation materializes):\n\n| setting | method | "
           + " | ".join(f"{t:.1f}" for t in targets) + " |\n|---|---|" + "---:|" * len(targets) + "\n")
    for setting, method in (("rot", "joint_mc"), ("rot", "joint"), ("raw", "paper_step"), ("raw", "paper_step+bc"),
                            ("rot", "shared_joint")):
        rows = {r["target"]: r for r in records if r["setting"] == setting and r["method"] == method and "stored_params" in r}
        if rows:
            out += f"| {setting} | {method} (stored) | " + " | ".join(
                f"{rows[t]['stored_params'] / 1e6:.3f}" if t in rows else "-" for t in targets) + " |\n"
            out += f"| {setting} | {method} (params) | " + " | ".join(
                f"{rows[t]['params'] / 1e6:.2f}" if t in rows else "-" for t in targets) + " |\n"
    return out


def speed_accuracy_figure(records, base, steps, cpu: dict | None, path: Path, title: str) -> None:
    """Equal-accuracy comparison: measured speed-up against top-1 (GPU fp16 batch 256; CPU fp32 if measured)."""
    panels = [("GPU (RTX 2080 Ti, fp16, batch 256)", "rel_throughput")]
    if cpu:
        panels.append((f"CPU ({cpu['threads']} threads, fp32, batch {cpu['batch']})", "rel_throughput_cpu"))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 4.3), squeeze=False)
    cpu_rows = {(r["method"], r.get("target", r.get("steps"))): r["rel_throughput_cpu"] for r in (cpu or {}).get("records", [])}
    for ax, (name, key) in zip(axes[0], panels):
        series = [("Proposed (rot, joint_mc)", rows_of(records, "rot", "joint_mc"), "target", "joint_mc", SLOTS[0], "o", "-"),
                  ("Fewer loop steps (reference)", steps, "steps", "steps", SLOTS[5], "P", ":")]
        if key == "rel_throughput":
            series.insert(1, ("bViT step-specific score, same units", rows_of(records, "raw", "paper_step"), "target",
                              "paper_step", SLOTS[1], "s", "-"))
        for label, rows, field, method, color, marker, linestyle in series:
            pts = []
            for r in rows:
                value = r.get(key) if key == "rel_throughput" else cpu_rows.get((method, r[field]))
                if value is not None:
                    pts.append((100 * r["top1"], value))
            if pts:
                pts.sort()
                ax.plot(*zip(*pts), color=color, marker=marker, linestyle=linestyle, label=label)
        ax.axhline(1.0, color=GRAY, linewidth=0.8, linestyle=":")
        ax.axvline(100 * base["top1"], color=GRAY, linewidth=0.8, linestyle=":")
        ax.set(xlabel="Top-1 accuracy (%), no retraining", ylabel="Throughput relative to unpruned", title=name)
    fig.suptitle(title, fontweight="bold")
    legend_below(fig, axes[0][0], columns=3)
    save(fig, path)


def allocation_figure(records, path: Path, title: str, setting: str = "rot", method: str = "joint_mc") -> None:
    """Kept fraction of every layer in every step, one line per FLOPs target (light = mild, dark = strong)."""
    rows = sorted([r for r in records if r["setting"] == setting and r["method"] == method], key=lambda r: -r["target"])
    if not rows:
        return
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    blues = plt.get_cmap("Blues")
    for ax, (key, name) in zip(axes, LAYERS):
        for i, row in enumerate(rows):
            kept = row["kept_per_step"][key]
            ax.plot(range(1, len(kept) + 1), [100 * k for k in kept], color=blues(0.3 + 0.7 * i / max(len(rows) - 1, 1)),
                    marker="o", markersize=4, label=f"FLOPs -{100 * (1 - row['target']):.0f}%")
        ax.set(xlabel="Loop step", title=name, ylim=(-3, 103), xticks=range(1, len(kept) + 1))
    axes[0].set_ylabel("Kept units (%)")
    fig.suptitle(title, fontweight="bold")
    legend_below(fig, axes[0], columns=7)
    save(fig, path)


def reuse_figure(masks: dict, path: Path, title: str, target: float, setting: str = "rot",
                 method: str = "joint_mc") -> None:
    """In how many of the loop steps is a unit used? (the bViT paper: some directions 9-12 times, some 1-4)"""
    key = (setting, method, target)
    if key not in masks:
        return
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (short, name) in zip(axes, LAYERS):
        mask = next(m for n, m in masks[key]["masks"].items() if n.endswith("." + short))
        uses = mask.sum(dim=0)
        counts = torch.bincount(uses, minlength=mask.shape[0] + 1)
        ax.bar(range(len(counts)), (100 * counts / counts.sum()).tolist(), color=SLOTS[0], width=0.8)
        ax.set(xlabel="Steps in which the unit is kept", title=name, xticks=range(0, len(counts), 2))
    axes[0].set_ylabel("Units (%)")
    fig.suptitle(title, fontweight="bold", y=1.03)
    save(fig, path)


def finetune_figure(finetune: dict, path: Path, title: str) -> None:
    names = {"joint_mc": ("Proposed", SLOTS[0], "o"),
             "joint_iter_mc": ("Proposed + re-scoring", SLOTS[6], "^"),
             "paper_step": ("bViT step-specific score, same units", SLOTS[1], "s"),
             "paper_step+bc": ("bViT score + bias correction", SLOTS[7], "X"),
             "steps": ("Fewer loop steps", SLOTS[5], "P"), "shared_joint": ("Loop-unaware shared mask", SLOTS[3], "v")}
    base = finetune["unpruned"]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for method, (label, color, marker) in names.items():
        rows = sorted([r for r in finetune["records"] if r["method"] == method], key=lambda r: -r["gflops.total"])
        if not rows:
            continue
        x = [reduction(r, base) for r in rows]
        ax.plot(x, [100 * r["top1"] for r in rows], color=color, marker=marker, label=f"{label}, fine-tuned")
        ax.plot(x, [100 * r["top1_before"] for r in rows], color=color, marker=marker, linestyle="--",
                markerfacecolor=SURFACE, markeredgecolor=color, label=f"{label}, no fine-tuning")
    unpruned_line(ax, 100 * base["top1"], 78)
    control = [r for r in finetune["records"] if r["method"] == "unpruned"]
    if control:
        ax.scatter([0], [100 * control[0]["top1"]], color=GRAY, marker="*", s=120, zorder=3,
                   label="Unpruned, same fine-tuning")
    ax.set(xlabel="FLOPs reduction (%)", ylabel="Top-1 accuracy (%)", title=title, xlim=(-3, 80), ylim=(0, None))
    legend_below(fig, ax, columns=2)
    save(fig, path)


# * ===== Tables =====

def table(records, base, steps, spectral, seeds: list[list[dict]], targets: list[float]) -> str:
    def cell(setting: str, method: str, target: float) -> str:
        values = [100 * r["top1"] for rs in [records, *seeds] for r in rs
                  if (r["setting"], r["method"], r["target"]) == (setting, method, target)]
        if not values:
            return "-"
        if len(values) == 1:
            return f"{values[0]:.1f}"
        return f"{statistics.mean(values):.1f} ± {statistics.stdev(values):.1f}"

    lines = [f"Unpruned: top-1 {100 * base['top1']:.2f}%, {base['gflops.total']:.3f} GFLOPs. "
             f"Cells: top-1 (%) at the FLOPs target; mean ± std over {1 + len(seeds)} calibration sets where available.", "",
             "| setting | method | " + " | ".join(f"-{100 * (1 - t):.0f}%" for t in targets) + " |",
             "|---|---|" + "---:|" * len(targets)]
    methods = ["paper_step", "paper_shared", "fisher_step", "shared_joint", "alloc", "joint", "alloc_iter",
               "joint_iter", "joint_iter_floor", "random_alloc", "paper_step+bc", "paper_step_mc", "joint_iter+bc",
               "joint_mc", "alloc_iter_mc", "joint_iter_mc"]
    for setting in ("raw", "rot"):
        for method in methods:
            if rows_of(records, setting, method):
                lines.append(f"| {setting} | {method} | " + " | ".join(cell(setting, method, t) for t in targets) + " |")
    lines += ["", "Reference points (not at the exact targets):", "", "| method | setting | FLOPs reduction | top-1 (%) | rel. throughput |",
              "|---|---|---:|---:|---:|"]
    for r in steps:
        lines.append(f"| fewer steps | {r['steps']} steps | -{reduction(r, base):.1f}% | {100 * r['top1']:.1f} | "
                     f"{r.get('rel_throughput', float('nan')):.2f} |")
    for r in spectral:
        lines.append(f"| bViT spectral step-specific (low-rank) | ratio {r['ratio']:.1f} | "
                     f"{-reduction(r, base):+.1f}% | {100 * r['top1']:.1f} | {r['rel_throughput']:.2f} |")
    lines += ["", "Throughput relative to the unpruned model (measured, batch 256):", "",
              "| setting | method | " + " | ".join(f"-{100 * (1 - t):.0f}%" for t in targets) + " |",
              "|---|---|" + "---:|" * len(targets)]
    for setting in ("raw", "rot"):
        for method in methods:
            rows = {r["target"]: r for r in rows_of(records, setting, method) if "rel_throughput" in r}
            if rows:
                lines.append(f"| {setting} | {method} | " + " | ".join(
                    f"{rows[t]['rel_throughput']:.2f}" if t in rows else "-" for t in targets) + " |")
    return "\n".join(lines) + "\n"


def within(records, base, setting: str, method: str, drop: float) -> str:
    """Largest FLOPs reduction whose top-1 stays within `drop` points of the unpruned model."""
    ok = [r for r in rows_of(records, setting, method) if 100 * (base["top1"] - r["top1"]) <= drop]
    if not ok:
        return "-"
    best = min(ok, key=lambda r: r["gflops.total"])
    return f"-{reduction(best, base):.0f}% ({100 * best['top1']:.1f}%)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--figure-dir", required=True)
    parser.add_argument("--compute-run", default=None)
    parser.add_argument("--reuse-target", type=float, default=0.5)
    add_wandb_args(parser)
    args = parser.parse_args()
    run, figures = Path(args.run), Path(args.figure_dir)
    figures.mkdir(parents=True, exist_ok=True)
    results = json.loads((run / "results.json").read_text())
    records, base = results["records"], results["unpruned"]
    name = f"{results['model']} / {results['dataset']}"
    steps = sorted([r for r in records if r["method"] == "steps"], key=lambda r: -r["steps"])
    records = [r for r in records if r["method"] != "steps"]
    if (run / "results_mc.json").exists():
        records += json.loads((run / "results_mc.json").read_text())["records"]
    seeds = [json.loads(p.read_text())["records"]
             + (json.loads((run / f"results_mc_{p.stem.split('_')[-1]}.json").read_text())["records"]
                if (run / f"results_mc_{p.stem.split('_')[-1]}.json").exists() else [])
             for p in sorted(run.glob("results_seed??.json"))]
    spectral = []
    if args.compute_run:
        compute = json.loads((Path(args.compute_run) / "results.json").read_text())["records"]
        spectral = sorted([r for r in compute if (r["family"], r["group"], r["method"]) == ("spectral", "ALL", "spectral_step")
                           and 0 < r["ratio"] < 1], key=lambda r: r["ratio"])
    targets = sorted({r["target"] for r in records}, reverse=True)
    params = json.loads((run / "params.json").read_text()) if (run / "params.json").exists() else None
    if params:  # from prune_loop_params.py
        counts = {(r["setting"], r["method"], r["target"]): r for r in params["records"]}
        for r in records:
            if (r["setting"], r["method"], r["target"]) in counts:
                c = counts[r["setting"], r["method"], r["target"]]
                r.update(stored_params=c["stored_params"], params=c["params"])

    accuracy_figure(records, base, steps, spectral, figures / "accuracy_vs_flops.png",
                    f"{name}: accuracy at equal FLOPs, no retraining")
    ablation_figure(records, base, figures / "ablation.png", f"{name}: what each component adds (no retraining)")
    compensation_figure(records, base, figures / "compensation.png",
                        f"{name}: replacing removed units by their step mean (no retraining)")
    compute_figure(records, base, steps, figures / "compute_vs_rate.png", f"{name}: compute against the pruning rate")
    allocation_figure(records, figures / "allocation.png", f"{name}: where the proposed method prunes")
    cpu = json.loads((run / "cpu_speed.json").read_text()) if (run / "cpu_speed.json").exists() else None
    speed_accuracy_figure(records, base, steps, cpu, figures / "speed_vs_accuracy.png",
                          f"{name}: measured speed at equal accuracy")
    if (run / "masks_mc.pt").exists():
        reuse_figure(torch.load(run / "masks_mc.pt"), figures / "unit_reuse.png",
                     f"{name}: steps that use a unit, proposed method at FLOPs -{100 * (1 - args.reuse_target):.0f}%",
                     args.reuse_target)
    if (run / "finetune.json").exists():
        finetune_figure(json.loads((run / "finetune.json").read_text()), figures / "finetune.png",
                        f"{name}: masks fixed, shared weights fine-tuned")

    if params:
        params_figure(records, base, steps, spectral, params["unpruned"]["stored_params"], figures / "params.png",
                      f"{name}: stored parameters, no retraining")

    summary = table(records, base, steps, spectral, seeds, targets)
    if params:
        summary += params_table(records, params["unpruned"]["stored_params"], targets)
    summary += "\nLargest FLOPs reduction within an accuracy drop (no retraining; top-1 in brackets):\n\n"
    summary += "| setting | method | <= 1 pt | <= 2 pt | <= 5 pt |\n|---|---|---|---|---|\n"
    for setting, method in (("rot", "joint_mc"), ("rot", "joint_iter_mc"), ("raw", "joint_mc"), ("rot", "joint_iter"), ("raw", "joint_iter"),
                            ("raw", "paper_step+bc"), ("raw", "paper_step"), ("rot", "paper_step"), ("rot", "shared_joint")):
        summary += f"| {setting} | {method} | " + " | ".join(within(records, base, setting, method, d) for d in (1, 2, 5)) + " |\n"
    for path in sorted(run.glob("results_round*.json")):
        extra = json.loads(path.read_text())["records"]
        summary += f"\n{path.name} (widths not rounded to multiples of 8):\n\n| setting | method | target | top-1 (%) | GFLOPs | rel. throughput |\n|---|---|---:|---:|---:|---:|\n"
        for r in extra:
            if r["method"] != "steps":
                summary += (f"| {r['setting']} | {r['method']} | {r['target']} | {100 * r['top1']:.1f} | {r['gflops.total']:.3f} | "
                            f"{r.get('rel_throughput', float('nan')):.2f} |\n")
    if (run / "finetune.json").exists():
        finetune = json.loads((run / "finetune.json").read_text())
        summary += "\nFine-tuning (masks fixed):\n\n| setting | method | target | GFLOPs | top-1 before | top-1 after |\n|---|---|---:|---:|---:|---:|\n"
        for r in finetune["records"]:
            summary += (f"| {r['setting']} | {r['method']} | {r['target']} | {r['gflops.total']:.3f} | "
                        f"{100 * r['top1_before']:.1f} | {100 * r['top1']:.1f} |\n")
    (run / "summary.md").write_text(summary)
    print(summary)
    upload_results(args, job_type="prune_loop_report", group=run.name,
                   files=[run / "summary.md", *([run / "params.json"] if params else [])])


if __name__ == "__main__":
    main()
