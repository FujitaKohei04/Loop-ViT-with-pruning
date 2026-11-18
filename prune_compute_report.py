"""Figures and tables for prune_compute_bvit.py results.

    uv run python prune_compute_report.py --run outputs/bvit_pruning_compute/<run> \
        --figure-dir docs/pruning_compute/figures/<run>

Writes <figure-dir>/*.png and <run>/summary.md (markdown tables used by docs/pruning_compute/).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

SURFACE, TEXT, TEXT_SECONDARY, GRID, GRAY = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8a8985"
# categorical slots in their fixed order; every series keeps its color and marker in all figures
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

# one representative per family: (label, family, group, method, color, marker)
MAIN = [
    ("Spectral, all 6 proj. (paper, step-cond.)", "spectral", "ALL", "spectral_step", SLOTS[0], "o"),
    ("Unstructured Taylor, all 6 proj.", "unstructured", "ALL", "taylor", SLOTS[1], "s"),
    ("FFN neurons (Taylor)", "neuron", "FFN_IN", "neuron_taylor", SLOTS[2], "^"),
    ("Attention heads (Taylor)", "head", "PROJ", "head_taylor", SLOTS[3], "v"),
    ("Heads + neurons (Taylor)", "width", "PROJ+FFN_IN", "width_taylor", SLOTS[4], "D"),
    ("Fewer loop steps", "steps", "-", "steps", SLOTS[5], "P"),
    ("ToMe (token merging)", "tome", "-", "tome", SLOTS[6], "X"),
    ("EViT (token pruning)", "evit", "-", "evit", SLOTS[7], "*"),
]
SPECTRAL_METHODS = [("Step-specific", "spectral_step", SLOTS[0], "o", "-"),
                    ("Global", "spectral_global", SLOTS[1], "s", "--"),
                    ("Permuted", "spectral_permuted", SLOTS[2], "^", "-."),
                    ("Largest-sigma", "spectral_sigma", SLOTS[3], "P", ":")]
# shared mask vs step-conditioned mask of the same unit: (title, family, group, shared method, step method)
STEP_PAIRS = [
    ("Spectral, all 6 proj.", "spectral", "ALL", "spectral_global", "spectral_step"),
    ("Unstructured, all 6 proj.", "unstructured", "ALL", "taylor", "taylor_step"),
    ("FFN neurons", "neuron", "FFN_IN", "neuron_taylor", "neuron_taylor_step"),
    ("Attention heads", "head", "PROJ", "head_taylor", "head_taylor_step"),
    ("Heads + neurons", "width", "PROJ+FFN_IN", "width_taylor", "width_taylor_step"),
]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_SECONDARY, "text.color": TEXT,
    "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 2,
    "lines.markersize": 7, "lines.markeredgecolor": SURFACE, "lines.markeredgewidth": 1,
})


def select(records: list[dict], family: str, group: str, method: str) -> list[dict]:
    rows = [r for r in records if (r["family"], r["group"], r["method"]) == (family, group, method)]
    return sorted(rows, key=lambda r: r["rate"])


def percent(rows: list[dict]) -> list[float]:
    return [100 * r["rate"] for r in rows]


def noise_band(records: list[dict]) -> tuple[float, float]:
    """5th / 95th percentile of the relative throughput of models that cost exactly as much as the
    unpruned one (rate 0, or unstructured masks): how far a measurement moves without any real change."""
    control = sorted(r["rel_throughput"] for r in records if r["rate"] == 0 or r["family"] == "unstructured")
    if len(control) < 5:
        return 1.0, 1.0
    return control[round(0.05 * (len(control) - 1))], control[round(0.95 * (len(control) - 1))]


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {path}")


def legend_below(fig: plt.Figure, ax: plt.Axes, columns: int = 2) -> None:
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=columns, fontsize=9)


# * ===== Figures =====

def main_figure(records: list[dict], base: dict, key: str, ylabel: str, title: str, path: Path, *,
                log: bool = False, with_nnz: bool = False) -> None:
    """One metric against the pruning rate, one line per family."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    scale = 100.0 if key == "top1" else 1.0
    if key == "rel_throughput":
        ax.axhspan(*noise_band(records), color=GRAY, alpha=0.18, linewidth=0,
                   label="Measurement noise (5-95% of same-cost models)")
    for label, family, group, method, color, marker in MAIN:
        rows = select(records, family, group, method)
        if not rows:
            continue
        ax.plot(percent(rows), [scale * r[key] for r in rows], color=color, marker=marker, label=label)
        if with_nnz and family == "unstructured":
            # same color, dashed + hollow markers: the cost if only non-zero weights were computed
            nnz = [r["gflops.total"] - r["gflops.linear_only"] + r["gflops.linear_nnz"] for r in rows]
            ax.plot(percent(rows), nnz, color=color, marker=marker, linestyle="--", markerfacecolor=SURFACE,
                    markeredgecolor=color, label="Unstructured, non-zero weights only (needs a sparse kernel)")
    reference = {"gflops.total": base["gflops.total"], "rel_throughput": 1.0, "top1": 100 * base["top1"]}[key]
    ax.axhline(reference, color=GRAY, linewidth=1, linestyle=":")
    ax.annotate("unpruned", (100, reference), xytext=(-2, 4), textcoords="offset points", ha="right",
                fontsize=8, color=TEXT_SECONDARY)
    if log:
        ax.set_yscale("log", base=2)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}x"))
    else:
        ax.set_ylim(bottom=0)
    ax.set_xlim(-2, 102)
    ax.set_xlabel("Pruning rate (%)  [removed units of each method]")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    legend_below(fig, ax)
    save(fig, path)


def tradeoff_figure(records: list[dict], base: dict, path: Path) -> None:
    """Accuracy against cost: the comparison that matters. Two panels, one x-axis each."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    panels = [("gflops.total", "GFLOPs (total)  <- cheaper", base["gflops.total"]),
              ("rel_throughput", "Throughput relative to unpruned  -> faster", 1.0)]
    for ax, (key, xlabel, reference) in zip(axes, panels):
        for label, family, group, method, color, marker in MAIN:
            rows = select(records, family, group, method)
            if family == "unstructured" and key == "gflops.total":
                continue  # dense FLOPs do not change: a vertical line that hides the others
            if rows:
                ax.plot([r[key] for r in rows], [100 * r["top1"] for r in rows], color=color, marker=marker,
                        label=label)
        ax.plot([reference], [100 * base["top1"]], marker="o", markersize=11, markerfacecolor="none",
                markeredgecolor=TEXT, markeredgewidth=1.5, linestyle="none", label="Unpruned")
        ax.set_xlabel(xlabel)
        if key == "rel_throughput":
            ax.set_xscale("log", base=2)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}x"))
        else:
            ax.invert_xaxis()
    axes[0].set_ylabel("Top-1 accuracy (%)")
    axes[0].set_title("Accuracy vs FLOPs (no retraining)", loc="left")
    axes[1].set_title("Accuracy vs measured throughput", loc="left")
    legend_below(fig, axes[1], columns=4)
    save(fig, path)


def paper_figure(records: list[dict], base: dict, path: Path) -> None:
    """Paper Fig. 6 (top row) with the cost of the same pruned models below it."""
    groups = ["Q", "V", "FFN_IN", "ALL"]
    rows_spec = [("top1", "Top-1 accuracy (%)", 100.0), ("gflops.total", "GFLOPs (total)", 1.0),
                 ("rel_throughput", "Relative throughput", 1.0)]
    fig, axes = plt.subplots(3, 4, figsize=(15, 9.5), sharex=True, sharey="row")
    for col, group in enumerate(groups):
        for row, (key, ylabel, scale) in enumerate(rows_spec):
            ax = axes[row, col]
            # the cost depends only on the mask shape: step-specific = permuted, global = largest-sigma
            methods = SPECTRAL_METHODS if key == "top1" else SPECTRAL_METHODS[:2]
            for label, method, color, marker, style in methods:
                rows = select(records, "spectral", group, method)
                if rows:
                    ax.plot(percent(rows), [scale * r[key] for r in rows], color=color, marker=marker,
                            linestyle=style, label=label)
            reference = {"top1": 100 * base["top1"], "gflops.total": base["gflops.total"], "rel_throughput": 1.0}[key]
            ax.axhline(reference, color=GRAY, linewidth=1, linestyle=":")
            if key == "rel_throughput":
                ax.axhspan(*noise_band(records), color=GRAY, alpha=0.18, linewidth=0)
            if row == 0:
                ax.set_title("All six projections" if group == "ALL" else group, loc="left")
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 2:
                ax.set_xlabel("Pruning ratio (%)")
                if col == 0:
                    ax.annotate("gray band: measurement noise", (0.03, 0.93), xycoords="axes fraction",
                                fontsize=8, color=TEXT_SECONDARY)
            if key != "rel_throughput":
                ax.set_ylim(bottom=0)
    axes[0, 3].legend(fontsize=9)
    axes[1, 3].legend(fontsize=9, title="same cost: step-specific = permuted,\nglobal = largest-sigma",
                      title_fontsize=8)
    fig.tight_layout()
    save(fig, path)


def step_figure(records: list[dict], base: dict, path: Path) -> None:
    """Does a step-conditioned mask beat one shared mask for the same unit?"""
    fig, axes = plt.subplots(1, len(STEP_PAIRS), figsize=(3.6 * len(STEP_PAIRS), 3.9), sharey=True)
    for ax, (title, family, group, shared, step) in zip(axes, STEP_PAIRS):
        for label, method, color, marker, style in (("Shared mask (all steps)", shared, SLOTS[1], "s", "--"),
                                                    ("Step-conditioned mask", step, SLOTS[0], "o", "-")):
            rows = select(records, family, group, method)
            if rows:
                ax.plot(percent(rows), [100 * r["top1"] for r in rows], color=color, marker=marker,
                        linestyle=style, label=label)
        ax.axhline(100 * base["top1"], color=GRAY, linewidth=1, linestyle=":")
        ax.set_title(title, loc="left")
        ax.set_xlabel("Pruning rate (%)")
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Top-1 accuracy (%)")
    legend_below(fig, axes[0])
    save(fig, path)


def speed_vs_flops_figure(records: list[dict], base: dict, path: Path) -> None:
    """How well does the FLOPs reduction predict the measured speed-up?"""
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ideal = [x / 32 for x in range(2, 33)]
    ax.plot(ideal, [1 / x for x in ideal], color=GRAY, linewidth=1, linestyle=":", label="Throughput = 1 / FLOPs")
    for label, family, group, method, color, marker in MAIN:
        rows = select(records, family, group, method)
        rows = [r for r in rows if r["gflops.total"] / base["gflops.total"] >= 1 / 16]
        if rows and family != "unstructured":
            ax.plot([r["gflops.total"] / base["gflops.total"] for r in rows], [r["rel_throughput"] for r in rows],
                    color=color, marker=marker, label=label)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.invert_xaxis()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: "1" if value >= 1 else f"1/{1 / value:g}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}x"))
    ax.set_xlabel("FLOPs relative to unpruned  <- cheaper")
    ax.set_ylabel("Measured throughput relative to unpruned")
    ax.set_title("Measured speed-up vs FLOPs reduction", loc="left")
    legend_below(fig, ax)
    save(fig, path)


# * ===== Tables =====

def method_keys(records: list[dict]) -> list[tuple[str, str, str]]:
    return list(dict.fromkeys((r["family"], r["group"], r["method"]) for r in records))


def best_within(rows: list[dict], base: dict, drop: float) -> dict | None:
    """Lowest-FLOPs config whose top-1 loss is at most `drop` (ties: the highest rate). FLOPs, not the
    measured throughput, pick the row: throughput noise would otherwise select outliers."""
    ok = [r for r in rows if base["top1"] - r["top1"] <= drop and r["rate"] > 0]
    return min(ok, key=lambda r: (round(r["gflops.total"], 4), -r["rate"])) if ok else None


def summary_tables(results: dict) -> str:
    records, base = results["records"], results["unpruned"]
    low, high = noise_band(records)
    lines = [f"unpruned: top-1 {100 * base['top1']:.2f}%, {base['gflops.total']:.3f} GFLOPs "
             f"(Linear-only {base['gflops.linear_only']:.3f}), {base['img_per_s']:.0f} img/s, "
             f"latency {base['latency_ms']:.2f} ms, {base['params'] / 1e6:.2f}M params", "",
             "throughput noise (5-95% of the relative throughput of same-cost models): "
             f"{low:.2f}x - {high:.2f}x", ""]

    for drop in (0.01, 0.03):
        lines += [f"### Lowest-FLOPs setting with a top-1 loss of at most {100 * drop:.0f} pt", "",
                  "| family | target | method | rate | top-1 | GFLOPs | FLOPs | throughput | latency (bs 1) | stored params |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for family, group, method in method_keys(records):
            best = best_within(select(records, family, group, method), base, drop)
            if best is None:
                lines.append(f"| {family} | {group} | {method} | - | - | - | - | - | - | - |")
                continue
            lines.append(
                f"| {family} | {group} | {method} | {100 * best['rate']:.0f}% | {100 * best['top1']:.2f}% | "
                f"{best['gflops.total']:.3f} | {100 * (best['gflops.total'] / base['gflops.total'] - 1):+.0f}% | "
                f"{best['rel_throughput']:.2f}x | {best['rel_latency']:.2f}x | {best['stored_params'] / 1e6:.2f}M |")
        lines.append("")

    lines += ["### All settings", "",
              "| family | target | method | setting | rate | top-1 | GFLOPs | Linear-only | non-zero only | "
              "throughput | img/s | latency (bs 1) | stored params | materialized params |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for family, group, method in method_keys(records):
        for r in select(records, family, group, method):
            lines.append(
                f"| {family} | {group} | {method} | {r['ratio']} | {100 * r['rate']:.1f}% | {100 * r['top1']:.2f}% | "
                f"{r['gflops.total']:.3f} | {r['gflops.linear_only']:.3f} | {r['gflops.linear_nnz']:.3f} | "
                f"{r['rel_throughput']:.2f}x | {r['rel_throughput'] * base['img_per_s']:.0f} | "
                f"{r['rel_latency']:.2f}x | {r['stored_params'] / 1e6:.2f}M | {r['params'] / 1e6:.2f}M |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="directory with results.json")
    parser.add_argument("--figure-dir", default=None, help="default: <run>/figures")
    args = parser.parse_args()
    run_dir = Path(args.run)
    results = json.loads((run_dir / "results.json").read_text())
    records, base = results["records"], results["unpruned"]
    figure_dir = Path(args.figure_dir) if args.figure_dir else run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    name = f"{results['model']} / {results['dataset']}"

    main_figure(records, base, "gflops.total", "GFLOPs (total, per image)",
                f"FLOPs vs pruning rate ({name})", figure_dir / "flops_vs_rate.png", with_nnz=True)
    main_figure(records, base, "rel_throughput", "Throughput relative to unpruned (GPU, fp16)",
                f"Measured throughput vs pruning rate ({name})", figure_dir / "throughput_vs_rate.png", log=True)
    main_figure(records, base, "top1", "Top-1 accuracy (%)",
                f"Accuracy vs pruning rate, no retraining ({name})", figure_dir / "accuracy_vs_rate.png")
    tradeoff_figure(records, base, figure_dir / "tradeoff.png")
    if select(records, "spectral", "Q", "spectral_step"):
        paper_figure(records, base, figure_dir / "paper_spectral.png")
    step_figure(records, base, figure_dir / "step_vs_shared.png")
    speed_vs_flops_figure(records, base, figure_dir / "throughput_vs_flops.png")
    (run_dir / "summary.md").write_text(summary_tables(results))
    print(f"saved: {run_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
