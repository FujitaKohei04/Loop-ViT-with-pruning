"""Fewer loop steps combined with the proposed pruning (rot / joint_mc on the model that stops after T steps).

    uv run python prune_loop_combine_report.py --run outputs/bvit_pruning_loop/bvit_s_tiny_imagenet \\
        --figure-dir docs/loop_pruning/figures/bvit_s_tiny_imagenet

Reads <run>/results.json (fewer steps only), <run>/results_mc.json (pruning only) and <run>_steps<T>/results_mc.json
(+ cpu_speed.json) written by prune_loop_bvit.py --max-steps T and prune_loop_speed.py --max-steps T.
Writes combine.png and prints a markdown table.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from prune_loop_report import GRAY, SLOTS, legend_below, plt, save
from wandb_results import add_wandb_args, upload_results


def load(run: Path) -> tuple[dict, list[dict], dict[int, list[dict]]]:
    results = json.loads((run / "results.json").read_text())
    steps_only = sorted([r for r in results["records"] if r["method"] == "steps"], key=lambda r: -r["steps"])

    def pruned(directory: Path) -> list[dict]:
        rows = [r for r in json.loads((directory / "results_mc.json").read_text())["records"]
                if (r["setting"], r["method"]) == ("rot", "joint_mc")]
        cpu_path = directory / "cpu_speed.json"
        if cpu_path.exists():
            cpu = {r.get("target"): r["rel_throughput_cpu"] for r in json.loads(cpu_path.read_text())["records"]
                   if r["method"] == "joint_mc"}
            for r in rows:
                r["rel_throughput_cpu"] = cpu.get(r["target"])
        return sorted(rows, key=lambda r: -r["target"])

    total = max(r["steps"] for r in steps_only) + 1
    combined = {total: pruned(run)}
    for directory in sorted(run.parent.glob(run.name + "_steps*")):
        combined[int(directory.name.rsplit("steps", 1)[1])] = pruned(directory)
    cpu: dict[int, list[float]] = {}  # fewer steps only is re-measured in every speed run: keep the median
    for cpu_path in [run / "cpu_speed.json", *sorted(run.parent.glob(run.name + "_steps*/cpu_speed.json"))]:
        if cpu_path.exists():
            for r in json.loads(cpu_path.read_text())["records"]:
                if r["method"] == "steps":
                    cpu.setdefault(r["steps"], []).append(r["rel_throughput_cpu"])
    for r in steps_only:
        if r["steps"] in cpu:
            r["rel_throughput_cpu"] = statistics.median(cpu[r["steps"]])
    return results["unpruned"], steps_only, dict(sorted(combined.items(), reverse=True))


def figure(base: dict, steps_only: list[dict], combined: dict[int, list[dict]], path: Path, title: str) -> None:
    panels = [("FLOPs reduction (%)", lambda r: 100 * (1 - r["gflops.total"] / base["gflops.total"])),
              ("GPU throughput relative to unpruned (fp16, batch 256)", lambda r: r.get("rel_throughput"))]
    if any(r.get("rel_throughput_cpu") for rows in combined.values() for r in rows):
        panels.append(("CPU throughput relative to unpruned (fp32, batch 32)", lambda r: r.get("rel_throughput_cpu")))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.4), sharey=True, squeeze=False)
    blues = plt.get_cmap("Blues")
    total = max(combined)
    for ax, (xlabel, x_of) in zip(axes[0], panels):
        for i, (steps, rows) in enumerate(combined.items()):
            pts = [(x_of(r), 100 * r["top1"]) for r in rows if x_of(r) is not None]
            if pts:
                label = f"Proposed pruning only ({total} steps)" if steps == total else f"{steps} steps + proposed pruning"
                ax.plot(*zip(*pts), color=blues(1.0 - 0.75 * i / max(len(combined) - 1, 1)), marker="o", markersize=5,
                        linewidth=2.4 if steps == total else 1.6, label=label)
        pts = [(x_of(r), 100 * r["top1"]) for r in steps_only if x_of(r) is not None and r["top1"] > 0.25]
        if pts:
            ax.plot(*zip(*pts), color=SLOTS[5], marker="P", linestyle=":", label="Fewer loop steps only")
        ax.axhline(100 * base["top1"], color=GRAY, linewidth=0.8, linestyle=":")
        ax.set(xlabel=xlabel, ylim=(30, None))
    axes[0][0].set_ylabel("Top-1 accuracy (%), no retraining")
    fig.suptitle(title, fontweight="bold")
    legend_below(fig, axes[0][0], columns=4)
    save(fig, path)


def table(base: dict, steps_only: list[dict], combined: dict[int, list[dict]]) -> str:
    targets = sorted({r["target"] for rows in combined.values() for r in rows}, reverse=True)
    lines = ["top-1 (%) at equal FLOPs; brackets: GPU throughput relative to the unpruned model", "",
             "| loop steps | no pruning | " + " | ".join(f"-{100 * (1 - t):.0f}%" for t in targets) + " |",
             "|---|---:|" + "---:|" * len(targets)]
    plain = {r["steps"]: r for r in steps_only}
    for steps, rows in combined.items():
        by_target = {r["target"]: r for r in rows}
        first = f"{100 * plain[steps]['top1']:.1f} ({plain[steps]['rel_throughput']:.2f})" if steps in plain else \
            f"{100 * base['top1']:.1f} (1.00)"
        lines.append(f"| {steps} | {first} | " + " | ".join(
            f"{100 * by_target[t]['top1']:.1f} ({by_target[t]['rel_throughput']:.2f})" if t in by_target else "-"
            for t in targets) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--figure-dir", required=True)
    add_wandb_args(parser)
    args = parser.parse_args()
    run = Path(args.run)
    base, steps_only, combined = load(run)
    Path(args.figure_dir).mkdir(parents=True, exist_ok=True)
    figure(base, steps_only, combined, Path(args.figure_dir) / "combine.png",
           f"{run.name}: fewer loop steps combined with the proposed pruning")
    print(table(base, steps_only, combined))
    upload_results(args, job_type="prune_loop_combine", group=run.name,
                   files=sorted(run.parent.glob(f"{run.name}_steps*/results_mc.json")))


if __name__ == "__main__":
    main()
