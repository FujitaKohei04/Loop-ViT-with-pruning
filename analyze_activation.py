from __future__ import annotations

import argparse
from pathlib import Path

from analysis.activation_metrics import ActivationAnalyzer, save_summary_csv
from analysis.activation_visualizer import save_cosine_plot, save_norm_plot
from analysis.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze saved SReT activations")
    parser.add_argument("--activation-root", default="outputs/activation")
    parser.add_argument("--output-dir", default="outputs/analysis")
    parser.add_argument("--figures-dir", default="figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    activation_root = Path(args.activation_root)
    output_dir = Path(args.output_dir)
    figures_dir = Path(args.figures_dir)
    ensure_dir(output_dir)
    ensure_dir(figures_dir)

    analyzer = ActivationAnalyzer(activation_root)
    epochs = sorted([int(path.name.split("_")[-1]) for path in activation_root.glob("epoch_*") if path.is_dir()])

    norm_rows = []
    mean_rows = []
    std_rows = []
    cosine_rows = []

    for epoch in epochs:
        summary = analyzer.summarize(epoch)
        for row in summary:
            norm_rows.append({"epoch": epoch, "loop": row["loop"], "norm": row["norm"]})
            mean_rows.append({"epoch": epoch, "loop": row["loop"], "mean": row["mean"]})
            std_rows.append({"epoch": epoch, "loop": row["loop"], "std": row["std"]})

        loops = sorted({row["loop"] for row in summary})
        for loop_a in loops:
            for loop_b in loops:
                cosine_rows.append({
                    "epoch": epoch,
                    "loop_a": loop_a,
                    "loop_b": loop_b,
                    "cosine": analyzer.compute_cosine(epoch, loop_a, loop_b),
                })

    save_summary_csv(output_dir / "norm.csv", norm_rows)
    save_summary_csv(output_dir / "mean.csv", mean_rows)
    save_summary_csv(output_dir / "std.csv", std_rows)
    save_summary_csv(output_dir / "cosine.csv", cosine_rows)

    save_norm_plot(output_dir / "norm.csv", figures_dir / "norm.png")
    save_cosine_plot(output_dir / "cosine.csv", figures_dir / "cosine.png")

    print(f"Saved activation analysis to {output_dir}")


if __name__ == "__main__":
    main()
