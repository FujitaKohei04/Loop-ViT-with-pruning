from __future__ import annotations

import argparse
from pathlib import Path

import torch

from analysis.mlp_activation_analyzer import MLPActivationAnalyzer
from analysis.utils import ensure_dir

try:
    from analysis.visualization import plot_neuron_heatmap, plot_summary
except Exception:
    plot_neuron_heatmap = None
    plot_summary = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze MLP activations collected during SReT inference")
    parser.add_argument("--input", default="outputs/mlp_activation.pt")
    parser.add_argument("--output-dir", default="outputs/mlp_analysis")
    parser.add_argument("--figures-dir", default="figures/mlp")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    figures_dir = Path(args.figures_dir)
    ensure_dir(output_dir)
    ensure_dir(figures_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"MLP activation data not found: {input_path}")

    analyzer = MLPActivationAnalyzer()
    payload = torch.load(input_path, map_location="cpu", weights_only=False)
    analyzer.consume_payload(payload)

    analyzer.save_summary_csv(output_dir / "summary.csv")
    analyzer.save_neuron_csv(output_dir / "neuron_profile.csv")

    if plot_summary is not None and plot_neuron_heatmap is not None:
        plot_summary(output_dir / "summary.csv", figures_dir / "summary.png")
        plot_neuron_heatmap(output_dir / "neuron_profile.csv", figures_dir / "neuron_heatmap.png")

    print(f"Saved MLP activation analysis to {output_dir}")


if __name__ == "__main__":
    main()
