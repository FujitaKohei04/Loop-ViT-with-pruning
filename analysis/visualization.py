from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .utils import ensure_dir


def plot_neuron_heatmap(csv_path: str | Path, output_path: str | Path) -> None:
    df = pd.read_csv(csv_path)
    ensure_dir(Path(output_path).parent)
    pivot = df.pivot(index="loop", columns="neuron", values="value")
    plt.figure(figsize=(8, 3))
    plt.imshow(pivot.values, aspect="auto", cmap="viridis")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel("neuron")
    plt.ylabel("loop")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_summary(csv_path: str | Path, output_path: str | Path, metric: str = "mean") -> None:
    df = pd.read_csv(csv_path)
    ensure_dir(Path(output_path).parent)
    plt.figure(figsize=(6, 4))
    for name in sorted(df["name"].unique()):
        subset = df[df["name"] == name]
        plt.plot(subset["loop"], subset[metric], marker="o", label=name)
    plt.xlabel("loop")
    plt.ylabel(metric)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
