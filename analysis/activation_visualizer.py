from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .utils import ensure_dir


def save_norm_plot(csv_path: str | Path, output_path: str | Path) -> None:
    df = pd.read_csv(csv_path)
    ensure_dir(Path(output_path).parent)
    plt.figure(figsize=(6, 4))
    for loop_id in sorted(df["loop"].unique()):
        subset = df[df["loop"] == loop_id]
        plt.plot(subset["epoch"], subset["norm"], marker="o", label=f"loop {loop_id}")
    plt.xlabel("epoch")
    plt.ylabel("norm")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_cosine_plot(csv_path: str | Path, output_path: str | Path) -> None:
    df = pd.read_csv(csv_path)
    ensure_dir(Path(output_path).parent)
    matrix = df.pivot(index="loop_a", columns="loop_b", values="cosine")
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix.values, cmap="viridis")
    plt.xticks(range(len(matrix.columns)), matrix.columns)
    plt.yticks(range(len(matrix.index)), matrix.index)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
