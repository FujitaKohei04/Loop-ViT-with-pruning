from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .utils import ensure_dir


def plot_attention(attention: torch.Tensor, output_path: str | Path) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    if attention.ndim == 4:
        attn = attention[0, 0].detach().cpu().numpy()
    else:
        attn = attention.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(attn, cmap="viridis")
    ax.set_title("Attention")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
