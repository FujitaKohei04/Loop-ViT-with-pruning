from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

from .utils import ensure_dir


class ActivationRecorder:
    """Save activation tensors by epoch, loop, and layer name."""

    def __init__(self, output_dir: str | Path = "outputs/activation") -> None:
        self.output_dir = Path(output_dir)

    def _safe_name(self, name: str) -> str:
        return name.replace("/", "_").replace(".", "_")

    def record(self, layer_name: str, loop_idx: int, activation: torch.Tensor, *, epoch: int) -> None:
        save_dir = self.output_dir / f"epoch_{epoch}" / f"loop_{loop_idx}"
        ensure_dir(save_dir)
        torch.save(activation.detach().cpu(), save_dir / f"{self._safe_name(layer_name)}.pt")

    def save(self, epoch: int) -> Path:
        return ensure_dir(self.output_dir / f"epoch_{epoch}")


def summarize_activation(tensor: torch.Tensor) -> dict[str, float]:
    tensor = tensor.detach().float()
    if tensor.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "l2": 0.0}
    return {
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std().item()),
        "l2": float(torch.linalg.norm(tensor).item()),
    }


def compute_activation_stats(activations: Iterable[dict[str, torch.Tensor]]) -> list[dict[str, object]]:
    rows = []
    for idx, layer_map in enumerate(activations):
        for layer_name, tensor in layer_map.items():
            summary = summarize_activation(tensor)
            rows.append({"loop": idx, "layer": layer_name, **summary})
    return rows
