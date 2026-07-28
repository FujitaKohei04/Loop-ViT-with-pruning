from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from .utils import ensure_dir, write_csv


class AnalysisRecorder:
    def __init__(self, output_dir: str | Path = "outputs") -> None:
        self.output_dir = Path(output_dir)
        self.activation_dir = ensure_dir(self.output_dir / "activation")
        self.gradient_dir = ensure_dir(self.output_dir / "gradients")
        self.attention_dir = ensure_dir(self.output_dir / "attention")
        self.weight_change_dir = ensure_dir(self.output_dir / "weight_change")

        self.activations: dict[tuple[int, str], torch.Tensor] = {}
        self.gradients: list[dict[str, Any]] = []
        self.attentions: dict[tuple[int, str], torch.Tensor] = {}
        self.weight_changes: list[dict[str, Any]] = []

    def add_activation(self, loop: int, layer: str, tensor: torch.Tensor) -> None:
        self.activations[(loop, layer)] = tensor.detach().cpu()

    def add_gradient(self, loop: int, layer: str, grad_norm: float, *, epoch: int | None = None) -> None:
        row = {"loop": int(loop), "layer": layer, "grad_norm": float(grad_norm)}
        if epoch is not None:
            row["epoch"] = int(epoch)
        self.gradients.append(row)

    def add_attention(self, loop: int, layer: str, tensor: torch.Tensor) -> None:
        self.attentions[(loop, layer)] = tensor.detach().cpu()

    def add_weight_change(self, layer: str, delta: float) -> None:
        self.weight_changes.append({"layer": layer, "delta": float(delta)})

    def save_epoch_gradients(self, epoch: int) -> None:
        rows = [row for row in self.gradients if row.get("epoch") == epoch]
        if rows:
            write_csv(self.gradient_dir / f"epoch{epoch}.csv", rows)

    def save_activations(self) -> None:
        grouped: dict[int, dict[str, torch.Tensor]] = defaultdict(dict)
        for (loop, layer), tensor in self.activations.items():
            grouped[loop][layer] = tensor

        for loop, tensors in sorted(grouped.items()):
            torch.save(tensors, self.activation_dir / f"loop{loop}.pt")

    def save_attentions(self) -> None:
        grouped: dict[int, dict[str, torch.Tensor]] = defaultdict(dict)
        for (loop, layer), tensor in self.attentions.items():
            grouped[loop][layer] = tensor

        for loop, tensors in sorted(grouped.items()):
            torch.save(tensors, self.attention_dir / f"loop{loop}.pt")

    def save_weight_changes(self) -> None:
        if self.weight_changes:
            write_csv(self.weight_change_dir / "delta.csv", self.weight_changes)

    def save(self) -> None:
        self.save_activations()
        self.save_attentions()
        self.save_weight_changes()
