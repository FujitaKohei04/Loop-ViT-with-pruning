from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .utils import ensure_dir, write_csv


class ActivationAnalyzer:
    def __init__(self, activation_root: str | Path) -> None:
        self.activation_root = Path(activation_root)

    def _load_tensors(self, epoch: int, loop_idx: int) -> list[tuple[str, torch.Tensor]]:
        loop_dir = self.activation_root / f"epoch_{epoch}" / f"loop_{loop_idx}"
        if not loop_dir.exists():
            return []
        tensors = []
        for path in sorted(loop_dir.glob("*.pt")):
            tensors.append((path.stem, torch.load(path, map_location="cpu")))
        return tensors

    def compute_norm(self, epoch: int, loop_idx: int) -> float:
        tensors = self._load_tensors(epoch, loop_idx)
        if not tensors:
            return 0.0
        norms = []
        for _, tensor in tensors:
            if tensor.ndim >= 2:
                norms.append(torch.linalg.norm(tensor, dim=-1).mean().item())
        return float(sum(norms) / len(norms)) if norms else 0.0

    def compute_mean(self, epoch: int, loop_idx: int) -> float:
        tensors = self._load_tensors(epoch, loop_idx)
        if not tensors:
            return 0.0
        values = [tensor.mean().item() for _, tensor in tensors if tensor.numel() > 0]
        return float(sum(values) / len(values)) if values else 0.0

    def compute_std(self, epoch: int, loop_idx: int) -> float:
        tensors = self._load_tensors(epoch, loop_idx)
        if not tensors:
            return 0.0
        values = [tensor.std().item() for _, tensor in tensors if tensor.numel() > 0]
        return float(sum(values) / len(values)) if values else 0.0

    def compute_cosine(self, epoch: int, loop_idx_a: int, loop_idx_b: int) -> float:
        tensors_a = self._load_tensors(epoch, loop_idx_a)
        tensors_b = self._load_tensors(epoch, loop_idx_b)
        if not tensors_a or not tensors_b:
            return 0.0

        flat_a = torch.cat([tensor.flatten().float() for _, tensor in tensors_a if tensor.numel() > 0])
        flat_b = torch.cat([tensor.flatten().float() for _, tensor in tensors_b if tensor.numel() > 0])
        if flat_a.numel() == 0 or flat_b.numel() == 0:
            return 0.0
        return float(torch.nn.functional.cosine_similarity(flat_a, flat_b, dim=0).item())

    def summarize(self, epoch: int) -> list[dict[str, Any]]:
        rows = []
        for loop_idx in sorted({int(path.parent.name.split("_")[-1]) for path in (self.activation_root / f"epoch_{epoch}").glob("loop_*") if path.is_dir()}):
            rows.append({
                "epoch": epoch,
                "loop": loop_idx,
                "norm": self.compute_norm(epoch, loop_idx),
                "mean": self.compute_mean(epoch, loop_idx),
                "std": self.compute_std(epoch, loop_idx),
            })
        return rows


def save_summary_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(Path(path).parent)
    write_csv(path, rows)
