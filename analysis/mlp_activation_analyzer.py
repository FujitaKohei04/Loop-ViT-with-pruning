from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .utils import ensure_dir, write_csv


class MLPActivationAnalyzer:
    def __init__(self) -> None:
        self.data: dict[int, dict[str, list[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))
        self.stats: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)

    def save(self, loop: int, name: str, tensor: torch.Tensor) -> None:
        self.data[int(loop)][name].append(tensor.detach().cpu())

    def accumulate(self, loop: int, name: str, tensor: torch.Tensor) -> None:
        loop_key = int(loop)
        name_stats = self.stats[loop_key].setdefault(
            name,
            {"count": 0, "sum": 0.0, "sum_sq": 0.0, "neuron_sum": None, "neuron_count": 0},
        )

        flat = tensor.detach().reshape(-1).float()
        name_stats["count"] += int(flat.numel())
        name_stats["sum"] += float(flat.sum().item())
        name_stats["sum_sq"] += float(torch.square(flat).sum().item())

        profile = tensor.detach().reshape(-1).float()

        if name_stats["neuron_sum"] is None:
            name_stats["neuron_sum"] = profile.detach().cpu()
            name_stats["neuron_count"] = 1
        elif profile.numel() == name_stats["neuron_sum"].numel() and name_stats["neuron_sum"].numel() == profile.numel():
            name_stats["neuron_sum"] = name_stats["neuron_sum"] + profile.detach().cpu()
            name_stats["neuron_count"] += 1

    def consume_payload(self, payload: dict[int, dict[str, list[torch.Tensor]]]) -> None:
        for loop, groups in payload.items():
            for name, tensors in groups.items():
                for tensor in tensors:
                    self.accumulate(loop, name, tensor)

    def compute_stats(self, loop: int, name: str) -> dict[str, float]:
        entry = self.stats.get(int(loop), {}).get(name)
        if entry is None or entry["count"] == 0:
            return {"mean": 0.0, "std": 0.0, "norm": 0.0}

        mean = entry["sum"] / entry["count"]
        variance = max(0.0, (entry["sum_sq"] / entry["count"]) - (mean * mean))
        std = float(torch.sqrt(torch.tensor(variance, dtype=torch.float32)).item())
        norm = float(torch.sqrt(torch.tensor(entry["sum_sq"], dtype=torch.float32)).item())
        return {"mean": float(mean), "std": std, "norm": norm}

    def compute_neuron_profile(self, loop: int, name: str) -> torch.Tensor | None:
        entry = self.stats.get(int(loop), {}).get(name)
        if entry is None or entry["neuron_sum"] is None or entry["neuron_count"] == 0:
            return None
        return entry["neuron_sum"] / entry["neuron_count"]

    def save_summary_csv(self, output_path: str | Path) -> None:
        rows: list[dict[str, Any]] = []
        for loop in sorted(self.stats.keys()):
            for name in sorted(self.stats[loop].keys()):
                stats = self.compute_stats(loop, name)
                rows.append({"loop": loop, "name": name, **stats})
        write_csv(output_path, rows)

    def save_neuron_csv(self, output_path: str | Path) -> None:
        rows: list[dict[str, Any]] = []
        for loop in sorted(self.stats.keys()):
            profile = self.compute_neuron_profile(loop, "gelu_output")
            if profile is None:
                continue
            for idx, value in enumerate(profile.tolist()):
                rows.append({"loop": loop, "neuron": idx, "value": value})
        write_csv(output_path, rows)
