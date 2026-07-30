from __future__ import annotations

import torch

from .recorder import AnalysisRecorder


def compute_weight_change(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor], recorder: AnalysisRecorder) -> None:
    for name, before_tensor in before.items():
        if name not in after:
            continue
        delta = (after[name].detach().float() - before_tensor.detach().float()).norm().item()
        recorder.add_weight_change(name, delta)
