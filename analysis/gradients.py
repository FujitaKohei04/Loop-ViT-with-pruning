from __future__ import annotations

from typing import List

import torch
from torch import nn

from .recorder import AnalysisRecorder


def register_gradient_hooks(model: nn.Module, recorder: AnalysisRecorder) -> List[object]:
    handles = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        def make_hook(param_name: str):
            def hook(grad: torch.Tensor) -> torch.Tensor:
                if grad is None:
                    return None
                loop = getattr(model, "current_loop", -1)
                epoch = getattr(model, "current_epoch", None)
                recorder.add_gradient(int(loop), param_name, grad.detach().norm().item(), epoch=epoch)
                return grad

            return hook

        handles.append(parameter.register_hook(make_hook(name)))

    return handles
