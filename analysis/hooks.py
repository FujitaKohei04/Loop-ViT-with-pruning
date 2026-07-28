from __future__ import annotations

from typing import List

from torch import nn

from .recorder import AnalysisRecorder
from .activation import ActivationRecorder

try:
    from SReT import Group_Attention, Mlp, Transformer_Block
except Exception:  # pragma: no cover - import will be resolved at runtime
    Group_Attention = Mlp = Transformer_Block = nn.Module


def register_activation_hooks(model: nn.Module, recorder: AnalysisRecorder, activation_recorder: ActivationRecorder | None = None) -> list:
    handles = []

    def make_hook(layer_name: str):
        def hook(_module: nn.Module, _inputs, output):
            loop = getattr(model, "current_loop", -1)
            epoch = getattr(model, "current_epoch", 0)
            if activation_recorder is not None:
                activation_recorder.record(layer_name, int(loop), output, epoch=epoch)
            recorder.add_activation(int(loop), layer_name, output)

        return hook

    for name, module in model.named_modules():
        if isinstance(module, (Transformer_Block, Mlp)):
            handles.append(module.register_forward_hook(make_hook(name)))

    return handles


def register_attention_hooks(model: nn.Module, recorder: AnalysisRecorder) -> list:
    handles = []

    def make_hook(layer_name: str):
        def hook(_module: nn.Module, _inputs, _output):
            loop = getattr(model, "current_loop", -1)
            attention = getattr(_module, "last_attention", None)
            if attention is not None:
                recorder.add_attention(int(loop), layer_name, attention)

        return hook

    for name, module in model.named_modules():
        if isinstance(module, Group_Attention):
            handles.append(module.register_forward_hook(make_hook(name)))

    return handles


def remove_hooks(handles: List[object]) -> None:
    for handle in handles:
        handle.remove()
