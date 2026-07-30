from __future__ import annotations

from torch import nn

from .activation import ActivationRecorder
from .recorder import AnalysisRecorder

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


def register_mlp_activation_hooks(model: nn.Module, analyzer) -> list:
    handles = []

    def make_input_hook(layer_name: str):
        def hook(_module: nn.Module, inputs, _outputs):
            loop = getattr(model, "current_loop", -1)
            analyzer.save(int(loop), f"{layer_name}_input", inputs[0])

        return hook

    def make_output_hook(layer_name: str):
        def hook(_module: nn.Module, _inputs, outputs):
            loop = getattr(model, "current_loop", -1)
            analyzer.save(int(loop), f"{layer_name}_output", outputs)

        return hook

    for name, module in model.named_modules():
        if isinstance(module, Mlp):
            handles.append(module.fc1.register_forward_hook(make_input_hook("fc1")))
            handles.append(module.fc2.register_forward_hook(make_output_hook("fc2")))
            if hasattr(module, "act"):
                handles.append(module.act.register_forward_hook(make_output_hook("gelu")))

    return handles


def remove_hooks(handles: list[object]) -> None:
    for handle in handles:
        handle.remove()
