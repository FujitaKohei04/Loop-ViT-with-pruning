"""Turn pruning masks into a model that is really smaller.

PrunableLinear in "mask" mode rebuilds a full-size weight, so the pruned model
costs exactly as much as the original one. compact_model() returns a copy in
which every masked layer is replaced by the cheapest equivalent layer:

    spectral  rank-k factorization  in -> k -> out   (U_k, S_k V_k^T)
              only when k (in + out) < in * out; otherwise the dense matrix is kept
    weight    dense layer with zeros (no speed-up without a sparse kernel)
    neuron    Mlp with the kept hidden neurons only
    head      attention with the kept heads only

Step-conditioned masks give one variant per distinct mask; StepSwitch picks the
variant of the running step through LoopContext. The compact model computes the
same function as the masked model (checked by max_output_difference).

    pruner.apply(masks)
    small = compact_model(model)          # model still has the pruner attached
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn

from models.bViT.schedule import LoopContext
from pruning.base import PrunableLinear
from pruning.units import HeadUnit, NeuronUnit, SpectralUnit, WeightUnit


class StepSwitch(nn.Module):
    """One module per distinct mask; `index[use]` says which one a loop step uses.

    stored_params: parameters that must be stored when the variants are sliced
    out of one shared matrix at load time (union of the kept units), as opposed
    to sum(p.numel()), which counts every variant separately."""

    def __init__(self, variants: list[nn.Module], index: list[int], loop_ctx: LoopContext | None,
                 stored_params: int) -> None:
        super().__init__()
        self.variants = nn.ModuleList(variants)
        self.index = index
        self.loop_ctx = loop_ctx
        self.stored_params = stored_params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        use = self.loop_ctx.use_index if self.loop_ctx is not None else 0
        return self.variants[self.index[use]](x)


class CompactAttention(nn.Module):
    """models.bViT.bViT.Attention with only the kept heads (head_dim unchanged)."""

    def __init__(self, attn: nn.Module, proj: nn.Linear, keep: torch.Tensor) -> None:
        super().__init__()
        dim = proj.in_features
        head_dim = dim // attn.num_heads
        self.num_heads = int(keep.sum())
        self.head_dim = head_dim
        rows = keep.repeat_interleave(head_dim).to(proj.weight.device)
        self.q, self.k, self.v = (_slice_linear(layer, rows=rows) for layer in (attn.q, attn.k, attn.v))
        self.proj = _slice_linear(proj, cols=rows)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        if self.num_heads == 0:  # every head removed: only the proj bias is left
            return self.proj(x.new_zeros(B, N, 0))
        q, k, v = (layer(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
                   for layer in (self.q, self.k, self.v))
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim))


class CompactMlp(nn.Module):
    def __init__(self, fc1: nn.Linear, act: nn.Module, fc2: nn.Linear, keep: torch.Tensor) -> None:
        super().__init__()
        keep = keep.to(fc1.weight.device)
        self.fc1 = _slice_linear(fc1, rows=keep)
        self.act = act
        self.fc2 = _slice_linear(fc2, cols=keep)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def _slice_linear(linear: nn.Linear, rows: torch.Tensor | None = None, cols: torch.Tensor | None = None
                  ) -> nn.Linear:
    weight, bias = linear.weight.detach(), linear.bias
    if rows is not None:
        weight = weight[rows]
        bias = None if bias is None else bias.detach()[rows]
    if cols is not None:
        weight = weight[:, cols]
    return _make_linear(weight, bias)


def _make_linear(weight: torch.Tensor, bias: torch.Tensor | None) -> nn.Linear:
    layer = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None,
                      device=weight.device, dtype=weight.dtype)
    with torch.no_grad():
        layer.weight.copy_(weight)
        if bias is not None:
            layer.bias.copy_(bias)
    return layer


def _low_rank(unit: SpectralUnit, keep: torch.Tensor) -> nn.Module:
    """Rank-k layer for the kept directions, or the dense matrix if that is cheaper."""
    keep = keep.to(unit.S.device)
    out_features, in_features = unit.U.shape[0], unit.Vh.shape[1]
    k = int(keep.sum())
    if k * (in_features + out_features) >= in_features * out_features:
        return _make_linear(unit.effective_weight(keep.float()), unit.bias)
    down = _make_linear(unit.S[keep, None] * unit.Vh[keep], None)  # in -> k
    up = _make_linear(unit.U[:, keep], unit.bias)                   # k -> out
    return nn.Sequential(down, up)


def _switch(masks: torch.Tensor, build, loop_ctx: LoopContext | None, stored_params: int) -> StepSwitch:
    """Deduplicate the per-use masks and build one variant per distinct mask."""
    distinct, index = torch.unique(masks.to(torch.uint8), dim=0, return_inverse=True)
    return StepSwitch([build(mask.bool()) for mask in distinct], index.tolist(), loop_ctx, stored_params)


def _parent_and_child(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    parent_name, _, child = name.rpartition(".")
    return (model.get_submodule(parent_name) if parent_name else model), child


@torch.no_grad()
def compact_model(model: nn.Module) -> nn.Module:
    """Copy of `model` in which every PrunableLinear in mask mode is replaced by a smaller layer."""
    small = copy.deepcopy(model)
    wrappers = {name: module for name, module in small.named_modules() if isinstance(module, PrunableLinear)}
    if any(wrapper.mode != "mask" for wrapper in wrappers.values()):
        raise RuntimeError("compact_model needs applied masks (call pruner.apply first)")
    loop_ctx = getattr(small, "loop_ctx", None)

    # structured units replace the parent module, so they go last
    order = sorted(wrappers, key=lambda name: isinstance(wrappers[name].unit, (NeuronUnit, HeadUnit)))
    for name in order:
        wrapper = wrappers[name]
        unit, linear, masks = wrapper.unit, wrapper.linear, wrapper.masks.cpu()
        union = masks.any(dim=0)
        parent, child = _parent_and_child(small, name)
        bias_params = 0 if linear.bias is None else linear.bias.numel()
        if isinstance(unit, SpectralUnit):
            stored = min(int(union.sum()) * (linear.in_features + linear.out_features), linear.weight.numel())
            setattr(parent, child, _switch(masks, lambda mask: _low_rank(unit, mask), loop_ctx,
                                           stored + bias_params))
        elif isinstance(unit, WeightUnit):
            setattr(parent, child, _switch(
                masks, lambda mask: _make_linear(unit.effective_weight(mask.float().to(linear.weight.device)),
                                                 linear.bias),
                loop_ctx, linear.weight.numel() + bias_params))
        elif isinstance(unit, NeuronUnit):
            fc2 = parent.fc2
            if not isinstance(fc2, nn.Linear):
                raise RuntimeError(f"{name}: fc2 must be a plain Linear to remove neurons")
            stored = int(union.sum()) * (linear.in_features + 1 + fc2.out_features) + fc2.bias.numel()
            grand, parent_child = _parent_and_child(small, name.rpartition(".")[0])
            setattr(grand, parent_child, _switch(
                masks, lambda mask: CompactMlp(linear, parent.act, fc2, mask), loop_ctx, stored))
        elif isinstance(unit, HeadUnit):
            if not all(isinstance(layer, nn.Linear) for layer in (parent.q, parent.k, parent.v)):
                raise RuntimeError(f"{name}: q / k / v must be plain Linears to remove heads")
            stored = (int(union.sum()) * unit.head_dim * (3 * (linear.in_features + 1) + linear.out_features)
                      + bias_params)
            grand, parent_child = _parent_and_child(small, name.rpartition(".")[0])
            setattr(grand, parent_child, _switch(
                masks, lambda mask: CompactAttention(parent, linear, mask), loop_ctx, stored))
        else:
            raise NotImplementedError(f"no compaction for unit {type(unit).__name__}")
    return small


def stored_params(model: nn.Module) -> int:
    """Parameters to store when step variants are slices of shared matrices."""
    total, switched = 0, set()
    for module in model.modules():
        if isinstance(module, StepSwitch):
            total += module.stored_params
            switched.update(id(p) for p in module.parameters())
    return total + sum(p.numel() for p in model.parameters() if id(p) not in switched)


@torch.no_grad()
def max_output_difference(a: nn.Module, b: nn.Module, x: torch.Tensor) -> float:
    """max |a(x) - b(x)| relative to max |a(x)| (fp32)."""
    a.eval()
    b.eval()
    out_a, out_b = a(x).float(), b(x).float()
    return float((out_a - out_b).abs().max() / out_a.abs().max().clamp_min(1e-12))
