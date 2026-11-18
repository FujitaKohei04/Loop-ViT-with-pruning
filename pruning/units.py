"""Pruning units: what a gate / mask entry removes.

    spectral  singular direction of W        (bViT paper)
    weight    single weight                  (unstructured)
    neuron    FFN hidden neuron, on mlp.fc1  (structured: Molchanov et al., CVPR 2019)
    head      attention head, on attn.proj   (structured: Michel et al., NeurIPS 2019)

Structured units remove whole rows / columns, so pruning/compact.py can turn
their masks into smaller layers.

To add a unit, subclass PruningUnit and decorate it with @register_unit("name").
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

UNITS: dict[str, type[PruningUnit]] = {}


def register_unit(name: str):
    def decorator(cls):
        UNITS[name] = cls
        return cls
    return decorator


class PruningUnit(nn.Module):
    """forward(x, gate): the layer output with every unit scaled by gate (num_units,).
    effective_weight(gate): the equivalent (out, in) weight matrix.
    static_scores(): data-free scores this unit offers (e.g. {"sigma": ...})."""

    num_units: int

    @classmethod
    def build(cls, linear: nn.Linear, parent: nn.Module) -> PruningUnit:
        """parent = the module that owns the Linear (structured units read it)."""
        return cls(linear)

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def effective_weight(self, gate: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def effective_bias(self, gate: torch.Tensor) -> torch.Tensor | None:
        """None = the bias is not changed by the gate."""
        return None

    def static_scores(self) -> dict[str, torch.Tensor]:
        return {}


@register_unit("spectral")
class SpectralUnit(PruningUnit):
    """Singular directions of W = U diag(sigma) V^T (paper Eq. 3-4, 7).
    d loss / d gate_i = sigma_i u_i^T G v_i (Eq. 5)."""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        U, S, Vh = torch.linalg.svd(linear.weight.detach().float(), full_matrices=False)
        self.register_buffer("U", U)
        self.register_buffer("S", S)
        self.register_buffer("Vh", Vh)
        self.bias = linear.bias
        self.num_units = S.numel()

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        h = F.linear(x, self.Vh)
        h = h * (self.S * gate).to(h.dtype)
        return F.linear(h, self.U, self.bias)

    def effective_weight(self, gate: torch.Tensor) -> torch.Tensor:
        return (self.U * (self.S * gate)) @ self.Vh

    def static_scores(self) -> dict[str, torch.Tensor]:
        return {"sigma": self.S}


@register_unit("weight")
class WeightUnit(PruningUnit):
    """Individual weights (unstructured). d loss / d gate_ij = w_ij g_ij, so the
    grad_gate scorer gives the first-order Taylor score |w g| used in main.py."""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias
        self.num_units = linear.weight.numel()

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * gate.view_as(self.weight).to(self.weight.dtype), self.bias)

    def effective_weight(self, gate: torch.Tensor) -> torch.Tensor:
        return self.weight.detach().float() * gate.view_as(self.weight)

    def static_scores(self) -> dict[str, torch.Tensor]:
        return {"magnitude": self.weight.detach().abs().flatten().float()}


@register_unit("neuron")
class NeuronUnit(PruningUnit):
    """Hidden neurons of the FFN. Attach to mlp.fc1: gate_i scales the i-th output
    (weight row and bias), so a closed gate gives GELU(0) = 0 and the neuron is
    removed exactly (fc2 never sees it)."""

    def __init__(self, linear: nn.Linear, out_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias
        self.out_weight = out_weight  # fc2 weight (dim, hidden), only for the magnitude score
        self.num_units = linear.out_features

    @classmethod
    def build(cls, linear: nn.Linear, parent: nn.Module) -> NeuronUnit:
        fc2 = getattr(parent, "fc2", None)
        fc2 = getattr(fc2, "linear", fc2)  # fc2 may be wrapped by another pruner
        return cls(linear, fc2.weight if isinstance(fc2, nn.Linear) else None)

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        h = F.linear(x, self.weight, self.bias)
        return h * gate.to(h.dtype)

    def effective_weight(self, gate: torch.Tensor) -> torch.Tensor:
        return self.weight.detach().float() * gate[:, None]

    def effective_bias(self, gate: torch.Tensor) -> torch.Tensor | None:
        return None if self.bias is None else self.bias.detach().float() * gate

    def static_scores(self) -> dict[str, torch.Tensor]:
        """magnitude: ||fc1 row|| * ||fc2 column|| ([A] product of the two norms)."""
        score = self.weight.detach().float().norm(dim=1)
        if self.out_weight is not None:
            score = score * self.out_weight.detach().float().norm(dim=0)
        return {"magnitude": score}


@register_unit("head")
class HeadUnit(PruningUnit):
    """Attention heads. Attach to attn.proj: gate_h scales the output of head h
    (the h-th slice of the proj input). A closed gate makes the q/k/v rows of
    that head unused."""

    def __init__(self, linear: nn.Linear, num_heads: int) -> None:
        super().__init__()
        if linear.in_features % num_heads != 0:
            raise ValueError(f"in_features {linear.in_features} is not divisible by num_heads {num_heads}")
        self.weight = linear.weight
        self.bias = linear.bias
        self.num_units = num_heads
        self.head_dim = linear.in_features // num_heads

    @classmethod
    def build(cls, linear: nn.Linear, parent: nn.Module) -> HeadUnit:
        if not hasattr(parent, "num_heads"):
            raise ValueError("the head unit must be attached to attn.proj (target PROJ)")
        return cls(linear, parent.num_heads)

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return F.linear(x * gate.repeat_interleave(self.head_dim, dim=-1).to(x.dtype), self.weight, self.bias)

    def effective_weight(self, gate: torch.Tensor) -> torch.Tensor:
        return self.weight.detach().float() * gate.repeat_interleave(self.head_dim)[None, :]
