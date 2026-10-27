"""Pruning units: what a gate / mask entry removes.

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

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def effective_weight(self, gate: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

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
