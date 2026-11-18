"""Pruner: unit x scorer x step_mode x allocation, and named method presets.

    with Pruner.from_method(model, ["V", "FFN_IN"], "spectral_step") as pruner:
        scores = pruner.compute_scores(ScoringContext(model, batches, loss_fn, device))
        for ratio in (0.2, 0.4):
            pruner.apply(pruner.make_masks(scores, ratio))
            evaluate(model, ...)
    # leaving the block puts the original Linear layers back

To add a method, add a METHODS entry (or a new unit / scorer / step mode).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from pruning.base import PrunableLinear, attach, detach, resolve_targets
from pruning.scorers import SCORERS, Scores, ScoringContext
from pruning.step_modes import STEP_MODES
from pruning.units import UNITS

Masks = dict[str, torch.Tensor]


@dataclass(frozen=True)
class Method:
    unit: str
    scorer: str
    step_mode: str
    allocation: str = "per_layer"  # per_layer: same ratio in every target / global: one ranking over all targets


METHODS: dict[str, Method] = {
    # paper Sec. 4.2.3 / Fig. 6
    "spectral_step": Method("spectral", "grad_gate", "per_use"),
    "spectral_global": Method("spectral", "grad_gate", "shared"),
    "spectral_permuted": Method("spectral", "grad_gate", "permuted"),
    "spectral_sigma": Method("spectral", "sigma", "shared"),
    # extensions (not in the paper)
    "spectral_random": Method("spectral", "random", "per_use"),
    "taylor": Method("weight", "grad_gate", "shared", "global"),  # = main.py Taylor pruning
    "taylor_per_layer": Method("weight", "grad_gate", "shared"),
    "taylor_step": Method("weight", "grad_gate", "per_use"),
    "magnitude": Method("weight", "magnitude", "shared", "global"),
    # structured (other papers); "_step" = step-conditioned variant for looped models
    "neuron_taylor": Method("neuron", "grad_gate", "shared"),       # Molchanov et al. 2019, target FFN_IN
    "neuron_taylor_step": Method("neuron", "grad_gate", "per_use"),
    "neuron_magnitude": Method("neuron", "magnitude", "shared"),
    "neuron_random": Method("neuron", "random", "shared"),
    "head_taylor": Method("head", "grad_gate", "shared"),           # Michel et al. 2019, target PROJ
    "head_taylor_step": Method("head", "grad_gate", "per_use"),
    "head_random": Method("head", "random", "shared"),
}


class Pruner:
    def __init__(self, model: nn.Module, targets: Sequence[str], method: Method, *, seed: int = 0) -> None:
        for kind, registry, key in (("unit", UNITS, method.unit), ("scorer", SCORERS, method.scorer),
                                    ("step_mode", STEP_MODES, method.step_mode)):
            if key not in registry:
                raise ValueError(f"unknown {kind} {key!r} (available: {list(registry)})")
        if method.allocation not in ("per_layer", "global"):
            raise ValueError(f"unknown allocation {method.allocation!r}")
        self.model = model
        self.method = method
        self.seed = seed
        self.targets = resolve_targets(model, targets)
        self.step_mode = STEP_MODES[method.step_mode](seed)
        self.wrappers: dict[str, PrunableLinear] = {}

    @classmethod
    def from_method(cls, model: nn.Module, targets: Sequence[str], name: str, *, seed: int = 0) -> Pruner:
        if name not in METHODS:
            raise ValueError(f"unknown method {name!r} (available: {list(METHODS)})")
        return cls(model, targets, METHODS[name], seed=seed)

    # * lifecycle

    def attach(self) -> Pruner:
        if not self.wrappers:
            self.wrappers = attach(self.model, self.targets, UNITS[self.method.unit])
        return self

    def detach(self) -> None:
        detach(self.model, self.wrappers)
        self.wrappers = {}

    def __enter__(self) -> Pruner:
        return self.attach()

    def __exit__(self, *exc) -> None:
        self.detach()

    # * scoring / masking

    def compute_scores(self, ctx: ScoringContext) -> Scores:
        self.attach()
        return SCORERS[self.method.scorer](self.wrappers, ctx)

    def make_masks(self, scores: Scores, ratio: float) -> Masks:
        """ratio = fraction of units removed in every use of every target
        (per_layer), or over all targets together (global)."""
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("ratio must be in [0, 1]")
        selection = {name: self.step_mode.transform_scores(scores[name]["total"].float())
                     for name in self.targets}
        if self.method.allocation == "per_layer":
            masks = {name: _topk_rows(score, ratio) for name, score in selection.items()}
        else:
            masks = _global_threshold(selection, ratio)
        return {name: self.step_mode.transform_masks(mask) for name, mask in masks.items()}

    def apply(self, masks: Masks) -> None:
        self.attach()
        for name, wrapper in self.wrappers.items():
            wrapper.set_masks(masks[name])

    def reset(self) -> None:
        for wrapper in self.wrappers.values():
            wrapper.reset()


def _topk_rows(scores: torch.Tensor, ratio: float) -> torch.Tensor:
    keep = round((1 - ratio) * scores.shape[1])
    masks = torch.zeros_like(scores, dtype=torch.bool)
    if keep > 0:
        masks.scatter_(1, scores.topk(keep, dim=1).indices, True)
    return masks


def _global_threshold(selection: dict[str, torch.Tensor], ratio: float) -> Masks:
    """One ranking over all targets; each target's scores are divided by its mean
    so layers with different gradient scales are comparable ([A] normalization)."""
    normalized = {name: score / score.mean().clamp_min(1e-12) for name, score in selection.items()}
    flat = torch.cat([score.flatten() for score in normalized.values()])
    keep = round((1 - ratio) * flat.numel())
    if keep == 0:
        return {name: torch.zeros_like(score, dtype=torch.bool) for name, score in normalized.items()}
    threshold = flat.topk(keep).values[-1]
    return {name: score >= threshold for name, score in normalized.items()}
