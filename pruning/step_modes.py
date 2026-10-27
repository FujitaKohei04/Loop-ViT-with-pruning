"""How per-use (per-step) scores become per-use masks.

    per_use:  every use (step) gets its own top-k        (paper: step-conditioned)
    shared:   scores averaged over uses, one mask for all (paper: global gradient mask)
    permuted: per_use masks assigned to other steps       (paper: permuted)
              [A] a random derangement of the steps, fixed by the seed

To add one, subclass StepMode and decorate it with @register_step_mode("name").
"""

from __future__ import annotations

import torch

STEP_MODES: dict[str, type[StepMode]] = {}


def register_step_mode(name: str):
    def decorator(cls):
        STEP_MODES[name] = cls
        return cls
    return decorator


class StepMode:
    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def transform_scores(self, scores: torch.Tensor) -> torch.Tensor:
        """(num_uses, num_units) -> scores used for selection."""
        return scores

    def transform_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """(num_uses, num_units) selected masks -> masks that are applied."""
        return masks


@register_step_mode("per_use")
class PerUse(StepMode):
    pass


@register_step_mode("shared")
class Shared(StepMode):
    def transform_scores(self, scores: torch.Tensor) -> torch.Tensor:
        return scores.mean(dim=0, keepdim=True).expand_as(scores)


@register_step_mode("permuted")
class Permuted(StepMode):
    def transform_masks(self, masks: torch.Tensor) -> torch.Tensor:
        return masks[derangement(masks.shape[0], self.seed)]


def derangement(n: int, seed: int) -> torch.Tensor:
    """Random permutation with no fixed point (identity when n == 1)."""
    if n == 1:
        return torch.zeros(1, dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    while True:
        permutation = torch.randperm(n, generator=generator)
        if not bool((permutation == torch.arange(n)).any()):
            return permutation
