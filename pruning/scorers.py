"""Importance scorers. Each returns {target: {"total": (num_uses, num_units), ...}}.

To add a scorer, write fn(wrappers, context) -> Scores and decorate it with
@register_scorer("name").
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn
from tqdm import tqdm

from pruning.base import PrunableLinear

Scores = dict[str, dict[str, torch.Tensor]]
SCORERS: dict[str, callable] = {}


@dataclass
class ScoringContext:
    model: nn.Module
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]] | None
    loss_fn: nn.Module
    device: torch.device
    amp: bool = False
    seed: int = 0


def register_scorer(name: str):
    def decorator(fn):
        SCORERS[name] = fn
        return fn
    return decorator


@register_scorer("grad_gate")
def grad_gate(wrappers: dict[str, PrunableLinear], ctx: ScoringContext) -> Scores:
    """Paper Eq. 6: I_{t,i} = E[|d loss / d a_{t,i}|], expectation over mini-batches
    ([A] the paper does not say whether |.| is taken per sample or per batch).
    Also returns the CLS / patch split of the gradient (G_t = G_cls + G_patch):
        total = |g_cls + g_patch|, cls = |g_cls|, patch = |g_patch|"""
    if ctx.batches is None:
        raise ValueError("grad_gate needs calibration batches")
    model = ctx.model
    model.eval()
    requires_grad = {p: p.requires_grad for p in model.parameters()}
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for wrapper in wrappers.values():
        wrapper.start_scoring()

    sums = {name: {key: torch.zeros(w.num_uses, w.num_units, device=ctx.device)
                   for key in ("total", "cls", "patch")} for name, w in wrappers.items()}
    num_batches = 0
    try:
        for images, targets in tqdm(ctx.batches, desc="scoring", leave=False):
            images, targets = images.to(ctx.device), targets.to(ctx.device)
            with torch.autocast(ctx.device.type, dtype=torch.float16, enabled=ctx.amp and ctx.device.type == "cuda"):
                loss = ctx.loss_fn(model(images), targets)
            loss.float().backward()
            for name, wrapper in wrappers.items():
                grad = wrapper.pop_gate_grad()
                sums[name]["total"] += (grad[:, 0] + grad[:, 1]).abs()
                sums[name]["cls"] += grad[:, 0].abs()
                sums[name]["patch"] += grad[:, 1].abs()
            num_batches += 1
    finally:
        for wrapper in wrappers.values():
            wrapper.reset()
        for parameter, flag in requires_grad.items():
            parameter.requires_grad_(flag)
    if num_batches == 0:
        raise ValueError("no calibration batches")
    return {name: {key: value.cpu() / num_batches for key, value in parts.items()} for name, parts in sums.items()}


def _static(key: str):
    def scorer(wrappers: dict[str, PrunableLinear], ctx: ScoringContext) -> Scores:
        scores = {}
        for name, wrapper in wrappers.items():
            static = wrapper.unit.static_scores()
            if key not in static:
                raise ValueError(f"scorer {key!r} is not available for unit {type(wrapper.unit).__name__}")
            scores[name] = {"total": static[key].cpu().expand(wrapper.num_uses, -1).clone()}
        return scores
    return scorer


register_scorer("sigma")(_static("sigma"))          # largest singular value first (paper baseline)
register_scorer("magnitude")(_static("magnitude"))  # |w|


@register_scorer("random")
def random_scores(wrappers: dict[str, PrunableLinear], ctx: ScoringContext) -> Scores:
    generator = torch.Generator().manual_seed(ctx.seed)
    return {name: {"total": torch.rand(w.num_uses, w.num_units, generator=generator)}
            for name, w in wrappers.items()}
