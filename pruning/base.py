"""Target resolution and the wrapper that makes a Linear prunable per loop step.

A pruning method is the combination of three registered components:
    unit       what is removed      (units.py:      spectral / weight / neuron / head)
    scorer     how important it is  (scorers.py:    grad_gate / sigma / magnitude / random)
    step_mode  how steps are used   (step_modes.py: per_use / shared / permuted)
plus an allocation rule (per_layer / global) that turns a pruning ratio into masks.

Every prunable unit gets a gate a_{u,i} for each use u of the layer (the u-th time
its block is applied in the loop). Gates are 1 by default, act as 0/1 masks
when pruning, and their gradient is the first-order importance (paper Eq. 5).
"""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from models.bViT.schedule import LoopContext

# paper names -> RecurrentViT module patterns
ALIASES: dict[str, str] = {
    "Q": "blocks.*.attn.q",
    "K": "blocks.*.attn.k",
    "V": "blocks.*.attn.v",
    "PROJ": "blocks.*.attn.proj",
    "ATTN_PROJ": "blocks.*.attn.proj",
    "FFN_IN": "blocks.*.mlp.fc1",
    "FFN_OUT": "blocks.*.mlp.fc2",
}

_BLOCK_RE = re.compile(r"^blocks\.(\d+)\.")


def resolve_targets(model: nn.Module, patterns: Sequence[str]) -> list[str]:
    """Expand aliases / glob patterns into names of Linear layers, in model order."""
    linears = [name for name, module in model.named_modules()
               if isinstance(module, (nn.Linear, PrunableLinear)) and not name.endswith(".linear")]
    selected: list[str] = []
    for pattern in patterns:
        expanded = ALIASES.get(pattern.upper(), pattern)
        matches = [name for name in linears if fnmatch.fnmatchcase(name, expanded)]
        if not matches:
            raise ValueError(f"pattern {pattern!r} ({expanded}) matches no Linear layer. "
                             f"Aliases: {list(ALIASES)}. Layers: {linears[:12]}{' ...' if len(linears) > 12 else ''}")
        selected.extend(name for name in matches if name not in selected)
    return selected


def parse_groups(groups: Sequence[str]) -> list[list[str]]:
    """'V+FFN_IN' -> ['V', 'FFN_IN'] (one experiment pruning both)."""
    return [group.split("+") for group in groups]


def num_uses(model: nn.Module, name: str) -> int:
    """How many times the layer is applied per forward pass (1 outside RecurrentViT blocks)."""
    schedule = getattr(model, "schedule", None)
    match = _BLOCK_RE.match(name)
    if schedule is None or match is None:
        return 1
    return schedule.uses_per_block()[int(match.group(1))]


class PrunableLinear(nn.Module):
    """Wraps a Linear. Modes:
        identity: original layer
        score:    unit.forward with gates (num_uses, 2, num_units) that require grad;
                  gate [:, 0] acts on the CLS token, [:, 1] on the other tokens
        mask:     per-use effective weights reconstructed from 0/1 masks (cached)
    """

    def __init__(self, linear: nn.Linear, unit: nn.Module, num_uses: int, loop_ctx: LoopContext | None) -> None:
        super().__init__()
        self.linear = linear
        self.unit = unit
        self.num_uses = num_uses
        self.loop_ctx = loop_ctx
        self.mode = "identity"
        self.gates: torch.Tensor | None = None
        self.masks: torch.Tensor | None = None
        self._weights: torch.Tensor | None = None
        self._biases: torch.Tensor | None = None  # only for units that also gate the bias

    @property
    def num_units(self) -> int:
        return self.unit.num_units

    def _use_index(self) -> int:
        use = self.loop_ctx.use_index if self.loop_ctx is not None else 0
        if use >= self.num_uses:
            raise RuntimeError(f"use index {use} >= {self.num_uses}: the loop schedule changed after attach()")
        return use

    def start_scoring(self) -> None:
        self.mode = "score"
        self.gates = torch.ones(self.num_uses, 2, self.num_units, device=self.linear.weight.device,
                                requires_grad=True)

    def pop_gate_grad(self) -> torch.Tensor:
        """(num_uses, 2, num_units) gradient of the last backward, then reset."""
        grad = self.gates.grad
        if grad is None:
            grad = torch.zeros_like(self.gates)
        self.gates.grad = None
        return grad.detach()

    @torch.no_grad()
    def set_masks(self, masks: torch.Tensor) -> None:
        if masks.shape != (self.num_uses, self.num_units):
            raise ValueError(f"mask shape {tuple(masks.shape)} != {(self.num_uses, self.num_units)}")
        self.mode = "mask"
        self.gates = None
        self.masks = masks.bool()
        gates = self.masks.to(self.linear.weight.device, torch.float32)
        self._weights = torch.stack([self.unit.effective_weight(gates[u]) for u in range(self.num_uses)])
        biases = [self.unit.effective_bias(gates[u]) for u in range(self.num_uses)]
        self._biases = None if biases[0] is None else torch.stack(biases)

    def reset(self) -> None:
        self.mode = "identity"
        self.gates = self.masks = self._weights = self._biases = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "identity":
            return self.linear(x)
        use = self._use_index()
        if self.mode == "mask":
            bias = self.linear.bias if self._biases is None else self._biases[use].to(x.dtype)
            return F.linear(x, self._weights[use].to(x.dtype), bias)
        if x.dim() < 3:
            return self.unit(x, self.gates[use, 1])
        return torch.cat((self.unit(x[:, :1], self.gates[use, 0]), self.unit(x[:, 1:], self.gates[use, 1])), dim=1)


def attach(model: nn.Module, targets: Sequence[str], unit_cls) -> dict[str, PrunableLinear]:
    """Replace each target Linear by a PrunableLinear built with unit_cls.build(linear, parent)."""
    loop_ctx = getattr(model, "loop_ctx", None)
    wrappers: dict[str, PrunableLinear] = {}
    for name in targets:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        linear = getattr(parent, child)
        if isinstance(linear, PrunableLinear):
            raise RuntimeError(f"{name} is already wrapped; detach the previous pruner first")
        wrapper = PrunableLinear(linear, unit_cls.build(linear, parent), num_uses(model, name), loop_ctx)
        setattr(parent, child, wrapper)
        wrappers[name] = wrapper
    return wrappers


def detach(model: nn.Module, wrappers: dict[str, PrunableLinear]) -> None:
    """Put the original Linear layers back."""
    for name, wrapper in wrappers.items():
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, wrapper.linear)
