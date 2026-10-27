"""Loop schedules for RecurrentViT (extension: not in the bViT paper).

A schedule is a sequence of stages. Each stage is (block ids, repeats): the
listed blocks are applied in order, and the group is repeated `repeats` times.

    "0x12"           A x12                       (bViT-B, paper)
    "(0-11)x1"       A B C ... L once each       (ViT-B, paper)
    "(0,1)x6"        A B A B ... (6 times)       (Table 3, ViT-S (2B,6S))
    "(0,1,2)x4"      A B C A B C ...             (LoopViT.py)
    "0x4,1x8"        A x4 then B x8
    "0x2,(1,2)x3,3"  A A B C B C B C D           (repeats default to 1)

`LoopContext` is updated by the model before every block call, so that pruning
wrappers, step embeddings and analysis hooks know which step is running.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

Stage = tuple[tuple[int, ...], int]

_STAGE_RE = re.compile(r"^(?:\((?P<group>[^()]*)\)|(?P<single>\d+))(?:x(?P<repeats>\d+))?$")


@dataclass(frozen=True)
class StepInfo:
    global_step: int  # index in the unrolled sequence (0-based)
    block_id: int     # which shared block is applied
    use_index: int    # how many times this block has been applied before (0-based)


@dataclass(frozen=True)
class LoopSchedule:
    stages: tuple[Stage, ...]

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("schedule must have at least one stage")
        for block_ids, repeats in self.stages:
            if not block_ids or repeats < 1 or min(block_ids) < 0:
                raise ValueError(f"invalid stage: {(block_ids, repeats)}")
        used = {block_id for block_ids, _ in self.stages for block_id in block_ids}
        missing = set(range(max(used) + 1)) - used
        if missing:
            raise ValueError(f"block ids {sorted(missing)} are never used (ids must be 0..n-1)")

    @classmethod
    def parse(cls, spec: str | LoopSchedule) -> LoopSchedule:
        if isinstance(spec, LoopSchedule):
            return spec
        stages = []
        for part in _split_top_level(spec.replace(" ", "")):
            match = _STAGE_RE.match(part)
            if match is None:
                raise ValueError(f"cannot parse stage {part!r} in loop spec {spec!r}")
            if match["single"] is not None:
                block_ids = (int(match["single"]),)
            else:
                block_ids = tuple(_parse_group(match["group"]))
            stages.append((block_ids, int(match["repeats"] or 1)))
        return cls(tuple(stages))

    @property
    def num_blocks(self) -> int:
        return 1 + max(block_id for block_ids, _ in self.stages for block_id in block_ids)

    def unroll(self) -> list[StepInfo]:
        steps: list[StepInfo] = []
        uses = [0] * self.num_blocks
        for block_ids, repeats in self.stages:
            for _ in range(repeats):
                for block_id in block_ids:
                    steps.append(StepInfo(len(steps), block_id, uses[block_id]))
                    uses[block_id] += 1
        return steps

    @property
    def num_steps(self) -> int:
        return sum(len(block_ids) * repeats for block_ids, repeats in self.stages)

    def uses_per_block(self) -> list[int]:
        uses = [0] * self.num_blocks
        for block_ids, repeats in self.stages:
            for block_id in block_ids:
                uses[block_id] += repeats
        return uses

    def __str__(self) -> str:
        parts = []
        for block_ids, repeats in self.stages:
            group = str(block_ids[0]) if len(block_ids) == 1 else "(" + ",".join(map(str, block_ids)) + ")"
            parts.append(f"{group}x{repeats}")
        return ",".join(parts)


class LoopContext:
    """Mutable "which step is running" state shared by the model and its wrappers."""

    def __init__(self) -> None:
        self.step: StepInfo | None = None

    @property
    def active(self) -> bool:
        return self.step is not None

    @property
    def use_index(self) -> int:
        """0 outside of the loop, so wrappers also work on plain (non-looped) calls."""
        return self.step.use_index if self.step is not None else 0


def _split_top_level(spec: str) -> list[str]:
    parts, depth, start = [], 0, 0
    for i, char in enumerate(spec):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(spec[start:i])
            start = i + 1
    parts.append(spec[start:])
    return [part for part in parts if part]


def _parse_group(group: str) -> list[int]:
    block_ids: list[int] = []
    for item in group.split(","):
        if "-" in item:
            first, last = map(int, item.split("-"))
            block_ids.extend(range(first, last + 1))
        else:
            block_ids.append(int(item))
    return block_ids
