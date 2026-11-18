"""Multigrid loop ViT experiments (docs/mgloop/multigrid_loop_vit.md).

Every experiment is bViT-S (1 shared block, width 384) on CIFAR-100 with the train_bvit.py
recipe (300 epochs, seed 42); only the loop schedule and the grid level of each step differ.
`F` = step on the full 8x8 token grid, `C` = step on the 2x2-pooled 4x4 grid, `CC` = 2x2 grid.

Usage:
    uv run python mgloop_experiments.py list
    uv run python mgloop_experiments.py run                 # all experiments, one after another
    uv run python mgloop_experiments.py run w_cycle bvit_6  # only these
Finished runs (final_eval.json exists) are skipped and interrupted runs are resumed, so the
command can simply be started again.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

OUTPUT_DIR = Path("outputs/mgloop")
MODEL = "bvit_s"
DATASET = "cifar100"
SEED = 42

# existing runs that are reused as baselines (same recipe, seed 42)
BASELINE_RUNS = {
    "bvit_12": Path("outputs/bvit/bvit_s_cifar100_20261027-140057"),
    "vit_s": Path("outputs/bvit/vit_s_scratch_cifar100_20261027_192724"),
}


@dataclass(frozen=True)
class Experiment:
    name: str
    label: str           # short name used in tables and figures
    loop: str            # LoopSchedule spec (which block per step)
    levels: str | None   # grid level per step, None = bViT
    note: str
    model: str = MODEL
    extra: tuple[str, ...] = ()

    @property
    def run_dir(self) -> Path:
        return BASELINE_RUNS.get(self.name, OUTPUT_DIR / self.name)


EXPERIMENTS: dict[str, Experiment] = {e.name: e for e in [
    # * baselines
    Experiment("vit_s", "ViT-S (12 blocks)", "(0-11)x1", None,
               "no weight sharing, existing run", model="vit_s_scratch"),
    Experiment("bvit_12", "bViT-S 12F", "0x12", None, "bViT paper model, existing run"),
    Experiment("bvit_6", "bViT-S 6F", "0x6", None, "bViT with half the steps: same FLOPs as the 4F+8C models"),
    Experiment("bvit_4", "bViT-S 4F", "0x4", None, "only the fine steps of the 4F+8C models: what do the 8C add?"),
    # * 4 fine + 8 coarse steps (about half the FLOPs of bViT-S 12F), different orders
    Experiment("fine_to_coarse", "MG 4F→8C", "0x12", "0x4,1x8", "pool once (PiT-like order)"),
    Experiment("coarse_to_fine", "MG 8C→4F", "0x12", "1x8,0x4", "SpiralFormer-like order"),
    Experiment("v_cycle", "MG V-cycle", "0x12", "0x2,1x8,0x2", "one cycle: 2F, 8C, 2F"),
    Experiment("w_cycle", "MG W-cycle", "0x12", "(0,1,1)x4", "loop of loops: (1F, 2C) x4"),
    # * same FLOPs as bViT-S 12F
    Experiment("w_cycle_deep", "MG W-cycle deep", "0x25", "(0,1,1,1)x6,0", "(1F, 3C) x6 + 1F: 7F + 18C"),
    # * three grid levels
    Experiment("w_cycle_3level", "MG W-cycle 3-level", "0x17", "0,(1,2,2)x2,1,0,(1,2,2)x2,1,0",
               "two W-cycles over 8x8 / 4x4 / 2x2 grids: 3F + 6C + 8CC"),
    # * two blocks: is one block doing two resolutions the problem?
    Experiment("w_cycle_2block", "MG W-cycle, own block for C", "(0,1,1)x4", "(0,1,1)x4",
               "block 0 on the fine grid, block 1 on the coarse grid"),
    Experiment("bvit_2block_6", "bViT-S 2 blocks, 6F", "(0,1)x3", None,
               "same FLOPs and parameters as w_cycle_2block, full grid only"),
    # * ablation
    Experiment("w_cycle_no_level_embed", "MG W-cycle w/o level emb.", "0x12", "(0,1,1)x4",
               "ablation: no level embedding", extra=("--no-use-level-embed",)),
]}


def command(experiment: Experiment) -> list[str]:
    last = experiment.run_dir / "last.pth"
    if last.exists():
        return [sys.executable, "train_bvit.py", "--resume", str(last)]
    args = [sys.executable, "train_bvit.py", "--model", experiment.model, "--dataset", DATASET,
            "--loop", experiment.loop, "--seed", str(SEED), "--output-dir", str(OUTPUT_DIR),
            "--run-name", experiment.name, *experiment.extra]
    if experiment.levels is not None:
        args += ["--levels", experiment.levels]
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["list", "run"])
    parser.add_argument("names", nargs="*", help="experiments to run (default: all that are not baselines)")
    args = parser.parse_args()

    if args.action == "list":
        for e in EXPERIMENTS.values():
            done = "done" if (e.run_dir / "final_eval.json").exists() else "    "
            print(f"{done} {e.name:24s} loop={e.loop:10s} levels={str(e.levels):32s} {e.note}")
        return

    names = args.names or [name for name in EXPERIMENTS if name not in BASELINE_RUNS]
    for name in names:
        experiment = EXPERIMENTS[name]
        if (experiment.run_dir / "final_eval.json").exists():
            print(f"skip {name}: already finished", flush=True)
            continue
        if name in BASELINE_RUNS:
            raise SystemExit(f"{name} is an existing baseline run and is not retrained here")
        print(f"\n===== {name}: {' '.join(command(experiment))} =====", flush=True)
        subprocess.run(command(experiment), check=True)


if __name__ == "__main__":
    main()
