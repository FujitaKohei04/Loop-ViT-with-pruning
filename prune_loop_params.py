"""Parameter counts of the pruned bViTs saved by prune_loop_bvit.py (no data needed).

For every saved mask set (masks.pt / masks_mc.pt of seed 42) the compact model is rebuilt and two counts
are written to <run>/params.json:

    stored_params   what has to be stored: the shared matrices keep a unit if any step uses it, plus the
                    step-specific biases of the mean replacement (the per-step widths are index sets
                    into the shared matrices, mask_bits in results*.json, not counted here)
    params          parameters the current implementation materializes (one sliced copy per distinct mask)

Usage:
    uv run python prune_loop_params.py --run outputs/bvit_pruning_loop/bvit_s_tiny_imagenet \
        outputs/bvit_pruning_loop/bvit_s_tiny_imagenet_steps10 ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from models.bViT.schedule import LoopSchedule
from pruning.channels import compact_channels
from pruning.compact import stored_params
from train_bvit import load_model
from wandb_results import add_wandb_args, upload_results


def counts(model: torch.nn.Module) -> dict:
    return {"stored_params": stored_params(model), "params": sum(p.numel() for p in model.parameters())}


def run(run_dir: Path, args: argparse.Namespace) -> None:
    results = run_dir / ("results_mc.json" if (run_dir / "results_mc.json").exists() else "results.json")
    saved = json.loads(results.read_text())["args"]
    model, _ = load_model(saved["checkpoint"], use_ema=not saved.get("no_ema", False), device="cpu")
    if saved.get("max_steps"):
        model.schedule = LoopSchedule.parse(f"0x{saved['max_steps']}")
    out = {"checkpoint": saved["checkpoint"], "max_steps": saved.get("max_steps"),
           "unpruned": counts(model), "records": []}
    for name, with_means in (("masks.pt", False), ("masks_mc.pt", True)):
        path = run_dir / name
        if not path.exists():
            continue
        for (setting, method, target), saved in torch.load(path).items():
            masks, means = (saved["masks"], saved["means"]) if with_means else (saved, None)
            row = {"setting": setting, "method": method, "target": target,
                   **counts(compact_channels(model, masks, means=means))}
            out["records"].append(row)
    (run_dir / "params.json").write_text(json.dumps(out, indent=1))
    joint = {r["target"]: r["stored_params"] for r in out["records"]
             if r["setting"] == "rot" and r["method"] == "joint_mc"}
    print(f"{run_dir.name}: unpruned {out['unpruned']['stored_params']:,}, {len(out['records'])} records, "
          f"rot joint_mc stored: " + ", ".join(f"{t}: {p:,}" for t, p in sorted(joint.items())), flush=True)
    upload_results(args, job_type="prune_loop_params", group=run_dir.name, files=[run_dir / "params.json"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", nargs="+", type=Path, required=True)
    add_wandb_args(parser)
    args = parser.parse_args()
    for run_dir in args.run:
        run(run_dir, args)


if __name__ == "__main__":
    main()
