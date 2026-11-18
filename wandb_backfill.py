"""Upload results that were produced before the scripts logged to W&B (one W&B run per directory).

    uv run python wandb_backfill.py \
        --train outputs/bvit/bvit_s_tiny_imagenet_20261111_190414 outputs/bvit/bvit_s_cifar100_20261027-140057 \
        --results outputs/bvit_pruning_loop/bvit_s_tiny_imagenet outputs/bvit_pruning_compute/bvit_s_tiny_imagenet

--train    a train_bvit.py output directory: history.json is replayed per epoch with the metric names of
           train_bvit.py (epoch_stats/*, val/*, ema_val/*), final_eval.json goes to the summary and the
           per-step table, and config / history / eval files and the log go into an artifact
           (checkpoints only with --checkpoints).
--results  a prune_loop_bvit.py / prune_compute_bvit.py output directory: every file in it via upload_results.
Figures are not uploaded (numbers only).
The group is <model>_<dataset> for both, so a checkpoint and its pruning results sit together.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from wandb_results import add_wandb_args, upload_results

JOB = {"bvit_pruning_loop": "prune_loop", "bvit_pruning_compute": "prune_compute"}


def train_run(directory: Path, args: argparse.Namespace) -> None:
    import wandb

    config = json.loads((directory / "config.json").read_text())
    history = json.loads((directory / "history.json").read_text())
    group = f"{config['model']}_{config['dataset']}"
    logs = sorted((directory / "logs").glob("*.log")) if (directory / "logs").exists() else []
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, mode=args.wandb_mode, group=group,
                     job_type="train", name=directory.name, tags=[config["model"], config["dataset"], "backfill"],
                     config=config, dir=os.environ.get("WANDB_DIR"), reinit="create_new")
    run.define_metric("epoch")
    for prefix in ("epoch_stats", "val", "ema_val"):
        run.define_metric(f"{prefix}/*", step_metric="epoch")
    for record in history:
        values = {"epoch": record["epoch"]}
        for key, value in record.items():
            if key == "epoch" or isinstance(value, (list, dict)):
                continue
            if key.startswith("ema_val_"):
                values[f"ema_val/{key[len('ema_val_'):]}"] = value
            elif key.startswith("val_"):
                values[f"val/{key[len('val_'):]}"] = value
            else:
                values[f"epoch_stats/{key}"] = value
        run.log(values)
    best = max(history, key=lambda r: r.get("ema_val_top1", r.get("val_top1", 0)))
    summary = {"best_top1": best.get("ema_val_top1", best.get("val_top1")), "best_epoch": best["epoch"],
               "train_hours_epochs": sum(r.get("epoch_time_s", 0) for r in history) / 3600, "status": "finished"}
    for log in logs:
        match = re.search(r"===== end: .*\(((\d+)h)?(\d+)m(\d+)s\)", log.read_text(errors="replace"))
        if match:
            summary["train_hours_wall"] = int(match[2] or 0) + int(match[3]) / 60 + int(match[4]) / 3600
    if (directory / "final_eval.json").exists():
        final = json.loads((directory / "final_eval.json").read_text())
        summary.update(final_top1=final["top1"], final_top5=final.get("top5"))
        if final.get("top1_per_step"):
            table = wandb.Table(columns=["step", "top1"], data=[[t + 1, v] for t, v in enumerate(final["top1_per_step"])])
            run.log({"final/top1_per_step": table})
    run.summary.update(summary)
    artifact = wandb.Artifact(name=directory.name, type="train")
    files = [directory / n for n in ("config.json", "history.json", "final_eval.json", "gap_eval.json")]
    files += [p.resolve() for p in logs]
    if args.checkpoints:
        files.append(directory / "best.pth")
    for path in files:
        if path.exists():
            artifact.add_file(str(path), name=path.name)
    run.log_artifact(artifact)
    run.finish()
    print(f"wandb: train {directory.name} ({group}) {len(history)} epochs -> {run.url or args.wandb_mode}", flush=True)


def results_run(directory: Path, args: argparse.Namespace) -> None:
    kind = directory.parent.name
    files = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix != ".png")
    saved = {}
    for name in ("results_mc.json", "results.json"):
        if (directory / name).exists():
            saved = json.loads((directory / name).read_text()).get("args", {})
            break
    group = re.sub(r"_steps\d+$", "", directory.name)
    upload_results(args, job_type=JOB.get(kind, kind), group=group, files=files,
                   name=f"{kind}_{directory.name}", tags=["backfill"], config=saved)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", nargs="*", type=Path, default=[])
    parser.add_argument("--results", nargs="*", type=Path, default=[])
    parser.add_argument("--checkpoints", action="store_true", help="also upload best.pth of --train runs")
    add_wandb_args(parser)
    args = parser.parse_args()
    for directory in args.train:
        train_run(directory, args)
    for directory in args.results:
        results_run(directory, args)


if __name__ == "__main__":
    main()
