"""Upload the results of an experiment script to W&B (one run per script call).

    from wandb_results import add_wandb_args, upload_results
    add_wandb_args(parser)                       # --wandb / --no-wandb (on by default), --wandb-project, --wandb-mode
    ...
    upload_results(args, job_type="prune_loop", group="bvit_s_tiny_imagenet",
                   files=[out_dir / "results.json", out_dir / "masks.pt"])

Every file is stored in an artifact ("<group>-<job_type>", type "results"); the "records" list of each JSON
file becomes a W&B table (per-step kept fractions also as a long table: method, target, layer, step, kept),
and the script's log file (run_logging) is added too. Figures are not uploaded: only numbers go to W&B.
Failures only print a warning: the experiment's own outputs on disk are never affected.
The project defaults to $WANDB_PROJECT or "bViT_Pruning" (also the default of train_bvit.py).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import run_logging

PROJECT = "bViT_Pruning"


def add_wandb_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True,
                        help="upload results to W&B (default: on)")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", PROJECT))
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline"],
                        help="offline: keep the run under ./wandb and upload later with `wandb sync`")


def _table(wandb, records: list[dict]):
    columns = sorted({key for record in records for key in record})
    rows = [[value if isinstance(value, (int, float, str, bool)) or value is None else json.dumps(value)
             for value in (record.get(column) for column in columns)] for record in records]
    return wandb.Table(columns=columns, data=rows)


def _kept_table(wandb, records: list[dict]):
    """Long table of kept_per_step ({layer: [fraction per step]}) for plotting in W&B."""
    keys = ("setting", "method", "target", "steps")
    rows = [[*(record.get(k) for k in keys), layer, step + 1, kept]
            for record in records for layer, values in (record.get("kept_per_step") or {}).items()
            for step, kept in enumerate(values)]
    return wandb.Table(columns=[*keys, "layer", "step", "kept"], data=rows) if rows else None


def upload_results(args: argparse.Namespace, *, job_type: str, group: str, files: list[Path],
                   summary: dict | None = None, name: str | None = None,
                   tags: list[str] = (), config: dict | None = None) -> None:
    if not getattr(args, "wandb", False):
        return
    files = [Path(f) for f in files if Path(f).exists()]
    log = run_logging._current.path if run_logging._current is not None else None
    try:
        import wandb

        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, mode=args.wandb_mode,
                         group=group, job_type=job_type, tags=list(tags),
                         name=name or f"{group}_{job_type}_{run_logging.run_timestamp()}",
                         config=config if config is not None else
                         {k: v for k, v in vars(args).items() if not k.startswith("wandb")},
                         dir=os.environ.get("WANDB_DIR"), reinit="create_new")
        logged = {}
        for path in files:
            if path.suffix == ".json":
                content = json.loads(path.read_text())
                if isinstance(content, dict) and isinstance(content.get("records"), list) and content["records"]:
                    logged[f"table/{path.stem}"] = _table(wandb, content["records"])
                    kept = _kept_table(wandb, content["records"])
                    if kept is not None:
                        logged[f"table/{path.stem}_kept_per_step"] = kept
        if logged:
            run.log(logged)
        if summary:
            run.summary.update(summary)
        artifact = wandb.Artifact(name=name or f"{group}-{job_type}", type="results")
        for path in [*files, *([log] if log is not None and log.exists() else [])]:
            artifact.add_file(str(path), name=f"{path.parent.name}/{path.name}")
        run.log_artifact(artifact)
        run.finish()
        print(f"wandb: uploaded {len(files)} files, {len(logged)} tables ({run.url or args.wandb_mode})", flush=True)
    except Exception as error:  # never lose an experiment because of the upload
        print(f"wandb upload failed ({type(error).__name__}: {error}); results are on disk", flush=True)
