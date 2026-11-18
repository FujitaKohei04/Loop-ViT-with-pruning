"""Learning-rate search for train_bvit.py.

Step 1  range-test : LR range test (Smith 2017). LR grows exponentially every optimizer step from
                     --lr-min to --lr-max; the (smoothed) loss curve shows where training diverges.
Step 2  search     : short runs (same warmup + cosine shape, shrunk to --epochs) over a log grid or
                     with TPE, early-stopping bad trials with Optuna (ASHA / median pruner).
Step 3  (manual)   : `search` prints the train_bvit.py commands that rerun the top-k LRs for the full
                     300 epochs. Run them yourself (hours each), then compare.

Everything not given on the command line uses the train_bvit.py defaults (paper recipe), and the
train_bvit.py flags (--model, --dataset, --loop, --batch-size, --effective-batch, ...) are accepted.

Usage:
    uv run python lr_search_bvit.py range-test --model bvit_s --dataset cifar100
    uv run python lr_search_bvit.py search --model bvit_s --dataset cifar100 --lrs 2.5e-4,5e-4,1e-3,2e-3
    uv run python lr_search_bvit.py search --model bvit_s --dataset cifar100 --lr-range 1e-4,3e-3 --n-trials 8
    uv run python lr_search_bvit.py search ... --study bvit_s_cifar100   # same name -> resume the study
    uv run python lr_search_bvit.py summary --study bvit_s_cifar100

Caveats:
    * a short run favours a different LR than the 300-epoch run can; that is why step 3 exists.
    * with few trials ASHA promotes a trial only if it is the best so far at that epoch, so a
      high LR that is slow early can be pruned. Use --pruner none for a plain grid if time allows.
    * the range test starts from random init without warmup: read its result as an upper bound.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
from dataclasses import asdict, fields, replace
from itertools import cycle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import optuna  # noqa: E402
import torch  # noqa: E402
from timm.data import Mixup  # noqa: E402
from timm.loss import SoftTargetCrossEntropy  # noqa: E402
from torch import nn  # noqa: E402
from tqdm import tqdm  # noqa: E402

from dl_dataset.registry import build_loaders, get_spec  # noqa: E402
from models.bViT.bViT import PRESETS  # noqa: E402
from run_logging import add_log_args, run_timestamp, start_run_log  # noqa: E402
from train_bvit import Config, autocast, build_model, parameter_groups, set_seed, train  # noqa: E402

DEFAULT_ROOT = Path("./outputs/bvit_lr_search")
# Config fields this script sets itself (not taken from the command line; --epochs means the short run)
SCRIPT_FIELDS = ("lr", "epochs", "warmup_epochs", "eval_interval", "run_name", "output_dir")


# * ===== Step 1: LR range test =====

def range_test(config: Config, *, lr_min: float, lr_max: float, num_steps: int, smoothing: float,
               diverge_factor: float, out_dir: Path) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(config.seed)
    spec = get_spec(config.dataset, config.data_root)
    train_loader, _, classes = build_loaders(
        spec, batch_size=config.batch_size, num_workers=config.num_workers, augment=config.augment,
        re_prob=config.re_prob, eval_batch_size=2 * config.batch_size)
    model = build_model(config, spec, len(classes)).to(device)
    model.train()
    optimizer = torch.optim.AdamW(parameter_groups(model, config.weight_decay), lr=lr_min)
    scaler = torch.amp.GradScaler(enabled=config.amp and device.type == "cuda")
    mixup = Mixup(mixup_alpha=config.mixup_alpha, cutmix_alpha=config.cutmix_alpha,
                  switch_prob=config.mixup_switch_prob, label_smoothing=config.label_smoothing,
                  num_classes=len(classes))
    loss_fn = SoftTargetCrossEntropy()
    accum = config.accum_steps
    gamma = (lr_max / lr_min) ** (1 / max(num_steps - 1, 1))

    lrs, losses, smoothed = [], [], []
    average, best = 0.0, math.inf
    batches = cycle(train_loader)  # [A] cycling repeats the same augmented order; fine for a few hundred steps
    stop_reason = "reached lr_max"
    bar = tqdm(range(num_steps), desc="range test")
    for step in bar:
        lr = lr_min * gamma ** step
        for group in optimizer.param_groups:
            group["lr"] = lr
        step_loss = 0.0
        for _ in range(accum):
            images, targets = next(batches)
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            images, soft_targets = mixup(images, targets)
            with autocast(config, device):
                loss = loss_fn(model(images), soft_targets)
            scaler.scale(loss / accum).backward()
            step_loss += loss.item() / accum
        if config.clip_grad is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if not math.isfinite(step_loss):
            stop_reason = f"non-finite loss at lr={lr:.2e}"
            break
        average = smoothing * average + (1 - smoothing) * step_loss
        value = average / (1 - smoothing ** (step + 1))  # bias correction
        lrs.append(lr)
        losses.append(step_loss)
        smoothed.append(value)
        best = min(best, value)
        bar.set_postfix(lr=f"{lr:.1e}", loss=f"{value:.3f}")
        if step > num_steps // 10 and value > diverge_factor * best:
            stop_reason = f"smoothed loss > {diverge_factor} x min at lr={lr:.2e}"
            break

    result = {"config": asdict(config), "lr_min": lr_min, "lr_max": lr_max, "num_steps": num_steps,
              "stop_reason": stop_reason, "lrs": lrs, "losses": losses, "smoothed": smoothed}
    result.update(suggest_from_curve(lrs, smoothed))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "range_test.json").write_text(json.dumps(result, indent=2))
    plot_range_test(result, out_dir / "range_test.png")
    return result


def suggest_from_curve(lrs: list[float], smoothed: list[float]) -> dict:
    """lr at the minimum smoothed loss, the usual "min / 10" rule, and the steepest descent."""
    if len(lrs) < 5:
        return {}
    lrs_arr, loss_arr = np.asarray(lrs), np.asarray(smoothed)
    i_min = int(loss_arr.argmin())
    skip = len(lrs) // 10  # the first steps are dominated by the bias-corrected average
    slope = np.gradient(loss_arr, np.log10(lrs_arr))
    i_steep = skip + int(slope[skip:i_min + 1].argmin()) if i_min > skip else i_min
    return {"lr_at_min_loss": float(lrs_arr[i_min]), "lr_min_loss_div10": float(lrs_arr[i_min] / 10),
            "lr_steepest": float(lrs_arr[i_steep])}


def plot_range_test(result: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(result["lrs"], result["losses"], color="0.75", linewidth=0.8, label="loss")
    ax.plot(result["lrs"], result["smoothed"], color="C0", label="smoothed loss")
    marks = [("lr_steepest", "C2", "steepest"), ("lr_min_loss_div10", "C1", "min / 10"),
             ("lr_at_min_loss", "C3", "min loss")]
    for key, color, label in marks:
        if key in result:
            ax.axvline(result[key], color=color, linestyle="--", label=f"{label}: {result[key]:.1e}")
    paper_lr = result["config"]["lr"]
    ax.axvline(paper_lr, color="k", linestyle=":", label=f"current lr: {paper_lr:.1e}")
    ax.set_xscale("log")
    low = min(result["smoothed"]) if result["smoothed"] else 0
    ax.set_ylim(low * 0.95, min(max(result["smoothed"], default=1), low * 2.5))
    ax.set_xlabel("learning rate")
    ax.set_ylabel("train loss (Mixup/CutMix soft targets)")
    cfg = result["config"]
    ax.set_title(f"LR range test: {cfg['model']} / {cfg['dataset']} / batch {cfg['effective_batch']}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# * ===== Step 2: short-run search with Optuna =====

def trial_score(record: dict) -> float | None:
    """Same metric train_bvit.py uses for best.pth: EMA top1 if available, else raw top1."""
    return record.get("ema_val_top1", record.get("val_top1"))


def make_pruner(name: str, epochs: int, min_resource: int | None, reduction_factor: int) -> optuna.pruners.BasePruner:
    min_resource = min_resource or max(1, epochs // 3)
    if name == "asha":
        return optuna.pruners.SuccessiveHalvingPruner(min_resource=min_resource, reduction_factor=reduction_factor)
    if name == "median":
        return optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=min_resource)
    return optuna.pruners.NopPruner()


def search(base: Config, args: argparse.Namespace) -> None:
    study_name = args.study or f"{base.model}_{base.dataset}_{run_timestamp()}"
    study_dir = Path(args.root) / study_name
    study_dir.mkdir(parents=True, exist_ok=True)
    warmup = max(1, round(args.epochs * base.warmup_epochs / base.epochs))  # keep the paper's 10% ratio

    if args.lrs:
        grid = sorted(float(v) for v in args.lrs.split(","))
        # every grid point is enqueued below, so the sampler is never consulted
        sampler: optuna.samplers.BaseSampler = optuna.samplers.RandomSampler(seed=base.seed)
        low, high = grid[0], grid[-1]
    else:
        low, high = (float(v) for v in args.lr_range.split(","))
        sampler = optuna.samplers.TPESampler(seed=base.seed, n_startup_trials=min(4, args.n_trials))
        n_trials = args.n_trials
    study = optuna.create_study(
        study_name=study_name, storage=f"sqlite:///{(study_dir / 'study.db').resolve()}", load_if_exists=True,
        direction="maximize", sampler=sampler,
        pruner=make_pruner(args.pruner, args.epochs, args.min_resource, args.reduction_factor))
    if args.lrs:
        # current (paper) lr first as the reference; on resume skip points that already finished
        done = {t.params.get("lr") for t in study.trials
                if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)}
        waiting = {t.params.get("lr") for t in study.trials if t.state == optuna.trial.TrialState.WAITING}
        todo = [lr for lr in sorted(grid, key=lambda v: v != base.lr) if lr not in done]
        for lr in todo:
            if lr not in waiting:
                study.enqueue_trial({"lr": lr})
        n_trials = len(todo)
    meta = {"base_config": asdict(base), "epochs": args.epochs, "warmup_epochs": warmup, "pruner": args.pruner,
            "lrs": args.lrs, "lr_range": None if args.lrs else [low, high]}
    (study_dir / "search_config.json").write_text(json.dumps(meta, indent=2))
    print(f"study={study_name} dir={study_dir} short schedule: {args.epochs} ep (warmup {warmup}) "
          f"pruner={args.pruner}")

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", low, high, log=True)
        config = replace(base, lr=lr, epochs=args.epochs, warmup_epochs=warmup, eval_interval=1,
                         output_dir=str(study_dir), run_name=f"trial{trial.number:03d}_lr{lr:.2e}")
        print(f"\n=== trial {trial.number}: lr={lr:.3e} ===", flush=True)

        def report(epoch: int, record: dict) -> None:
            score = trial_score(record)
            if score is None:
                return
            if not math.isfinite(record["train_loss"]):
                raise optuna.TrialPruned(f"non-finite train loss at epoch {epoch}")
            trial.report(score, epoch)
            if epoch < args.epochs and trial.should_prune():  # a finished run is never pruned
                raise optuna.TrialPruned(f"pruned at epoch {epoch} (score {score:.4f})")

        history = train(config, None, epoch_callback=report, save_checkpoints=args.save_checkpoints)
        scores = [s for s in map(trial_score, history) if s is not None]
        trial.set_user_attr("best_score", max(scores))
        return scores[-1]  # final epoch: with cosine to 0 this is what a full run would keep

    remaining = n_trials if args.lrs else n_trials - len([t for t in study.trials if t.state.is_finished()])
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, catch=(torch.cuda.OutOfMemoryError,))
    summarize(study, study_dir, base, args.top_k)


def summarize(study: optuna.Study, study_dir: Path, base: Config | None, top_k: int) -> None:
    rows = []
    for t in study.trials:
        steps = sorted(t.intermediate_values)
        last_epoch = steps[-1] if steps else 0
        rows.append({"trial": t.number, "lr": t.params.get("lr"), "state": t.state.name, "final": t.value,
                     "best": t.user_attrs.get("best_score"), "last_epoch": last_epoch,
                     "last_score": t.intermediate_values[last_epoch] if steps else None,
                     "curve": [t.intermediate_values[s] for s in steps]})
    (study_dir / "summary.json").write_text(json.dumps(rows, indent=2))

    print(f"\n{'trial':>5} {'lr':>10} {'state':>9} {'final':>7} {'best':>7} {'stopped@':>8}")
    for r in sorted(rows, key=lambda r: (r["state"] != "COMPLETE", -(r["final"] or 0))):
        fmt = lambda v: f"{v:.4f}" if v is not None else "-"  # noqa: E731
        print(f"{r['trial']:>5} {r['lr']:>10.3e} {r['state']:>9} {fmt(r['final']):>7} {fmt(r['best']):>7} "
              f"{r['last_epoch']:>8}")
    plot_curves(rows, study_dir / "curves.png")
    print(f"\ncurves: {study_dir / 'curves.png'}  summary: {study_dir / 'summary.json'}")

    complete = sorted((r for r in rows if r["state"] == "COMPLETE"), key=lambda r: -r["final"])
    if not complete or base is None:
        return
    print(f"\nStep 3: rerun the top {min(top_k, len(complete))} for the full schedule "
          f"({base.epochs} epochs):")
    passthrough = changed_flags(base)
    for r in complete[:top_k]:
        print(f"  uv run python train_bvit.py{passthrough} "
              f"--lr {r['lr']:.3e} --run-name {base.model}_{base.dataset}_lr{r['lr']:.2e}")


def changed_flags(config: Config) -> str:
    """train_bvit.py flags for every field that differs from the defaults (model/dataset always)."""
    default = Config()
    parts = []
    for f in fields(Config):
        value = getattr(config, f.name)
        if f.name in SCRIPT_FIELDS or f.name in ("max_train_batches", "max_eval_batches"):
            continue
        if value == getattr(default, f.name) and f.name not in ("model", "dataset"):
            continue
        flag = "--" + f.name.replace("_", "-")
        if isinstance(value, bool):
            parts.append(flag if value else "--no-" + flag[2:])
        else:
            parts.append(f"{flag} {shlex.quote(str(value))}")
    return "".join(" " + part for part in parts)


def plot_curves(rows: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in sorted(rows, key=lambda r: r["lr"] or 0):
        if not r["curve"]:
            continue
        style = "-" if r["state"] == "COMPLETE" else ":"
        ax.plot(range(1, len(r["curve"]) + 1), r["curve"], style, label=f"lr {r['lr']:.2e} ({r['state'].lower()})")
    ax.set_xlabel("epoch (short schedule)")
    ax.set_ylabel("val top1 (EMA)")
    ax.set_title("LR search: short runs")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# * ===== CLI =====

def add_config_args(parser: argparse.ArgumentParser) -> None:
    for field in fields(Config):
        if field.name in SCRIPT_FIELDS:
            continue  # set by this script
        flag = "--" + field.name.replace("_", "-")
        if field.type == "bool":
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=None)
        else:
            caster = {"int": int, "float": float}.get(field.type.split(" |")[0], str)
            parser.add_argument(flag, type=caster, default=None)


def config_from_args(args: argparse.Namespace) -> Config:
    overrides = {f.name: getattr(args, f.name) for f in fields(Config)
                 if f.name not in SCRIPT_FIELDS and getattr(args, f.name, None) is not None}
    config = replace(Config(), **overrides)
    if config.model not in PRESETS:
        raise ValueError(f"unknown model {config.model} (available: {list(PRESETS)})")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("range-test", help="step 1: LR range test")
    add_config_args(p)
    p.add_argument("--lr-min", type=float, default=1e-6)
    p.add_argument("--lr-max", type=float, default=1e-1)
    p.add_argument("--steps", type=int, default=150, help="optimizer steps (each = effective_batch images)")
    p.add_argument("--smoothing", type=float, default=0.95, help="EMA factor of the plotted loss")
    p.add_argument("--diverge-factor", type=float, default=1.5,
                   help="stop when the smoothed loss exceeds this x its minimum")
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    add_log_args(p)

    p = sub.add_parser("search", help="step 2: short runs + Optuna")
    add_config_args(p)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--lrs", help="comma-separated grid, e.g. 2.5e-4,5e-4,1e-3,2e-3")
    group.add_argument("--lr-range", help="'low,high' for TPE (log scale), e.g. 1e-4,3e-3")
    p.add_argument("--n-trials", type=int, default=8, help="TPE only")
    p.add_argument("--epochs", type=int, default=100, help="length of each short run")
    p.add_argument("--pruner", choices=["asha", "median", "none"], default="asha")
    p.add_argument("--min-resource", type=int, default=None, help="first pruning epoch (default epochs // 3)")
    p.add_argument("--reduction-factor", type=int, default=3, help="ASHA")
    p.add_argument("--save-checkpoints", action="store_true", help="keep best/last.pth of every trial")
    p.add_argument("--study", default=None, help="study name; reuse it to resume")
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    add_log_args(p)

    p = sub.add_parser("summary", help="print the table / plot of an existing study")
    p.add_argument("--study", required=True)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--root", default=str(DEFAULT_ROOT))

    args = parser.parse_args()
    if args.command == "summary":
        study_dir = Path(args.root) / args.study
        study = optuna.load_study(study_name=args.study, storage=f"sqlite:///{(study_dir / 'study.db').resolve()}")
        meta = json.loads((study_dir / "search_config.json").read_text())
        summarize(study, study_dir, Config(**meta["base_config"]), args.top_k)
        return

    config = config_from_args(args)
    if args.command == "range-test":
        tag = f"range_{config.model}_{config.dataset}"
        start_run_log(__file__, tag=tag, log_dir=args.log_dir, config={**asdict(config), **vars(args)},
                      enabled=not args.no_log)
        out_dir = Path(args.root) / f"{tag}_{run_timestamp()}"
        result = range_test(config, lr_min=args.lr_min, lr_max=args.lr_max, num_steps=args.steps,
                            smoothing=args.smoothing, diverge_factor=args.diverge_factor, out_dir=out_dir)
        print(f"stop: {result['stop_reason']}")
        for key in ("lr_steepest", "lr_min_loss_div10", "lr_at_min_loss"):
            if key in result:
                print(f"{key:>18}: {result[key]:.2e}")
        print(f"current lr: {config.lr:.2e}\nplot: {out_dir / 'range_test.png'}")
        print("Pick a log grid for `search` around steepest .. min/10 (and include the current lr).")
    else:
        start_run_log(__file__, tag=f"search_{config.model}_{config.dataset}", log_dir=args.log_dir,
                      config={**asdict(config), **vars(args)}, enabled=not args.no_log)
        search(config, args)


if __name__ == "__main__":
    main()
