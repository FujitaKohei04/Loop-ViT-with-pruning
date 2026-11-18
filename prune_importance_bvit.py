"""Which importance measure suits step-conditioned pruning of a looped ViT?

The bViT paper scores unit i at step t by I[t, i] = E|d loss / d gate[t, i]|
and keeps the top units of every step. That score treats every (step, unit)
pair on its own. Here the same per-sample gate gradients g[n, t, i] give
several scores, and only the score / allocation changes between methods:

    paper         mean over batches of |batch-mean g|, same ratio in every step   (existing grad_gate)
    abs           mean_n |g_n|, same ratio in every step                           (no cancellation between samples)
    fisher        mean_n g_n^2, same ratio in every step                           (second-order, diagonal)
    fisher_alloc  mean_n g_n^2, one ranking over all (step, unit) pairs of a layer (steps get different widths)
    joint         greedy on E_n (sum_{t in removed} g[n, t, i])^2: adds the cross-step terms
                  F_i[t, s] = E_n g[n, t, i] g[n, s, i] of the same unit, one ranking per layer
    iter_fisher   fisher_alloc, re-scored on the pruned model before every further 10%
    iter_joint    joint, re-scored the same way
    shared_mean   one mask for all steps, score = mean_t paper                     (paper "global")
    shared_max    one mask, score = max_t (paper / its step mean)                  (a unit vital in one step survives)
    shared_joint  one mask, score = E_n (sum_t g[n, t, i])^2

Units: neuron (FFN_IN), head (PROJ), spectral (all six projections). No retraining.

Usage:
    uv run python prune_importance_bvit.py --checkpoint outputs/bvit/<run>/best.pth

Outputs (./outputs/bvit_pruning_importance/<run>/): results.json, diagnostics.pt
"""

from __future__ import annotations

import argparse
import json
from itertools import islice
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from dl_dataset.registry import build_loaders, get_spec
from model_profile import count_flops
from pruning import Pruner, compact_model, stored_params
from pruning.pruner import Method
from run_logging import add_log_args, link_run_log, run_timestamp, start_run_log
from train_bvit import Config, evaluate, load_model, set_seed

SETTINGS = {  # label -> (unit, targets)
    "neuron": ("neuron", ["FFN_IN"]),
    "head": ("head", ["PROJ"]),
    "spectral": ("spectral", ["Q", "K", "V", "PROJ", "FFN_IN", "FFN_OUT"]),
}
ONE_SHOT = ["paper", "abs", "fisher", "fisher_alloc", "joint", "shared_mean", "shared_max", "shared_joint"]
ITERATIVE = ["iter_fisher", "iter_joint"]


# * ===== Per-sample gate gradients =====

def collect(model: nn.Module, wrappers: dict, batches: list, device: torch.device,
            keep: dict[str, torch.Tensor] | None = None) -> dict[str, dict[str, torch.Tensor]]:
    """Statistics of g[n, t, i] = d loss_n / d gate[t, i] (CLS + patch gates), per target layer.
    keep: current masks (num_uses, num_units); the gradients are then taken on the pruned model.
        paper (T, U): mean over batches of |batch-mean g|
        abs   (T, U): mean_n |g|
        cross (U, T, T): mean_n g[n, t, i] g[n, s, i]   (diagonal = Fisher)"""
    model.eval()
    requires_grad = {p: p.requires_grad for p in model.parameters()}
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    stats = {name: {"paper": torch.zeros(w.num_uses, w.num_units, device=device),
                    "abs": torch.zeros(w.num_uses, w.num_units, device=device),
                    "cross": torch.zeros(w.num_units, w.num_uses, w.num_uses, device=device)}
             for name, w in wrappers.items()}
    num_images = num_batches = 0
    try:
        for images, targets in tqdm(batches, desc="scoring", leave=False):
            images, targets = images.to(device), targets.to(device)
            batch = len(images)
            for name, w in wrappers.items():
                # one gate per sample: (uses, cls/patch, batch, 1, units) broadcasts over the tokens
                gates = torch.ones(w.num_uses, 2, batch, 1, w.num_units, device=device)
                if keep is not None:
                    gates = gates * keep[name].to(device, torch.float32)[:, None, None, None, :]
                w.mode, w.gates = "score", gates.requires_grad_()
            F.cross_entropy(model(images).float(), targets, reduction="sum").backward()
            for name, w in wrappers.items():
                grad = (w.gates.grad[:, 0] + w.gates.grad[:, 1]).squeeze(2)  # (uses, batch, units)
                stats[name]["paper"] += grad.mean(dim=1).abs()
                stats[name]["abs"] += grad.abs().sum(dim=1)
                stats[name]["cross"] += torch.einsum("tbi,sbi->its", grad, grad)
                w.gates = None
            num_images += batch
            num_batches += 1
    finally:
        for w in wrappers.values():
            w.reset()
        for parameter, flag in requires_grad.items():
            parameter.requires_grad_(flag)
    return {name: {"paper": s["paper"].cpu() / num_batches, "abs": s["abs"].cpu() / num_images,
                   "cross": s["cross"].cpu() / num_images} for name, s in stats.items()}


# * ===== Removal orders =====

def order_by_score(score: torch.Tensor, alive: torch.Tensor) -> torch.Tensor:
    """Flat indices of the alive (step, unit) pairs, least important first."""
    flat = score.masked_fill(~alive, float("inf")).flatten()
    return flat.argsort()[: int(alive.sum())]


def order_joint(cross: torch.Tensor, alive: torch.Tensor) -> torch.Tensor:
    """Greedy removal on cost(S) = E_n (sum_{(t, i) in S} g[n, t, i])^2 restricted to pairs of the same unit:
    removing (t, i) raises the cost of removing the same unit at another step s by 2 F_i[t, s]."""
    num_uses, num_units = alive.shape
    marginal = cross.diagonal(dim1=1, dim2=2).T.clone()  # (T, U)
    marginal[~alive] = float("inf")
    order = torch.empty(int(alive.sum()), dtype=torch.long)
    for k in range(len(order)):
        index = int(marginal.argmin())
        step, unit = divmod(index, num_units)
        order[k] = index
        marginal[:, unit] += 2 * cross[unit, :, step]
        marginal[step, unit] = float("inf")
    return order


def masks_from_order(order: torch.Tensor, shape: tuple[int, int], removed: int,
                     keep: torch.Tensor | None = None) -> torch.Tensor:
    mask = torch.ones(shape, dtype=torch.bool) if keep is None else keep.clone()
    mask.view(-1)[order[:removed]] = False
    return mask


def per_step_masks(score: torch.Tensor, ratio: float) -> torch.Tensor:
    keep = round((1 - ratio) * score.shape[1])
    mask = torch.zeros_like(score, dtype=torch.bool)
    if keep > 0:
        mask.scatter_(1, score.topk(keep, dim=1).indices, True)
    return mask


def one_shot_masks(method: str, stat: dict[str, torch.Tensor], ratio: float, cache: dict) -> torch.Tensor:
    paper, fisher = stat["paper"], stat["cross"].diagonal(dim1=1, dim2=2).T
    shape = tuple(paper.shape)
    total = paper.numel()
    if method == "paper":
        return per_step_masks(paper, ratio)
    if method == "abs":
        return per_step_masks(stat["abs"], ratio)
    if method == "fisher":
        return per_step_masks(fisher, ratio)
    if method in ("fisher_alloc", "joint"):
        if method not in cache:
            alive = torch.ones(shape, dtype=torch.bool)
            cache[method] = order_by_score(fisher, alive) if method == "fisher_alloc" else order_joint(stat["cross"], alive)
        return masks_from_order(cache[method], shape, round(ratio * total))
    shared = {"shared_mean": paper.mean(dim=0),
              "shared_max": (paper / paper.mean(dim=1, keepdim=True).clamp_min(1e-20)).amax(dim=0),
              "shared_joint": stat["cross"].sum(dim=(1, 2))}[method]
    return per_step_masks(shared[None].expand(shape), ratio)


# * ===== Evaluation =====

@torch.no_grad()
def measure(model: nn.Module, pruner: Pruner, masks: dict[str, torch.Tensor], val_batches: list, img_size: int,
            device: torch.device) -> dict:
    pruner.apply(masks)
    metrics = evaluate(model, val_batches, device)
    small = compact_model(model)
    uses = torch.cat([mask.float().sum(dim=0) for mask in masks.values()])  # steps in which each unit is kept
    record = {"top1": metrics["top1"], "loss": metrics["loss"],
              "gflops.total": count_flops(small, img_size, device)["gflops.total"],
              "stored_params": stored_params(small),
              "union_kept": float((uses > 0).float().mean()),
              "kept_per_step": torch.stack([m.float().mean(dim=1) for m in masks.values()]).mean(dim=0).tolist()}
    pruner.reset()
    return record


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    model, checkpoint = load_model(args.checkpoint, use_ema=not args.no_ema, device=device)
    model.eval()
    train_config = Config(**checkpoint["config"])
    spec = get_spec(checkpoint["dataset"], args.data_root or train_config.data_root)
    calib_loader, val_loader, _ = build_loaders(spec, batch_size=args.batch_size, num_workers=args.num_workers,
                                                train_eval_transform=True)
    calib = list(islice(calib_loader, max(1, args.calib_images // args.batch_size)))
    iter_calib = calib[: max(1, args.iter_calib_images // args.batch_size)]
    val_batches = [(images.to(device), targets.to(device)) for images, targets in val_loader]

    out_dir = Path(args.output_dir) / (args.run_name or f"{Path(args.checkpoint).parent.name}_{run_timestamp()}")
    out_dir.mkdir(parents=True, exist_ok=True)
    link_run_log(out_dir)
    base = evaluate(model, val_batches, device)
    results = {"checkpoint": str(args.checkpoint), "model": train_config.model, "dataset": spec.name,
               "args": vars(args), "records": [],
               "unpruned": {"top1": base["top1"], "gflops.total": count_flops(model, spec.img_size, device)["gflops.total"]}}
    print(f"unpruned: top1={base['top1']:.4f} GFLOPs={results['unpruned']['gflops.total']:.3f}")
    diagnostics = {}
    if (out_dir / "results.json").exists():  # keep the settings of an earlier call that are not run again
        previous = json.loads((out_dir / "results.json").read_text())["records"]
        results["records"] = [r for r in previous if r["setting"] not in args.settings]
        if (out_dir / "diagnostics.pt").exists():
            diagnostics = {key: value for key, value in torch.load(out_dir / "diagnostics.pt").items()
                           if key not in args.settings}

    def record(setting: str, method: str, ratio: float, masks: dict, pruner: Pruner) -> None:
        row = {"setting": setting, "method": method, "ratio": ratio,
               **measure(model, pruner, masks, val_batches, spec.img_size, device)}
        results["records"].append(row)
        print(f"[{setting}] {method:13s} ratio={ratio:.1f} top1={row['top1']:.4f} GFLOPs={row['gflops.total']:.3f} "
              f"union={row['union_kept']:.2f} stored={row['stored_params']:,}", flush=True)
        (out_dir / "results.json").write_text(json.dumps(results, indent=1))

    for setting in args.settings:
        unit, targets = SETTINGS[setting]
        with Pruner(model, targets, Method(unit, "grad_gate", "per_use")) as pruner:
            stats = collect(model, pruner.wrappers, calib, device)
            diagnostics[setting] = {name: {"paper": s["paper"], "abs": s["abs"],
                                           "fisher": s["cross"].diagonal(dim1=1, dim2=2).T.clone(),
                                           # mean over units of the correlation of g between two steps
                                           "step_corr": _step_correlation(s["cross"])} for name, s in stats.items()}
            torch.save(diagnostics, out_dir / "diagnostics.pt")
            caches = {name: {} for name in stats}
            for method in ONE_SHOT:
                for ratio in args.ratios:
                    masks = {name: one_shot_masks(method, stats[name], ratio, caches[name]) for name in stats}
                    record(setting, method, ratio, masks, pruner)

            for method in ITERATIVE:
                keep = {name: torch.ones_like(s["paper"], dtype=torch.bool) for name, s in stats.items()}
                current = {name: {"cross": s["cross"]} for name, s in stats.items()}
                for ratio in args.ratios:
                    for name in keep:
                        cross, alive = current[name]["cross"], keep[name]
                        order = (order_joint(cross, alive) if method == "iter_joint"
                                 else order_by_score(cross.diagonal(dim1=1, dim2=2).T, alive))
                        extra = round(ratio * alive.numel()) - int((~alive).sum())
                        keep[name] = masks_from_order(order, tuple(alive.shape), extra, alive)
                    record(setting, method, ratio, {name: mask.clone() for name, mask in keep.items()}, pruner)
                    if ratio != args.ratios[-1]:
                        current = collect(model, pruner.wrappers, iter_calib, device, keep)
    print(f"saved: {out_dir}")


def _step_correlation(cross: torch.Tensor) -> torch.Tensor:
    """(T, T): correlation of the per-sample gate gradient of one unit between two steps, averaged over units."""
    scale = cross.diagonal(dim1=1, dim2=2).clamp_min(1e-30).sqrt()
    return (cross / (scale[:, :, None] * scale[:, None, :])).mean(dim=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--settings", nargs="+", default=list(SETTINGS), choices=list(SETTINGS))
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--calib-images", type=int, default=10240)
    parser.add_argument("--iter-calib-images", type=int, default=2560, help="images per re-scoring pass")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="./outputs/bvit_pruning_importance")
    parser.add_argument("--run-name", default=None)
    add_log_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    start_run_log(__file__, tag=arguments.run_name or Path(arguments.checkpoint).parent.name,
                  log_dir=arguments.log_dir, config=vars(arguments), enabled=not arguments.no_log)
    run(arguments)
