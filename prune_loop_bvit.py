"""Loop-aware structured pruning of a looped ViT (bViT): accuracy at a given FLOPs budget.

Units (pruning/channels.py): FFN neurons, QK channels, V channels. Each is removed per loop step,
the weights stay shared, and the compact model is really narrower (no low-rank side layers).

Methods (all reach the same FLOPs target; only the importance / allocation differs):
    paper_step     E|batch-mean gate gradient|, the same ratio in every step and layer   (bViT step-specific)
    paper_shared   the same, one mask for all steps                                       (bViT global)
    fisher_step    per-sample Fisher E g^2, the same ratio in every step and layer
    shared_joint   one mask for all steps, E (sum_t g_t)^2 / FLOPs, one ranking over layers (loop-unaware structured)
    alloc          Fisher / FLOPs, one ranking over all (step, layer, unit)
    joint          alloc + cross-step terms of the same unit: greedy on E (sum_{t removed} g_t)^2
    joint_iter     joint, re-scored on the pruned model every --rescore-every of the FLOPs  (proposed)
    joint_iter_floor  joint_iter, every step keeps at least --floor of each layer's units   (proposed, no step collapse)
    alloc_iter     alloc, re-scored (ablation: no cross-step terms)
    --mc-methods   second stage, see run_mc: removed units are replaced by their step-specific mean output and
                   the importance is measured against that mean (joint_iter_mc = final proposal)
    random_alloc   random units with the per-step, per-layer counts of joint_iter           (allocation only)
    steps          fewer loop steps (reference, not a pruning method here)

Settings: raw (channels as trained) / rot (heads rotated to the singular basis of proj_h v_h and q_h^T k_h).

Usage:
    uv run python prune_loop_bvit.py --checkpoint outputs/bvit/<run>/best.pth --run-name <name>
    uv run python prune_loop_bvit.py --checkpoint ... --run-name <name> --finetune joint_iter paper_step steps \
        --ft-targets 0.7 0.5 --ft-epochs 5

Outputs (./outputs/bvit_pruning_loop/<run>/): results[_seed].json, masks[_seed].pt, finetune.json
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from bisect import bisect_right
from itertools import islice
from pathlib import Path

import torch
from torch import nn

from dl_dataset.registry import build_loaders, get_spec
from model_profile import count_flops
from prune_compute_bvit import Meter
from prune_importance_bvit import collect, per_step_masks
from pruning import Pruner, StepLimited, max_output_difference
from pruning.channels import LoopGates, Masks, close_dead_heads, compact_channels, rotate_heads
from pruning.pruner import Method
from run_logging import add_log_args, link_run_log, start_run_log
from wandb_results import add_wandb_args, upload_results
from models.bViT.schedule import LoopSchedule
from train_bvit import Config, evaluate, load_model, parameter_groups, set_seed

TARGETS = ["FFN_IN", "Q", "V"]
ONE_SHOT = ["paper_step", "paper_shared", "fisher_step", "shared_joint", "alloc", "joint"]
ITERATIVE = {"alloc_iter": (False, 0.0), "joint_iter": (True, 0.0), "joint_iter_floor": (True, None)}  # (cross, floor)
ALL_METHODS = [*ONE_SHOT, *ITERATIVE, "random_alloc", "steps"]


# * ===== FLOPs model =====

class Cost:
    """FLOPs of a masked model as base - sum(cost[layer] * removed (step, unit) pairs).
    The per-unit costs are measured once with count_flops on half-width models."""

    def __init__(self, model: nn.Module, full: Masks, img_size: int, device: torch.device, num_heads: int) -> None:
        """full: all-True masks (layer name -> (steps, units)) of the prunable layers."""
        self.num_heads = num_heads
        self.base = count_flops(compact_channels(model, full), img_size, device)["gflops.total"]
        self.unit: dict[str, float] = {}
        for name, mask in full.items():
            half = mask.clone()
            half.view(mask.shape[0], -1, 2)[:, :, 1] = False  # every second unit: all heads keep the same width
            flops = count_flops(compact_channels(model, {**full, name: half}), img_size, device)["gflops.total"]
            self.unit[name] = (self.base - flops) / int((~half).sum())

    def flops(self, masks: Masks) -> float:
        masks = close_dead_heads(masks, self.num_heads)
        return self.base - sum(self.unit[name] * int((~mask).sum()) for name, mask in masks.items())


# * ===== Removal orders =====

def removal_order(stats: dict, alive: Masks, cost: Cost, *, cross: bool, floor: float) -> list[tuple[str, int, int]]:
    """Greedy order of (layer, step, unit) removals over all layers and steps, least harmful first.
    priority = marginal loss increase / FLOPs saved, with the marginal increase of removing unit i at
    step t given the already removed steps R of the same unit:  F_i[t, t] + 2 sum_{s in R} F_i[t, s]
    (F_i[t, s] = E_n g[n, t, i] g[n, s, i]; without `cross` only the diagonal is used).
    floor: every step keeps at least this fraction of the units of every layer."""
    names = list(stats)
    marginal = {name: stats[name]["cross"].diagonal(dim1=1, dim2=2).T.clone().double() for name in names}
    offsets, total = [], 0
    for name in names:
        offsets.append(total)
        total += alive[name].numel()
    priority = torch.cat([(marginal[name] / cost.unit[name]).masked_fill(~alive[name], math.inf).flatten()
                          for name in names])
    remaining = {name: alive[name].sum(dim=1).tolist() for name in names}
    min_keep = {name: math.ceil(floor * alive[name].shape[1]) for name in names}
    for position, name in enumerate(names):  # steps that are already at the floor
        steps, units = alive[name].shape
        for step in range(steps):
            if remaining[name][step] <= min_keep[name]:
                priority[offsets[position] + step * units: offsets[position] + (step + 1) * units] = math.inf
    order: list[tuple[str, int, int]] = []
    for _ in range(int(sum(mask.sum() for mask in alive.values()))):
        index = int(priority.argmin())
        if priority[index] == math.inf:
            break
        position = bisect_right(offsets, index) - 1
        name, offset = names[position], offsets[position]
        steps, units = alive[name].shape
        step, unit = divmod(index - offset, units)
        order.append((name, step, unit))
        priority[index] = math.inf
        if cross:
            column = marginal[name][:, unit]
            column += 2 * stats[name]["cross"][unit, :, step].double()
            slots = offset + torch.arange(steps) * units + unit
            old = priority[slots]
            priority[slots] = torch.where(old.isinf(), old, column / cost.unit[name])
        remaining[name][step] -= 1
        if remaining[name][step] <= min_keep[name]:
            priority[offset + step * units: offset + (step + 1) * units] = math.inf
    return order


def shared_order(stats: dict, cost: Cost) -> list[tuple[str, int]]:
    """One mask for all steps: units ranked by E (sum_t g_t)^2 / FLOPs over all layers."""
    scored = []
    for name, stat in stats.items():
        score = stat["cross"].sum(dim=(1, 2)).double() / (cost.unit[name] * stat["cross"].shape[1])
        scored += [(float(value), name, unit) for unit, value in enumerate(score)]
    return [(name, unit) for _, name, unit in sorted(scored)]


def round_widths(masks: Masks, scores: dict[str, torch.Tensor], multiple: int, num_heads: int) -> Masks:
    """Widths that are not multiples of 8 make fp16 matrix products slow. Per step, raise the kept count of
    every head (q / v) or of the layer (fc1) to the next multiple by restoring the best-scored removed units."""
    if multiple <= 1:
        return masks
    out = {}
    for name, mask in masks.items():
        groups = num_heads if ".attn." in name else 1
        kept = mask.clone().view(mask.shape[0], groups, -1)
        score = scores[name].view_as(kept).double()
        count = kept.sum(dim=2)
        goal = torch.where(count > 0, ((count + multiple - 1) // multiple * multiple).clamp_max(kept.shape[2]), count)
        rank = score.masked_fill(kept, -math.inf).argsort(dim=2, descending=True).argsort(dim=2)  # among removed
        kept |= rank < (goal - count)[:, :, None]
        out[name] = kept.view_as(mask)
    return out


def cut_to_target(apply_count, limit: int, cost: Cost, target_flops: float) -> Masks:
    """Smallest number of removals n (0..limit) whose masks apply_count(n) reach target_flops."""
    low, high = 0, limit
    while low < high:
        middle = (low + high) // 2
        if cost.flops(apply_count(middle)) <= target_flops:
            high = middle
        else:
            low = middle + 1
    return close_dead_heads(apply_count(low), cost.num_heads)


def masks_after(order: list, keep: Masks, count: int) -> Masks:
    masks = {name: mask.clone() for name, mask in keep.items()}
    for item in order[:count]:
        if len(item) == 3:
            masks[item[0]][item[1], item[2]] = False
        else:
            masks[item[0]][:, item[1]] = False
    return masks


def uniform_masks(scores: dict[str, torch.Tensor], cost: Cost, target_flops: float, multiple: int) -> Masks:
    """The same ratio in every step of every layer (bViT step-specific), ratio found by bisection."""
    def build(ratio: float) -> Masks:
        return round_widths({name: per_step_masks(score, ratio) for name, score in scores.items()},
                            scores, multiple, cost.num_heads)

    low, high = 0.0, 1.0
    for _ in range(30):
        middle = (low + high) / 2
        if cost.flops(build(middle)) <= target_flops:
            high = middle
        else:
            low = middle
    return close_dead_heads(build(high), cost.num_heads)


def random_like(masks: Masks, generator: torch.Generator, num_heads: int) -> Masks:
    """Random units, the same number kept in every (layer, step) as in `masks`."""
    out = {}
    for name, mask in masks.items():
        random = torch.zeros_like(mask)
        for step, kept in enumerate(mask.sum(dim=1).tolist()):
            random[step, torch.randperm(mask.shape[1], generator=generator)[:kept]] = True
        if ".attn." in name:  # keep the per-head counts (multiples of 8): permute inside every head
            random = torch.zeros_like(mask).view(mask.shape[0], num_heads, -1)
            for (step, head), kept in zip(torch.cartesian_prod(torch.arange(mask.shape[0]), torch.arange(num_heads)),
                                          mask.view(mask.shape[0], num_heads, -1).sum(dim=2).flatten().tolist()):
                random[step, head, torch.randperm(random.shape[2], generator=generator)[:kept]] = True
            random = random.view_as(mask)
        out[name] = random
    return close_dead_heads(out, num_heads)


# * ===== Fine-tuning with fixed masks =====

def finetune(model: nn.Module, forward, train_loader, config: Config, epochs: int, lr: float,
             device: torch.device, max_batches: int | None = None) -> None:
    """AdamW + cosine, label smoothing, the training augmentation; no mixup / EMA. `forward(images)` gives logits."""
    optimizer = torch.optim.AdamW(parameter_groups(model, config.weight_decay), lr=lr)
    total = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 0.5 * (1 + math.cos(math.pi * s / total)))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    loss_fn = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    model.train()
    for epoch in range(epochs):
        running, seen = 0.0, 0
        for images, targets in islice(train_loader, max_batches):
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = loss_fn(forward(images).float(), targets)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += float(loss) * len(images)
            seen += len(images)
        print(f"    epoch {epoch + 1}/{epochs} train loss {running / seen:.4f}", flush=True)
    model.eval()


# * ===== Main =====

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
    num_heads = model.blocks[0].attn.num_heads
    num_steps = model.schedule.num_steps

    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    link_run_log(out_dir)
    original = copy.deepcopy(model)
    base = evaluate(model, val_batches, device)
    base_flops = count_flops(model, spec.img_size, device)["gflops.total"]
    print(f"unpruned: top1={base['top1']:.4f} GFLOPs={base_flops:.3f}")
    meter = Meter(original, val_batches, spec.img_size, device, args)
    reference = original  # FLOPs targets and relative throughput always refer to the full loop
    if args.max_steps:  # combine with fewer loop steps: prune the model that stops after --max-steps
        original = copy.deepcopy(original)
        original.schedule = LoopSchedule.parse(f"0x{args.max_steps}")
        model.schedule = original.schedule
        num_steps = args.max_steps
        short = evaluate(original, val_batches, device)["top1"]
        print(f"{args.max_steps} steps: top1={short:.4f} "
              f"GFLOPs={count_flops(original, spec.img_size, device)['gflops.total']:.3f}")

    if args.mc_methods:
        run_mc(args, original, calib, iter_calib, spec, val_batches, base, base_flops, out_dir, device, reference)
        return
    if args.finetune:
        run_finetune(args, original, train_config, spec, val_batches, base, base_flops, out_dir, device)
        return

    suffix = ("" if args.seed == 42 else f"_seed{args.seed}") + ("" if args.round == 8 else f"_round{args.round}")
    results_path = out_dir / f"results{suffix}.json"
    results = {"checkpoint": str(args.checkpoint), "model": train_config.model, "dataset": spec.name,
               "args": vars(args), "records": [], "unpruned": {"top1": base["top1"], "gflops.total": base_flops}}
    saved_masks: dict = {}
    if results_path.exists():  # keep the settings that are not run again
        results["records"] = [r for r in json.loads(results_path.read_text())["records"]
                              if r["setting"] not in args.settings]
        if (out_dir / f"masks{suffix}.pt").exists():
            saved_masks = {key: value for key, value in torch.load(out_dir / f"masks{suffix}.pt").items()
                           if key[0] not in args.settings}

    def save() -> None:
        results_path.write_text(json.dumps(results, indent=1))
        torch.save(saved_masks, out_dir / f"masks{suffix}.pt")

    if "steps" in args.methods:
        results["records"] = [r for r in results["records"] if r["method"] != "steps"]
        for steps in range(num_steps - 1, 2, -1):
            limited = StepLimited(original, steps)
            row = {"setting": "raw", "method": "steps", "target": None, "steps": steps,
                   "top1": evaluate(limited, val_batches, device)["top1"],
                   "gflops.total": count_flops(limited, spec.img_size, device)["gflops.total"]}
            if args.throughput:
                row.update({k: v for k, v in meter.compute(limited).items() if k.startswith(("rel_", "img_per"))})
            results["records"].append(row)
            print(f"[steps] {steps:2d} top1={row['top1']:.4f} GFLOPs={row['gflops.total']:.3f}", flush=True)
        save()

    for setting in args.settings:
        if setting == "rot":
            rotate_heads(model)
            print(f"rotation: max output difference {max_output_difference(original, model, val_batches[0][0][:64]):.2e}")
        with Pruner(model, TARGETS, Method("neuron", "grad_gate", "per_use")) as pruner:
            cost = Cost(model, {name: torch.ones(w.num_uses, w.num_units, dtype=torch.bool)
                                for name, w in pruner.wrappers.items()}, spec.img_size, device, num_heads)
            print(f"[{setting}] FLOPs per removed (step, unit): "
                  + ", ".join(f"{name.split('.')[-1]}={value * 1e9:,.0f}" for name, value in cost.unit.items()))
            start = time.perf_counter()
            stats = collect(model, pruner.wrappers, calib, device)
            print(f"[{setting}] scoring {len(calib) * args.batch_size} images: {time.perf_counter() - start:.0f}s")
            full = {name: torch.ones_like(stat["paper"], dtype=torch.bool) for name, stat in stats.items()}

            def record(method: str, target: float, masks: Masks) -> None:
                pruner.apply(masks)
                metrics = evaluate(model, val_batches, device)
                small = compact_channels(model, masks)
                row = {"setting": setting, "method": method, "target": target, "top1": metrics["top1"],
                       "loss": metrics["loss"], "gflops.total": count_flops(small, spec.img_size, device)["gflops.total"],
                       "gflops.model": cost.flops(masks),
                       "kept_per_step": {name.split(".")[-1]: mask.float().mean(dim=1).tolist()
                                         for name, mask in masks.items()},
                       "union_kept": {name.split(".")[-1]: float(mask.any(dim=0).float().mean())
                                      for name, mask in masks.items()},
                       "mask_bits": sum(mask.numel() for mask in masks.values())}
                if args.throughput and method in args.throughput:
                    row["max_diff"] = max_output_difference(model, small, val_batches[0][0][:64])
                    row.update({k: v for k, v in meter.compute(small).items() if k.startswith(("rel_", "img_per"))})
                pruner.reset()
                results["records"].append(row)
                saved_masks[(setting, method, target)] = masks
                print(f"[{setting}] {method:16s} target={target:.2f} top1={row['top1']:.4f} "
                      f"GFLOPs={row['gflops.total']:.3f}"
                      + (f" rel_thr={row['rel_throughput']:.2f}" if "rel_throughput" in row else ""), flush=True)
                save()

            fisher = {name: stat["cross"].diagonal(dim1=1, dim2=2).T for name, stat in stats.items()}
            paper = {name: stat["paper"] for name, stat in stats.items()}
            orders: dict[str, list] = {}
            for method in [m for m in ONE_SHOT if m in args.methods]:
                for target in args.targets:
                    goal = target * base_flops
                    if method == "paper_step":
                        masks = uniform_masks(paper, cost, goal, args.round)
                    elif method == "fisher_step":
                        masks = uniform_masks(fisher, cost, goal, args.round)
                    elif method == "paper_shared":
                        masks = uniform_masks({name: score.mean(dim=0, keepdim=True).expand_as(score)
                                               for name, score in paper.items()}, cost, goal, args.round)
                    else:
                        if method not in orders:
                            orders[method] = (shared_order(stats, cost) if method == "shared_joint" else
                                              removal_order(stats, full, cost, cross=method == "joint", floor=0.0))
                        order = orders[method]
                        tie = ({name: stat["cross"].sum(dim=(1, 2))[None].expand_as(fisher[name])
                                for name, stat in stats.items()} if method == "shared_joint" else fisher)
                        masks = cut_to_target(lambda n: round_widths(masks_after(order, full, n), tie, args.round,
                                                                     num_heads), len(order), cost, goal)
                    record(method, target, masks)

            for method, (cross, floor) in ITERATIVE.items():
                if method not in args.methods:
                    continue
                floor = args.floor if floor is None else floor
                keep, stat = full, stats
                stages = sorted({round(1 - i * args.rescore_every, 4) for i in range(1, int(1 / args.rescore_every))}
                                | set(args.targets), reverse=True)
                for stage in [s for s in stages if s >= min(args.targets)]:
                    order = removal_order(stat, keep, cost, cross=cross, floor=floor)
                    now = {name: s["cross"].diagonal(dim1=1, dim2=2).T for name, s in stat.items()}
                    keep = cut_to_target(lambda n: round_widths(masks_after(order, keep, n), now, args.round,
                                                                num_heads), len(order), cost, stage * base_flops)
                    if stage in args.targets:
                        record(method, stage, keep)
                        if method == "joint_iter" and "random_alloc" in args.methods:
                            generator = torch.Generator().manual_seed(args.seed)
                            record("random_alloc", stage, random_like(keep, generator, num_heads))
                    if stage > min(args.targets):
                        stat = collect(model, pruner.wrappers, iter_calib, device, keep)
    print(f"saved: {out_dir}")


def run_mc(args, original, calib, iter_calib, spec, val_batches, base, base_flops, out_dir, device,
           reference=None) -> None:
    """Mean-compensated variants (LoopGates): a removed unit is replaced by its step-specific mean output.

        <method>+bc     masks of the main call, means measured afterwards (bias correction only)
        paper_step_mc   bViT score measured against the mean, equal ratio per step
        joint_mc        one-shot: Fisher against the mean + cross-step terms, one ranking / FLOPs
        joint_iter_mc   joint_mc re-scored every --rescore-every of the FLOPs               (proposed, final)
    Accuracy is that of the compact model with the means folded into step-specific biases."""
    suffix = "" if args.seed == 42 else f"_seed{args.seed}"
    v1_path = out_dir / f"masks{suffix}.pt"
    v1_masks = torch.load(v1_path) if v1_path.exists() else {}
    path = out_dir / f"results_mc{suffix}.json"
    results = {"unpruned": {"top1": base["top1"], "gflops.total": base_flops}, "args": vars(args), "records": []}
    saved: dict = {}
    if path.exists():
        results["records"] = [r for r in json.loads(path.read_text())["records"] if r["setting"] not in args.settings]
        saved = {k: v for k, v in torch.load(out_dir / f"masks_mc{suffix}.pt").items() if k[0] not in args.settings}
    meter = Meter(reference or original, val_batches, spec.img_size, device, args)
    num_heads = original.blocks[0].attn.num_heads

    for setting in args.settings:
        plain = copy.deepcopy(original)
        if setting == "rot":
            rotate_heads(plain)
        sim = copy.deepcopy(plain)
        gates = LoopGates(sim)
        full = {name: torch.ones(shape, dtype=torch.bool) for name, shape in gates.shapes().items()}
        cost = Cost(plain, full, spec.img_size, device, num_heads)

        def record(method: str, target: float, masks: Masks) -> None:
            means = gates.get_means()
            small = compact_channels(plain, masks, means=means)
            metrics = evaluate(small, val_batches, device)
            row = {"setting": setting, "method": method, "target": target, "top1": metrics["top1"],
                   "loss": metrics["loss"], "gflops.total": count_flops(small, spec.img_size, device)["gflops.total"],
                   "max_diff": max_output_difference(sim, small, val_batches[0][0][:64]),
                   "kept_per_step": {n.split(".")[-1]: m.float().mean(dim=1).tolist() for n, m in masks.items()},
                   "union_kept": {n.split(".")[-1]: float(m.any(dim=0).float().mean()) for n, m in masks.items()},
                   "mask_bits": sum(m.numel() for m in masks.values()),
                   "bias_params": sum(m.numel() for m in means.values())}
            if method in args.throughput:
                row.update({k: v for k, v in meter.compute(small).items() if k.startswith(("rel_", "img_per"))})
            results["records"].append(row)
            saved[(setting, method, target)] = {"masks": masks, "means": means}
            print(f"[{setting}] {method:16s} target={target:.2f} top1={row['top1']:.4f} GFLOPs={row['gflops.total']:.3f}"
                  + (f" rel_thr={row['rel_throughput']:.2f}" if "rel_throughput" in row else ""), flush=True)
            path.write_text(json.dumps(results, indent=1))
            torch.save(saved, out_dir / f"masks_mc{suffix}.pt")

        def settle(masks: Masks) -> None:
            gates.set_masks(masks)
            gates.measure_means(iter_calib)

        for method in args.mc_methods:
            if not method.endswith("+bc"):
                continue
            for target in args.targets:
                key = (setting, method[:-3], target)
                if key in v1_masks:
                    gates.zero_means()
                    settle(v1_masks[key])
                    record(method, target, v1_masks[key])

        gates.zero_means()
        settle(full)
        stats = gates.score(calib)
        if "paper_step_mc" in args.mc_methods:
            for target in args.targets:
                masks = uniform_masks({name: stat["paper"] for name, stat in stats.items()}, cost,
                                      target * base_flops, args.round)
                settle(masks)
                record("paper_step_mc", target, masks)
        if "joint_mc" in args.mc_methods:
            order = removal_order(stats, full, cost, cross=True, floor=0.0)
            fisher = {name: stat["cross"].diagonal(dim1=1, dim2=2).T for name, stat in stats.items()}
            for target in args.targets:
                masks = cut_to_target(lambda n: round_widths(masks_after(order, full, n), fisher, args.round, num_heads),
                                      len(order), cost, target * base_flops)
                settle(masks)
                record("joint_mc", target, masks)
        for method, cross in (("alloc_iter_mc", False), ("joint_iter_mc", True)):
            if method not in args.mc_methods:
                continue
            gates.zero_means()
            settle(full)
            keep, stat = full, stats
            stages = sorted({round(1 - i * args.rescore_every, 4) for i in range(1, int(1 / args.rescore_every))}
                            | set(args.targets), reverse=True)
            for stage in [s for s in stages if s >= min(args.targets)]:
                order = removal_order(stat, keep, cost, cross=cross, floor=0.0)
                now = {name: s["cross"].diagonal(dim1=1, dim2=2).T for name, s in stat.items()}
                keep = cut_to_target(lambda n: round_widths(masks_after(order, keep, n), now, args.round, num_heads),
                                     len(order), cost, stage * base_flops)
                settle(keep)
                if stage in args.targets:
                    record(method, stage, keep)
                if stage > min(args.targets):
                    stat = gates.score(iter_calib)
        gates.remove()
    print(f"saved: {out_dir}")


def run_finetune(args, original, train_config, spec, val_batches, base, base_flops, out_dir, device) -> None:
    """Fine-tune the shared weights with masks (and means) of earlier calls fixed. One recipe for every method.
    Jobs: "setting/method", e.g. rot/joint_iter_mc raw/paper_step raw/steps; the unpruned model is the control."""
    train_loader, _, _ = build_loaders(spec, batch_size=args.ft_batch_size, num_workers=args.num_workers,
                                       augment=train_config.augment, re_prob=train_config.re_prob)
    v1 = torch.load(out_dir / "masks.pt")
    mc = torch.load(out_dir / "masks_mc.pt") if (out_dir / "masks_mc.pt").exists() else {}
    path = out_dir / "finetune.json"
    records = json.loads(path.read_text())["records"] if path.exists() else []
    done = {(r["setting"], r["method"], r["target"]) for r in records}
    jobs = [("raw", "unpruned", 1.0)] + [(*job.split("/"), target) for target in args.ft_targets for job in args.finetune]
    for setting, method, target in jobs:
        if (setting, method, target) in done:
            continue
        model = copy.deepcopy(original)
        if setting == "rot":
            rotate_heads(model)
        start = time.perf_counter()
        row = {"setting": setting, "method": method, "target": target, "epochs": args.ft_epochs, "lr": args.ft_lr}
        if method in ("unpruned", "steps"):
            steps = None
            if method == "steps":  # the largest step count that fits the budget
                flops = {s: count_flops(StepLimited(model, s), spec.img_size, device)["gflops.total"]
                         for s in range(1, model.schedule.num_steps + 1)}
                steps = max(s for s, f in flops.items() if f <= target * base_flops * 1.01)
                row["steps"] = steps
            net = model if steps is None else StepLimited(model, steps)
            row["top1_before"] = evaluate(net, val_batches, device)["top1"]
            finetune(model, lambda images: model(images, max_steps=steps), train_loader, train_config,
                     args.ft_epochs, args.ft_lr, device, args.ft_max_batches)
            row["top1"] = evaluate(net, val_batches, device)["top1"]
            row["gflops.total"] = count_flops(net, spec.img_size, device)["gflops.total"]
        else:
            entry = mc[(setting, method, target)] if (setting, method, target) in mc else \
                {"masks": v1[(setting, method, target)], "means": None}
            masks, means = entry["masks"], entry["means"]
            plain = copy.deepcopy(model)  # weights before fine-tuning, only for the "before" accuracy
            row["top1_before"] = evaluate(compact_channels(plain, masks, means=means), val_batches, device)["top1"]
            gates = LoopGates(model)
            gates.set_masks(masks, means)
            finetune(model, model, train_loader, train_config, args.ft_epochs, args.ft_lr, device, args.ft_max_batches)
            gates.remove()
            small = compact_channels(model, masks, means=means)
            row["top1"] = evaluate(small, val_batches, device)["top1"]
            row["gflops.total"] = count_flops(small, spec.img_size, device)["gflops.total"]
        row["minutes"] = (time.perf_counter() - start) / 60
        records.append(row)
        print(f"[finetune] {setting}/{method} target={target} top1 {row['top1_before']:.4f} -> {row['top1']:.4f} "
              f"GFLOPs={row['gflops.total']:.3f} ({row['minutes']:.1f} min)", flush=True)
        path.write_text(json.dumps({"unpruned": {"top1": base["top1"], "gflops.total": base_flops},
                                    "args": vars(args), "records": records}, indent=1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--settings", nargs="+", default=["raw", "rot"], choices=["raw", "rot"])
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    parser.add_argument("--targets", nargs="+", type=float, default=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
                        help="FLOPs relative to the unpruned model")
    parser.add_argument("--rescore-every", type=float, default=0.05, help="FLOPs fraction between two re-scorings")
    parser.add_argument("--floor", type=float, default=0.1,
                        help="joint_iter_floor: fraction of each layer's units every step keeps")
    parser.add_argument("--round", type=int, default=8,
                        help="kept widths per step (per head for q / v) are raised to a multiple of this; 1 = off")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="prune the model that stops after this many loop steps (targets stay relative to the full loop)")
    parser.add_argument("--calib-images", type=int, default=10240)
    parser.add_argument("--iter-calib-images", type=int, default=2560, help="images per re-scoring pass")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--seed", type=int, default=42, help="also selects the calibration images")
    parser.add_argument("--output-dir", default="./outputs/bvit_pruning_loop")
    # throughput (interleaved with the unpruned model)
    parser.add_argument("--throughput", nargs="*", default=[], help="methods whose compact models are timed")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--infer-batch", type=int, default=256)
    parser.add_argument("--iters", type=int, default=15)
    parser.add_argument("--latency-iters", type=int, default=30)
    parser.add_argument("--mc-methods", nargs="*", default=[],
                        help="mean-compensated stage (needs the masks of a main call): paper_step+bc joint_iter+bc "
                             "paper_step_mc joint_mc alloc_iter_mc joint_iter_mc")
    # fine-tuning
    parser.add_argument("--finetune", nargs="*", default=[],
                        help="setting/method jobs to fine-tune (masks from earlier calls), e.g. rot/joint_iter_mc raw/steps")
    parser.add_argument("--ft-targets", nargs="+", type=float, default=[0.7, 0.5])
    parser.add_argument("--ft-epochs", type=int, default=5)
    parser.add_argument("--ft-lr", type=float, default=1e-4)
    parser.add_argument("--ft-batch-size", type=int, default=256)
    parser.add_argument("--ft-max-batches", type=int, default=None, help="per epoch, for smoke tests")
    add_log_args(parser)
    add_wandb_args(parser)
    return parser.parse_args()


def upload(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir) / args.run_name
    if args.finetune:
        upload_results(args, job_type="prune_loop_finetune", group=args.run_name, files=[out_dir / "finetune.json"])
        return
    seed = "" if args.seed == 42 else f"_seed{args.seed}"
    suffix = seed + ("" if args.round == 8 else f"_round{args.round}")
    upload_results(args, job_type="prune_loop", group=args.run_name,
                   files=[out_dir / f"results{suffix}.json", out_dir / f"masks{suffix}.pt",
                          *([out_dir / f"results_mc{seed}.json", out_dir / f"masks_mc{seed}.pt"] if args.mc_methods else [])])


if __name__ == "__main__":
    arguments = parse_args()
    start_run_log(__file__, tag=arguments.run_name, log_dir=arguments.log_dir, config=vars(arguments),
                  enabled=not arguments.no_log)
    run(arguments)
    upload(arguments)
