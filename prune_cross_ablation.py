"""Ablation of the cross-step Fisher F_i[t, s] = E_n g[n, t, i] g[n, s, i] of the proposed method
(rot, mean replacement, one-shot = the setting of rot joint_mc in prune_loop_bvit.py).

Part A, accuracy at the same FLOPs  -> <run>/results_cross[_seed].json
    joint_mc       proposed: F_i[t, t] + cross-step terms F_i[t, s], one ranking over (step, layer, unit) / FLOPs
    alloc_mc       the same ranking with the diagonal only (cross-step terms off)
    sel_*_mc       how many units every (step, layer) keeps (per head for q / v) is copied from joint_mc;
                   only WHICH units are kept is chosen by another score:
                       sel_fisher_mc  F_i[t, t]                 (Fisher, no cross-step terms)
                       sel_paper_mc   mean over batches |batch-mean g|  (bViT paper)
                       sel_random_mc  random (--random-draws draws)
Part B, oracle  -> <run>/oracle_cross[_seed].json
    Replace one unit i by its step mean at a set of steps S and measure the loss change dL on a fixed image set
    (fp32, train images), against the second-order predictions
        paper  sum_{t in S} |E g_t|            diag   1/2 sum_{t in S} F[t, t]
        full   1/2 sum_{t, s in S} F[t, s]     full1  -E sum_{t in S} g_t + full   (with the first-order term)
    and, for pairs of steps, the measured interaction dL({t, s}) - dL({t}) - dL({s}) against F[t, s].
    Statistics: Spearman / Pearson over events, per |S|.

    uv run python prune_cross_ablation.py --checkpoint outputs/bvit/bvit_s_tiny_imagenet_20261111_190414/best.pth \
        --run-name bvit_s_tiny_imagenet
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from itertools import islice
from pathlib import Path

import torch
import torch.nn.functional as F

from dl_dataset.registry import build_loaders, get_spec
from model_profile import count_flops
from prune_loop_bvit import Cost, cut_to_target, masks_after, random_like, removal_order, round_widths
from pruning.channels import LoopGates, Masks, close_dead_heads, compact_channels, rotate_heads
from run_logging import add_log_args, link_run_log, start_run_log
from train_bvit import Config, evaluate, load_model, set_seed
from wandb_results import add_wandb_args, upload_results

SELECT = ["sel_fisher_mc", "sel_paper_mc", "sel_random_mc"]


# * ===== Scores =====

def score(gates: LoopGates, batches: list) -> dict[str, dict[str, torch.Tensor]]:
    """LoopGates.score plus the signed mean E_n g[n, t, i] (first-order term), fp32."""
    model = gates.model
    model.eval()
    flags = {p: p.requires_grad for p in model.parameters()}
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    stats = {key: {"paper": torch.zeros(gates.steps, units, device=gates.device),
                   "mean": torch.zeros(gates.steps, units, device=gates.device),
                   "cross": torch.zeros(units, gates.steps, gates.steps, device=gates.device)}
             for key, units in gates.units.items()}
    seen = 0
    try:
        for images, targets in batches:
            images, targets = images.to(gates.device), targets.to(gates.device)
            gates.gates = {key: mask[:, None, None, :].repeat(1, len(images), 1, 1).requires_grad_()
                           for key, mask in gates.masks.items()}
            gates.mode = "score"
            F.cross_entropy(model(images).float(), targets, reduction="sum").backward()
            for key in gates.units:
                grad = gates.gates[key].grad.squeeze(2)  # (steps, batch, units)
                stats[key]["paper"] += grad.mean(dim=1).abs()
                stats[key]["mean"] += grad.sum(dim=1)
                stats[key]["cross"] += torch.einsum("tbi,sbi->its", grad, grad)
            seen += len(images)
    finally:
        gates.gates, gates.mode = {}, "mask"
        for parameter, flag in flags.items():
            parameter.requires_grad_(flag)
    return {gates.names[key]: {"paper": s["paper"].cpu() / len(batches), "mean": s["mean"].cpu() / seen,
                               "cross": s["cross"].cpu() / seen} for key, s in stats.items()}


def select_like(reference: Masks, scores: dict[str, torch.Tensor], num_heads: int) -> Masks:
    """Keep the highest-scored units with the per-(step, layer) counts of `reference` (per head for q / v)."""
    out = {}
    for name, mask in reference.items():
        groups = num_heads if ".attn." in name else 1
        kept = mask.view(mask.shape[0], groups, -1)
        rank = scores[name].double().view_as(kept).argsort(dim=2, descending=True).argsort(dim=2)
        out[name] = (rank < kept.sum(dim=2, keepdim=True)).view_as(mask)
    return close_dead_heads(out, num_heads)


# * ===== Part B: oracle =====

def spearman(x: list[float], y: list[float]) -> float:
    def ranks(v):
        t = torch.tensor(v, dtype=torch.float64)
        return t.argsort().argsort().double()
    return pearson(ranks(x).tolist(), ranks(y).tolist())


def pearson(x: list[float], y: list[float]) -> float:
    a, b = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-30))


@torch.no_grad()
def mean_loss(model, batches: list, device) -> float:
    total, count = 0.0, 0
    for images, targets in batches:
        total += float(F.cross_entropy(model(images.to(device)).float(), targets.to(device), reduction="sum"))
        count += len(images)
    return total / count


def oracle(gates: LoopGates, stats: dict, batches: list, args, device) -> dict:
    """Single-unit removals at random step sets, and pairs of steps for the interaction."""
    rng = random.Random(args.seed)
    keys = {name: key for key, name in gates.names.items()}
    names = list(stats)
    steps = gates.steps
    base = mean_loss(gates.model, batches, device)
    noise = abs(mean_loss(gates.model, batches, device) - base)

    def loss_without(name: str, unit: int, removed: list[int]) -> float:
        row = gates.masks[keys[name]]
        row[removed, unit] = 0.0
        value = mean_loss(gates.model, batches, device)
        row[removed, unit] = 1.0
        return value - base

    events, pairs = [], []
    sizes = [1, 2, 3, 4, 6, 8, 12]
    start = time.perf_counter()
    for index in range(args.oracle_events):
        name = rng.choice(names)
        unit = rng.randrange(stats[name]["cross"].shape[0])
        removed = sorted(rng.sample(range(steps), sizes[index % len(sizes)]))
        cross = stats[name]["cross"][unit].double()
        sub = cross[removed][:, removed]
        diag = 0.5 * float(sub.diagonal().sum())
        full = 0.5 * float(sub.sum())
        first = -float(stats[name]["mean"][removed, unit].double().sum())
        events.append({"layer": name.split(".")[-1], "unit": unit, "size": len(removed), "steps": removed,
                       "actual": loss_without(name, unit, removed),
                       "paper": float(stats[name]["paper"][removed, unit].double().sum()),
                       "diag": diag, "full": full, "full1": first + full, "cross_part": full - diag})
    print(f"oracle: {len(events)} events in {time.perf_counter() - start:.0f}s", flush=True)
    start = time.perf_counter()
    for _ in range(args.oracle_pairs):
        name = rng.choice(names)
        unit = rng.randrange(stats[name]["cross"].shape[0])
        t, s = sorted(rng.sample(range(steps), 2))
        single_t, single_s = loss_without(name, unit, [t]), loss_without(name, unit, [s])
        both = loss_without(name, unit, [t, s])
        pairs.append({"layer": name.split(".")[-1], "unit": unit, "t": t, "s": s, "gap": s - t,
                      "interaction": both - single_t - single_s,
                      "predicted": float(stats[name]["cross"][unit, t, s]),
                      "single_t": single_t, "single_s": single_s, "both": both})
    print(f"oracle: {len(pairs)} pairs in {time.perf_counter() - start:.0f}s", flush=True)

    summary = {"base_loss": base, "repeat_noise": noise, "events": len(events), "pairs": len(pairs)}
    groups = {"all": events, "size1": [e for e in events if e["size"] == 1],
              "size2+": [e for e in events if e["size"] >= 2]}
    for group, rows in groups.items():
        if len(rows) < 3:
            continue
        actual = [e["actual"] for e in rows]
        for predictor in ("paper", "diag", "full", "full1"):
            summary[f"{group}/spearman_{predictor}"] = spearman([e[predictor] for e in rows], actual)
            summary[f"{group}/pearson_{predictor}"] = pearson([e[predictor] for e in rows], actual)
        # how much of the actual change the cross part explains beyond the diagonal
        summary[f"{group}/pearson_cross_vs_residual"] = pearson([e["cross_part"] for e in rows],
                                                                [e["actual"] - e["diag"] for e in rows])
    if len(pairs) >= 3:
        summary["pairs/pearson"] = pearson([p["predicted"] for p in pairs], [p["interaction"] for p in pairs])
        summary["pairs/spearman"] = spearman([p["predicted"] for p in pairs], [p["interaction"] for p in pairs])
        summary["pairs/sign_agreement"] = sum((p["predicted"] > 0) == (p["interaction"] > 0) for p in pairs) / len(pairs)
        for gap_name, rows in (("adjacent", [p for p in pairs if p["gap"] == 1]),
                               ("far", [p for p in pairs if p["gap"] >= 4])):
            if len(rows) >= 3:
                summary[f"pairs/{gap_name}_pearson"] = pearson([p["predicted"] for p in rows],
                                                               [p["interaction"] for p in rows])
                summary[f"pairs/{gap_name}_count"] = len(rows)
    for key, value in summary.items():
        print(f"  {key}: {value:.4g}" if isinstance(value, float) else f"  {key}: {value}")
    return {"summary": summary, "records": events, "pairs": pairs}


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
    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    link_run_log(out_dir)
    suffix = "" if args.seed == 42 else f"_seed{args.seed}"
    base = evaluate(model, val_batches, device)
    base_flops = count_flops(model, spec.img_size, device)["gflops.total"]
    print(f"unpruned: top1={base['top1']:.4f} GFLOPs={base_flops:.3f}")

    plain = copy.deepcopy(model)
    rotate_heads(plain)
    sim = copy.deepcopy(plain)
    gates = LoopGates(sim)
    full = {name: torch.ones(shape, dtype=torch.bool) for name, shape in gates.shapes().items()}
    cost = Cost(plain, full, spec.img_size, device, num_heads)
    gates.zero_means()
    gates.set_masks(full)
    gates.measure_means(iter_calib)
    full_means = gates.get_means()
    start = time.perf_counter()
    stats = score(gates, calib)
    print(f"scoring {len(calib) * args.batch_size} images: {time.perf_counter() - start:.0f}s", flush=True)

    if not args.skip_accuracy:
        path = out_dir / f"results_cross{suffix}.json"
        results = {"checkpoint": str(args.checkpoint), "unpruned": {"top1": base["top1"], "gflops.total": base_flops},
                   "args": vars(args), "records": []}

        def record(method: str, target: float, masks: Masks, **extra) -> None:
            gates.set_masks(masks)
            gates.measure_means(iter_calib)  # means under the pruned model, as in run_mc
            small = compact_channels(plain, masks, means=gates.get_means())
            metrics = evaluate(small, val_batches, device)
            row = {"setting": "rot", "method": method, "target": target, "top1": metrics["top1"],
                   "loss": metrics["loss"], "gflops.total": count_flops(small, spec.img_size, device)["gflops.total"],
                   "kept_per_step": {n.split(".")[-1]: m.float().mean(dim=1).tolist() for n, m in masks.items()},
                   "overlap_with_joint_mc": extra.pop("overlap", None), **extra}
            results["records"].append(row)
            print(f"[rot] {method:14s} target={target:.2f} top1={row['top1']:.4f} GFLOPs={row['gflops.total']:.3f}"
                  + (f" overlap={row['overlap_with_joint_mc']:.3f}" if row["overlap_with_joint_mc"] is not None else ""),
                  flush=True)
            path.write_text(json.dumps(results, indent=1))

        def overlap(a: Masks, b: Masks) -> float:
            """Fraction of the removed (step, unit) pairs of b that a removes too."""
            removed_b = sum(int((~m).sum()) for m in b.values())
            both = sum(int((~a[n] & ~b[n]).sum()) for n in b)
            return both / max(1, removed_b)

        fisher = {name: stat["cross"].diagonal(dim1=1, dim2=2).T for name, stat in stats.items()}
        paper = {name: stat["paper"] for name, stat in stats.items()}
        orders = {"joint_mc": removal_order(stats, full, cost, cross=True, floor=0.0),
                  "alloc_mc": removal_order(stats, full, cost, cross=False, floor=0.0)}
        generator = torch.Generator().manual_seed(args.seed)
        for target in args.targets:
            built = {method: cut_to_target(lambda n, o=order: round_widths(masks_after(o, full, n), fisher,
                                                                            args.round, num_heads),
                                           len(order), cost, target * base_flops)
                     for method, order in orders.items()}
            reference = built["joint_mc"]
            record("joint_mc", target, reference)
            record("alloc_mc", target, built["alloc_mc"], overlap=overlap(built["alloc_mc"], reference))
            record("sel_fisher_mc", target, (m := select_like(reference, fisher, num_heads)), overlap=overlap(m, reference))
            record("sel_paper_mc", target, (m := select_like(reference, paper, num_heads)), overlap=overlap(m, reference))
            for draw in range(args.random_draws):
                m = random_like(reference, generator, num_heads)
                record("sel_random_mc", target, m, overlap=overlap(m, reference), draw=draw)
        print(f"saved: {path}")

    if args.oracle_events or args.oracle_pairs:
        gates.set_masks(full, full_means)
        batches = calib[: max(1, args.oracle_images // args.batch_size)]
        result = oracle(gates, stats, batches, args, device)
        result["args"] = vars(args)
        (out_dir / f"oracle_cross{suffix}.json").write_text(json.dumps(result, indent=1))
        print(f"saved: {out_dir / f'oracle_cross{suffix}.json'}")
    gates.remove()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-name", required=True, help="output folder under --output-dir (same as prune_loop_bvit)")
    parser.add_argument("--targets", nargs="+", type=float, default=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
    parser.add_argument("--round", type=int, default=8)
    parser.add_argument("--random-draws", type=int, default=3)
    parser.add_argument("--skip-accuracy", action="store_true", help="only part B")
    parser.add_argument("--oracle-events", type=int, default=420, help="0 = skip part B single-unit events")
    parser.add_argument("--oracle-pairs", type=int, default=300, help="0 = skip part B pairs")
    parser.add_argument("--oracle-images", type=int, default=2048, help="train images for the measured loss")
    parser.add_argument("--calib-images", type=int, default=10240)
    parser.add_argument("--iter-calib-images", type=int, default=2560)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--seed", type=int, default=42, help="also selects the calibration images")
    parser.add_argument("--output-dir", default="./outputs/bvit_pruning_loop")
    add_log_args(parser)
    add_wandb_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    start_run_log(__file__, tag=f"{arguments.run_name}_cross", log_dir=arguments.log_dir, config=vars(arguments),
                  enabled=not arguments.no_log)
    run(arguments)
    out = Path(arguments.output_dir) / arguments.run_name
    seed = "" if arguments.seed == 42 else f"_seed{arguments.seed}"
    oracle_path = out / f"oracle_cross{seed}.json"
    upload_results(arguments, job_type="prune_cross_ablation", group=arguments.run_name,
                   files=[out / f"results_cross{seed}.json", oracle_path],
                   summary=json.loads(oracle_path.read_text())["summary"] if oracle_path.exists() else None)
