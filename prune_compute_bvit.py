"""Compute cost (FLOPs, throughput) of pruned RecurrentViT models, without retraining.

The bViT paper (Sec. 4.2.3) uses pruning only as an analysis tool and reports
accuracy. Here every pruned model is turned into a really smaller model
(pruning/compact.py) and its accuracy, FLOPs and throughput are measured.

Families (--families):
    spectral      paper: spectral_step / _global / _permuted / _sigma on Q, V, FFN_IN and on all six projections
    unstructured  taylor (global ranking) / taylor_step / magnitude on all six projections
    neuron        FFN hidden neurons: neuron_taylor / _step / _magnitude      (Molchanov et al. 2019)
    head          attention heads: head_taylor / _step                         (Michel et al. 2019)
    width         heads + neurons at the same ratio: width_taylor / _step
    steps         fewer loop steps (layer dropping / early exit)
    tome          token merging, r tokens per step                             (Bolya et al. 2023)
    evit          CLS-attention token pruning at 1/4, 1/2, 3/4 depth           (Liang et al. 2022)

Pruning rate ("rate"): fraction of the family's units that is removed
    spectral / unstructured / neuron / head / width: removed directions / weights / neurons / heads
    steps: removed loop steps;  tome / evit: mean token reduction over all attention and FFN calls

FLOPs come from model_profile.count_flops (total and Linear-only); "linear_nnz"
counts only non-zero weights (what a sparse kernel could reach for unstructured
pruning). Throughput is measured on the GPU with random images, interleaved
with the unpruned model, so "rel_throughput" is robust to other jobs on the GPU.

Usage:
    uv run python prune_compute_bvit.py --checkpoint outputs/bvit/<run>/best.pth
    uv run python prune_compute_bvit.py --checkpoint ... --families steps tome --run-name <existing run>  # adds to it

Outputs (./outputs/bvit_pruning_compute/<run>/): results.json / results.csv. Figures: prune_compute_report.py
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from contextlib import ExitStack
from itertools import islice
from pathlib import Path

import torch
from torch import nn

from dl_dataset.registry import build_loaders, get_spec
from model_profile import count_flops, latency_ms, throughput
from pruning import (METHODS, Pruner, ScoringContext, StepLimited, TokenReducedViT, compact_model,
                     max_output_difference, stored_params)
from run_logging import add_log_args, link_run_log, run_timestamp, start_run_log
from train_bvit import Config, evaluate, load_model, set_seed

ALL = ["Q", "K", "V", "PROJ", "FFN_IN", "FFN_OUT"]
SPECTRAL = ["spectral_step", "spectral_global", "spectral_permuted", "spectral_sigma"]

# family -> [(group label, method label, [(targets, METHODS name), ...])]; the parts are applied together
PLANS: dict[str, list[tuple[str, str, list[tuple[list[str], str]]]]] = {
    "spectral": [(group, method, [(targets, method)])
                 for group, targets in (("ALL", ALL), ("Q", ["Q"]), ("V", ["V"]), ("FFN_IN", ["FFN_IN"]))
                 for method in SPECTRAL],
    "unstructured": [("ALL", method, [(ALL, method)]) for method in ("taylor", "taylor_step", "magnitude")],
    "neuron": [("FFN_IN", method, [(["FFN_IN"], method)])
               for method in ("neuron_taylor", "neuron_taylor_step", "neuron_magnitude")],
    "head": [("PROJ", method, [(["PROJ"], method)]) for method in ("head_taylor", "head_taylor_step")],
    "width": [("PROJ+FFN_IN", "width_taylor", [(["PROJ"], "head_taylor"), (["FFN_IN"], "neuron_taylor")]),
              ("PROJ+FFN_IN", "width_taylor_step",
               [(["PROJ"], "head_taylor_step"), (["FFN_IN"], "neuron_taylor_step")])],
}
# methods whose compact model has the same layer shapes as another method: reuse its compute measurement
SAME_COMPUTE = {"spectral_permuted": "spectral_step", "spectral_sigma": "spectral_global",
                "neuron_magnitude": "neuron_taylor"}
FAMILIES = [*PLANS, "steps", "tome", "evit"]


# * ===== Measurement =====

@torch.no_grad()
def linear_flops_nnz(model: nn.Module, img_size: int, device: torch.device) -> float:
    """Linear-only GFLOPs counting non-zero weights only (Conv layers are counted in full)."""
    flops = 0

    def hook(module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
        nonlocal flops
        if isinstance(module, nn.Linear):
            flops += 2 * (output.numel() // max(module.out_features, 1)) * int(module.weight.count_nonzero())
        else:
            flops += 2 * output.numel() * (module.in_channels // module.groups
                                           * module.kernel_size[0] * module.kernel_size[1])

    handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d))]
    try:
        model.eval()
        model(torch.randn(1, 3, img_size, img_size, device=device))
    finally:
        for handle in handles:
            handle.remove()
    return flops / 1e9


class Meter:
    """Accuracy and compute of a candidate model, relative to the unpruned model."""

    def __init__(self, base: nn.Module, val_batches: list, img_size: int, device: torch.device,
                 args: argparse.Namespace) -> None:
        self.base = base
        self.val_batches = val_batches
        self.img_size = img_size
        self.device = device
        self.args = args
        self.compute_cache: dict[tuple, dict] = {}

    def accuracy(self, model: nn.Module) -> dict:
        metrics = evaluate(model, self.val_batches, self.device)
        return {key: metrics[key] for key in ("top1", "top5", "loss")}

    def compute(self, model: nn.Module, key: tuple | None = None) -> dict:
        """FLOPs, parameters, throughput and batch-1 latency. Throughput / latency are measured in
        rounds that alternate with the unpruned model; the ratio of each round is kept (median)."""
        if key is not None and key in self.compute_cache:
            return self.compute_cache[key]
        args = self.args
        stats = dict(count_flops(model, self.img_size, self.device))
        stats["gflops.linear_nnz"] = linear_flops_nnz(model, self.img_size, self.device)
        stats["params"] = sum(p.numel() for p in model.parameters())
        stats["stored_params"] = stored_params(model)
        speed, base_speed, lat, base_lat = [], [], [], []
        for _ in range(args.rounds):
            for target, speeds, latencies in ((self.base, base_speed, base_lat), (model, speed, lat)):
                speeds.append(throughput(target, self.img_size, self.device, train=False,
                                         batch_size=args.infer_batch, iters=args.iters, warmup=3)["infer.img_per_s"])
                latencies.append(latency_ms(target, self.img_size, self.device, iters=args.latency_iters))
        stats.update({
            "img_per_s": max(speed), "base_img_per_s": max(base_speed),
            "rel_throughput": statistics.median(s / b for s, b in zip(speed, base_speed)),
            "latency_ms": min(lat), "base_latency_ms": min(base_lat),
            "rel_latency": statistics.median(s / b for s, b in zip(lat, base_lat)),
        })
        if key is not None:
            self.compute_cache[key] = stats
        return stats


# * ===== Families =====

def run_pruner_family(family: str, model: nn.Module, meter: Meter, calib_batches: list, args: argparse.Namespace,
                      device: torch.device) -> list[dict]:
    records: list[dict] = []
    score_cache: dict[tuple[str, str, str], dict] = {}  # (unit, scorer, layer name) -> scores of that layer
    loss_fn = nn.CrossEntropyLoss()
    probe = meter.val_batches[0][0][:64]
    for group, label, parts in PLANS[family]:
        with ExitStack() as stack:
            pruners = [stack.enter_context(Pruner.from_method(model, targets, name, seed=args.seed))
                       for targets, name in parts]
            all_scores = []
            for pruner in pruners:
                method = pruner.method
                keys = [(method.unit, method.scorer, target) for target in pruner.targets]
                if not all(key in score_cache for key in keys):
                    scores = pruner.compute_scores(
                        ScoringContext(model, calib_batches, loss_fn, device, seed=args.seed))
                    score_cache.update({(method.unit, method.scorer, name): value for name, value in scores.items()})
                all_scores.append({target: score_cache[key] for target, key in zip(pruner.targets, keys)})
            previous_rate = None
            for ratio in args.ratios:
                kept = total = 0
                for pruner, scores in zip(pruners, all_scores):
                    masks = pruner.make_masks(scores, ratio)
                    pruner.apply(masks)
                    kept += sum(int(mask.sum()) for mask in masks.values())
                    total += sum(mask.numel() for mask in masks.values())
                # one pruner: the real removed fraction; several (width): the requested ratio
                rate = 1 - kept / total if len(pruners) == 1 else ratio
                if previous_rate is not None and abs(rate - previous_rate) < 1e-9:
                    continue  # e.g. 6 heads: several ratios round to the same masks
                previous_rate = rate
                small = compact_model(model)
                difference = max_output_difference(model, small, probe)
                if difference > 1e-3:
                    raise RuntimeError(f"{label} ratio {ratio}: compact model differs from masked model "
                                       f"({difference:.2e})")
                record = {"family": family, "group": group, "method": label, "ratio": ratio, "rate": rate,
                          "max_diff": difference, **meter.accuracy(small),
                          **meter.compute(small, key=(family, group, SAME_COMPUTE.get(label, label), ratio))}
                records.append(record)
                log_record(record)
                del small
    return records


def run_wrapper_family(family: str, model: nn.Module, meter: Meter, args: argparse.Namespace) -> list[dict]:
    """Families that change the forward pass instead of the weights."""
    num_steps = model.schedule.num_steps
    if family == "steps":
        candidates = [(n, StepLimited(model, n)) for n in range(num_steps, 0, -1)]
    elif family == "tome":
        candidates = [(r, TokenReducedViT(model, tome_r=r)) for r in args.tome_r]
    else:
        candidates = [(k, TokenReducedViT(model, evit_keep_rate=k)) for k in args.evit_keep]
    records = []
    for param, wrapped in candidates:
        accuracy = meter.accuracy(wrapped)
        rate = 1 - param / num_steps if family == "steps" else wrapped.token_reduction()
        record = {"family": family, "group": "-", "method": family, "ratio": param, "rate": rate,
                  "max_diff": 0.0, **accuracy, **meter.compute(wrapped)}
        if family != "steps":
            record["tokens_per_step"] = [n for _, n in wrapped.token_counts]
        records.append(record)
        log_record(record)
    return records


def log_record(record: dict) -> None:
    print(f"[{record['family']}/{record['group']}] {record['method']:20s} ratio={record['ratio']:<5} "
          f"rate={record['rate']:.3f} top1={record['top1']:.4f} GFLOPs={record['gflops.total']:.3f} "
          f"(nnz {record['gflops.linear_nnz']:.3f}) rel_thr={record['rel_throughput']:.2f} "
          f"rel_lat={record['rel_latency']:.2f} params={record['stored_params']:,}", flush=True)


# * ===== Main =====

def save(out_dir: Path, results: dict) -> None:
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    records = results["records"]
    fields = [key for key in dict.fromkeys(key for record in records for key in record) if key != "tokens_per_step"]
    with open(out_dir / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    model, checkpoint = load_model(args.checkpoint, use_ema=not args.no_ema, device=device)
    model.eval()
    train_config = Config(**checkpoint["config"])
    spec = get_spec(checkpoint["dataset"], args.data_root or train_config.data_root)
    calib_loader, val_loader, _ = build_loaders(spec, batch_size=args.batch_size, num_workers=args.num_workers,
                                                train_eval_transform=True)
    calib_batches = list(islice(calib_loader, max(1, args.calib_images // args.batch_size)))
    # the val set is evaluated hundreds of times: keep it on the GPU
    val_batches = [(images.to(device), targets.to(device))
                   for images, targets in islice(val_loader, args.eval_batches)]

    out_dir = Path(args.output_dir) / (args.run_name or f"{Path(args.checkpoint).parent.name}_{run_timestamp()}")
    out_dir.mkdir(parents=True, exist_ok=True)
    link_run_log(out_dir)
    results_path = out_dir / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {"records": []}
    results.update({"checkpoint": str(args.checkpoint), "dataset": spec.name, "img_size": spec.img_size,
                    "loop": str(model.schedule), "model": train_config.model, "args": vars(args),
                    "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"})
    print(f"model={train_config.model} loop={model.schedule} dataset={spec.name} "
          f"val_images={sum(len(t) for _, t in val_batches)} calib_images={len(calib_batches) * args.batch_size}")
    print(f"output: {out_dir}")

    meter = Meter(model, val_batches, spec.img_size, device, args)
    results["unpruned"] = {**meter.accuracy(model), **meter.compute(model)}
    print(f"unpruned: top1={results['unpruned']['top1']:.4f} GFLOPs={results['unpruned']['gflops.total']:.3f} "
          f"img/s={results['unpruned']['img_per_s']:.0f} latency={results['unpruned']['latency_ms']:.2f}ms")

    for family in args.families:
        if family in PLANS:
            records = run_pruner_family(family, model, meter, calib_batches, args, device)
        else:
            records = run_wrapper_family(family, model, meter, args)
        results["records"] = [r for r in results["records"] if r["family"] != family] + records
        save(out_dir, results)
    print(f"saved: {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="best.pth / last.pth from train_bvit.py")
    parser.add_argument("--families", nargs="+", default=FAMILIES, choices=FAMILIES)
    parser.add_argument("--ratios", nargs="+", type=float, default=[round(0.1 * i, 1) for i in range(11)])
    parser.add_argument("--tome-r", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 8, 11, 16, 32],
                        help="ToMe: tokens merged per step")
    parser.add_argument("--evit-keep", nargs="+", type=float, default=[round(1 - 0.1 * i, 1) for i in range(10)],
                        help="EViT: fraction of patch tokens kept at each pruning step")
    parser.add_argument("--calib-images", type=int, default=10240)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batches", type=int, default=None, help="limit val batches (default: all)")
    parser.add_argument("--infer-batch", type=int, default=256, help="batch size of the throughput measurement")
    parser.add_argument("--iters", type=int, default=10, help="throughput iterations per round")
    parser.add_argument("--latency-iters", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=3, help="measurement rounds (alternating with the unpruned model)")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--no-ema", action="store_true", help="use raw weights instead of EMA weights")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="./outputs/bvit_pruning_compute")
    parser.add_argument("--run-name", default=None)
    add_log_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    start_run_log(__file__, tag="+".join(args.families), log_dir=args.log_dir, config=vars(args),
                  enabled=not args.no_log)
    run(args)
