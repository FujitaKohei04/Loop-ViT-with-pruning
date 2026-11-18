"""Measure parameters, model size, FLOPs and throughput.

Works for RecurrentViT presets (bvit_*, vit_*_scratch), SReT models and any timm
model name (built without pretrained weights).

FLOPs are reported two ways:
    total:       every op seen by torch.utils.flop_counter (incl. QK^T and AV)
    linear-only: Linear + Conv layers only. This is the paper's convention
                 (ViT-B 33.7G, ViT-S 8.5G in Table 1).
FLOPs = 2 x MACs. Shared blocks are counted once per call.

Usage:
    uv run python model_profile.py --models bvit_b vit_b_scratch --dataset imagenet100
    uv run python model_profile.py --models bvit_s bvit_b --dataset cifar100 --loop "0x6"
    uv run python model_profile.py --models sret_s vit_base_patch16_224 --img-size 224 --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import time

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from dl_dataset.registry import get_spec
from models.bViT.bViT import PRESETS, RecurrentViT, build_recurrent_vit
from run_logging import add_log_args, start_run_log


def build_any_model(name: str, *, img_size: int, num_classes: int, patch_size: int | None = None,
                    loop: str | None = None) -> nn.Module:
    if name in PRESETS:
        overrides = {"img_size": img_size, "num_classes": num_classes}
        if patch_size is not None:
            overrides["patch_size"] = patch_size
        if loop is not None:
            overrides["loop"] = loop
        return build_recurrent_vit(name, **overrides)
    if name in ("sret_s", "sret_t", "sret_lt"):
        import SReT
        constructor = {"sret_s": SReT.SReT_S, "sret_t": SReT.SReT_T, "sret_lt": SReT.SReT_LT}[name]
        return constructor(pretrained=False)
    import timm
    return timm.create_model(name, pretrained=False, num_classes=num_classes)


# * ===== Parameters / size =====

def count_params(model: nn.Module) -> dict[str, float]:
    """Unique parameters (shared blocks counted once), per top-level child too."""
    stats = {
        "params": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    for name, child in model.named_children():
        stats[f"params.{name}"] = sum(p.numel() for p in child.parameters())
    if isinstance(model, RecurrentViT):
        # parameters of the equivalent unrolled (non-shared) network, for reference
        block_params = [sum(p.numel() for p in block.parameters()) for block in model.blocks]
        uses = model.schedule.uses_per_block()
        stats["params.unrolled"] = (stats["params"] - sum(block_params)
                                    + sum(n * u for n, u in zip(block_params, uses)))
    return stats


def model_size(model: nn.Module) -> dict[str, float]:
    """Size in MB: fp32 / fp16 tensors and the serialized state_dict."""
    tensors = list(model.state_dict().values())
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return {
        "size_mb.fp32": sum(t.numel() * 4 for t in tensors if t.is_floating_point()) / 2**20,
        "size_mb.fp16": sum(t.numel() * 2 for t in tensors if t.is_floating_point()) / 2**20,
        "size_mb.state_dict": buffer.getbuffer().nbytes / 2**20,
    }


# * ===== FLOPs =====

@torch.no_grad()
def count_flops(model: nn.Module, img_size: int, device: torch.device) -> dict[str, float]:
    model.eval()
    x = torch.randn(1, 3, img_size, img_size, device=device)

    linear_flops = 0

    def hook(module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
        nonlocal linear_flops
        if isinstance(module, nn.Linear):
            # rows from the output, so that zero-width layers of pruned models (in_features == 0) count as 0
            rows = output.numel() // max(module.out_features, 1)
            linear_flops += 2 * rows * module.in_features * module.out_features
        else:  # Conv2d
            kernel = module.in_channels // module.groups * module.kernel_size[0] * module.kernel_size[1]
            linear_flops += 2 * output.numel() * kernel

    handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Linear, nn.Conv2d))]
    try:
        with FlopCounterMode(display=False) as counter:
            model(x)
    finally:
        for handle in handles:
            handle.remove()
    return {"gflops.total": counter.get_total_flops() / 1e9, "gflops.linear_only": linear_flops / 1e9}


# * ===== Throughput =====

def throughput(model: nn.Module, img_size: int, device: torch.device, *, train: bool, batch_size: int,
               iters: int = 20, warmup: int = 5, amp: bool = True) -> dict[str, float]:
    """Images/s with random data. train=True includes backward + AdamW step."""
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    amp = amp and device.type == "cuda"
    model.train(train)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    if train:
        optimizer = torch.optim.AdamW(model.parameters())
        scaler = torch.amp.GradScaler(enabled=amp)

        def step() -> None:
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                out = model(x)
                loss = out.float().logsumexp(dim=-1).mean()
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    else:
        def step() -> None:
            with torch.inference_mode(), torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                model(x)

    for _ in range(warmup):
        step()
    _sync(device)
    start = time.perf_counter()
    for _ in range(iters):
        step()
    _sync(device)
    elapsed = time.perf_counter() - start

    prefix = "train" if train else "infer"
    stats = {f"{prefix}.img_per_s": batch_size * iters / elapsed, f"{prefix}.batch_size": batch_size}
    if device.type == "cuda":
        stats[f"{prefix}.peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
    if train:
        model.zero_grad(set_to_none=True)
        del optimizer
    return stats


@torch.inference_mode()
def latency_ms(model: nn.Module, img_size: int, device: torch.device, *, iters: int = 50,
               amp: bool = True) -> float:
    """Batch-1 latency in milliseconds."""
    model.eval()
    x = torch.randn(1, 3, img_size, img_size, device=device)
    amp = amp and device.type == "cuda"
    with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
        for _ in range(10):
            model(x)
        _sync(device)
        start = time.perf_counter()
        for _ in range(iters):
            model(x)
        _sync(device)
    return (time.perf_counter() - start) * 1000 / iters


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def profile(model: nn.Module, img_size: int, device: torch.device, *, train_batch: int, infer_batch: int,
            skip_throughput: bool = False, amp: bool = True) -> dict[str, float]:
    model.to(device)
    stats: dict[str, float] = {**count_params(model), **model_size(model), **count_flops(model, img_size, device)}
    if not skip_throughput:
        stats.update(throughput(model, img_size, device, train=False, batch_size=infer_batch, amp=amp))
        stats.update(throughput(model, img_size, device, train=True, batch_size=train_batch, amp=amp))
        stats["latency_ms.bs1"] = latency_ms(model, img_size, device, amp=amp)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--dataset", default=None, help="sets img size / patch / classes (e.g. cifar100)")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--loop", default=None, help="override the loop schedule of RecurrentViT presets")
    parser.add_argument("--train-batch", type=int, default=64)
    parser.add_argument("--infer-batch", type=int, default=256)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--skip-throughput", action="store_true")
    parser.add_argument("--csv", default=None)
    add_log_args(parser)
    args = parser.parse_args()
    start_run_log(__file__, tag="-".join(args.models), log_dir=args.log_dir, config=vars(args),
                  enabled=not args.no_log)

    img_size, patch_size, num_classes = args.img_size, args.patch_size, args.num_classes
    if args.dataset is not None:
        spec = get_spec(args.dataset)
        img_size, num_classes = spec.img_size, spec.num_classes
        patch_size = patch_size or spec.patch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    for name in args.models:
        model = build_any_model(name, img_size=img_size, num_classes=num_classes, patch_size=patch_size,
                                loop=args.loop)
        stats = profile(model, img_size, device, train_batch=args.train_batch, infer_batch=args.infer_batch,
                        skip_throughput=args.skip_throughput, amp=not args.no_amp)
        rows.append({"model": name, "img_size": img_size, **stats})
        print(f"== {name} (img {img_size}) ==")
        for key, value in stats.items():
            print(f"  {key:24s} {value:,.2f}" if isinstance(value, float) else f"  {key:24s} {value:,}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.csv:
        keys = list(dict.fromkeys(key for row in rows for key in row))
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved: {args.csv}")


if __name__ == "__main__":
    main()
