"""Train RecurrentViT (bViT / ViT from scratch) with the bViT paper recipe.

Recipe (paper Sec. 3.2), with [A] = assumption, [E] = adaptation to this environment:
    AdamW, base lr 5e-4 for batch 1024, weight decay 0.05
    300 epochs, linear warmup for 30 epochs from 0.033 x lr, cosine annealing to 0
    cross entropy, label smoothing 0.1, Mixup alpha 0.8, CutMix alpha 1.0
    EMA of the weights (paper: 0.99995 for ~375k steps on ImageNet-1k)
    no stochastic depth, no dropout
    [A] Mixup/CutMix chosen 50/50 per batch, gradient clipping 1.0 (torchvision ViT recipe),
        no weight decay on norms / biases / cls / pos / step embeddings, EMA updated every step
    [E] effective batch 1024 reached with gradient accumulation (no BatchNorm, so this equals
        a real 1024 batch; lr is kept at 5e-4), fp16 AMP, single GPU, no repeated-aug sampler
    [E] ema_decay="auto": keep the paper's EMA horizon / total steps ratio (~5.3%)
    [E] dataset, augmentation profile: see dl_dataset/registry.py

Usage:
    uv run python train_bvit.py --model bvit_s --dataset cifar100
    uv run python train_bvit.py --model vit_s_scratch --dataset cifar100 --epochs 100
    uv run python train_bvit.py --model bvit_b --loop "(0,1)x6" --dataset cifar100
    uv run python train_bvit.py --model bvit_s --dataset cifar100 --smoke   # few batches, checks the pipeline
    uv run python train_bvit.py --resume outputs/bvit/<run>/last.pth
    uv run python train_bvit.py --eval-only outputs/bvit/<run>/best.pth --all-steps
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.utils import ModelEmaV3
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dl_dataset.registry import DatasetSpec, build_loaders, get_spec
from models.bViT.bViT import PRESETS, RecurrentViT, build_recurrent_vit

Batch = tuple[torch.Tensor, torch.Tensor]

# EMA horizon of the paper: 1 / (1 - 0.99995) = 20k steps out of 300 ep x 1251 steps.
PAPER_EMA_HORIZON_RATIO = (1 / (1 - 0.99995)) / (300 * math.ceil(1_281_167 / 1024))


@dataclass(frozen=True)
class Config:
    # * model
    model: str = "bvit_s"            # see models/bViT/bViT.py PRESETS
    loop: str | None = None          # override the preset's loop schedule, e.g. "(0,1)x6"
    patch_size: int | None = None    # default: dataset's patch size
    use_step_embed: bool = False
    num_registers: int = 0

    # * data
    dataset: str = "cifar100"
    data_root: str | None = None
    augment: str = "auto"            # auto / paper / small / none
    re_prob: float = 0.25            # [A] random erasing probability
    num_workers: int = 8

    # * optimization (paper values)
    epochs: int = 300
    warmup_epochs: int = 30
    warmup_start_factor: float = 0.033
    lr: float = 5e-4
    weight_decay: float = 0.05
    effective_batch: int = 1024
    batch_size: int = 128            # per forward pass; accum steps = effective_batch / batch_size
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.8
    cutmix_alpha: float = 1.0
    mixup_switch_prob: float = 0.5   # [A]
    clip_grad: float | None = 1.0    # [A]
    ema_decay: str = "auto"          # "auto", "none" or a float like "0.99995"
    amp: bool = True

    # * run
    seed: int = 42
    eval_interval: int = 1           # epochs
    output_dir: str = "./outputs/bvit"
    run_name: str | None = None
    max_train_batches: int | None = None  # per epoch, for smoke tests
    max_eval_batches: int | None = None

    @property
    def accum_steps(self) -> int:
        if self.effective_batch % self.batch_size != 0:
            raise ValueError("effective_batch must be a multiple of batch_size")
        return self.effective_batch // self.batch_size


# * ===== Setup =====

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(config: Config, spec: DatasetSpec, num_classes: int | None = None) -> RecurrentViT:
    overrides = {
        "img_size": spec.img_size,
        "patch_size": config.patch_size or spec.patch_size,
        "num_classes": num_classes or spec.num_classes,
        "use_step_embed": config.use_step_embed,
        "num_registers": config.num_registers,
    }
    if config.loop is not None:
        overrides["loop"] = config.loop
    return build_recurrent_vit(config.model, **overrides)


def parameter_groups(model: RecurrentViT, weight_decay: float) -> list[dict]:
    """[A] no weight decay on 1-d parameters (norms, biases) and token / position embeddings."""
    skip = model.no_weight_decay()
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (no_decay if parameter.ndim <= 1 or name in skip else decay).append(parameter)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def warmup_cosine(total_steps: int, warmup_steps: int, start_factor: float) -> Callable[[int], float]:
    """LR multiplier: linear start_factor -> 1 during warmup, then cosine to 0."""
    def factor(step: int) -> float:
        if step < warmup_steps:
            return start_factor + (1 - start_factor) * step / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return factor


def resolve_ema_decay(config: Config, total_steps: int) -> float | None:
    if config.ema_decay == "none":
        return None
    if config.ema_decay == "auto":
        return 1 - 1 / max(PAPER_EMA_HORIZON_RATIO * total_steps, 1.0)
    return float(config.ema_decay)


# * ===== Train / eval =====

def autocast(config: Config, device: torch.device):
    return torch.autocast(device.type, dtype=torch.float16, enabled=config.amp and device.type == "cuda")


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader[Batch], device: torch.device, *, amp: bool = True,
             max_batches: int | None = None, all_steps: bool = False) -> dict:
    """top1 / top5. With all_steps=True also per-step top1 (paper Fig. 2a)."""
    model.eval()
    correct1 = correct5 = total = 0
    step_correct: list[int] = []
    for i, (images, targets) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            if all_steps:
                per_step = model(images, return_all_steps=True)
                logits = per_step[-1]
                predictions = [step_logits.argmax(dim=1) for step_logits in per_step]
                if not step_correct:
                    step_correct = [0] * len(predictions)
                for t, prediction in enumerate(predictions):
                    step_correct[t] += int(prediction.eq(targets).sum())
            else:
                logits = model(images)
        top5 = logits.topk(min(5, logits.shape[-1]), dim=1).indices.eq(targets[:, None])
        correct1 += int(top5[:, 0].sum())
        correct5 += int(top5.sum())
        total += targets.numel()
    metrics: dict = {"top1": correct1 / max(total, 1), "top5": correct5 / max(total, 1)}
    if all_steps:
        metrics["top1_per_step"] = [c / max(total, 1) for c in step_correct]
    return metrics


def train_one_epoch(model, loader, optimizer, scheduler, scaler, mixup, loss_fn, ema, config, device,
                    desc) -> dict[str, float]:
    model.train()
    accum = config.accum_steps
    num_batches = len(loader) if config.max_train_batches is None else min(len(loader), config.max_train_batches)
    num_batches -= num_batches % accum  # only full optimizer steps
    if num_batches == 0:
        raise ValueError("not enough batches for one optimizer step; lower effective_batch")

    total_loss, seen = 0.0, 0
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    bar = tqdm(loader, total=num_batches, desc=desc, leave=False)
    for i, (images, targets) in enumerate(bar):
        if i >= num_batches:
            break
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        images, soft_targets = mixup(images, targets)
        with autocast(config, device):
            loss = loss_fn(model(images), soft_targets)
        scaler.scale(loss / accum).backward()

        if (i + 1) % accum == 0:
            if config.clip_grad is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            if ema is not None:
                ema.update(model)

        total_loss += loss.item() * targets.numel()
        seen += targets.numel()
        bar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{scheduler.get_last_lr()[0]:.1e}")
    elapsed = time.perf_counter() - start
    return {"train_loss": total_loss / max(seen, 1), "epoch_time_s": elapsed, "train_img_per_s": seen / elapsed}


# * ===== Checkpoints =====

def save_checkpoint(path: Path, *, model, ema, optimizer, scheduler, scaler, epoch, config, spec, classes,
                    history, best_top1) -> None:
    torch.save({
        "model": model.state_dict(),
        "ema": ema.module.state_dict() if ema is not None else None,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "config": asdict(config),
        "model_config": asdict(model.config),
        "dataset": spec.name,
        "classes": classes,
        "history": history,
        "best_top1": best_top1,
    }, path)


def load_model(path: str | Path, *, use_ema: bool = True, device: torch.device | str = "cpu"
               ) -> tuple[RecurrentViT, dict]:
    """Rebuild a model saved by this script. Returns (model, checkpoint dict)."""
    from models.bViT.bViT import RecurrentViTConfig
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = RecurrentViT(RecurrentViTConfig(**checkpoint["model_config"]))
    state = checkpoint["ema"] if use_ema and checkpoint.get("ema") is not None else checkpoint["model"]
    model.load_state_dict(state)
    return model.to(device), checkpoint


# * ===== Main =====

def train(config: Config, resume: str | None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(config.seed)
    spec = get_spec(config.dataset, config.data_root)
    train_loader, val_loader, classes = build_loaders(
        spec, batch_size=config.batch_size, num_workers=config.num_workers, augment=config.augment,
        re_prob=config.re_prob, eval_batch_size=2 * config.batch_size)
    if len(classes) != spec.num_classes:
        print(f"warning: {spec.name} is registered with {spec.num_classes} classes, found {len(classes)}; "
              f"using {len(classes)}")
    model = build_model(config, spec, len(classes)).to(device)

    steps_per_epoch = (len(train_loader) if config.max_train_batches is None
                       else min(len(train_loader), config.max_train_batches)) // config.accum_steps
    total_steps = steps_per_epoch * config.epochs
    optimizer = torch.optim.AdamW(parameter_groups(model, config.weight_decay), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine(total_steps, steps_per_epoch * config.warmup_epochs, config.warmup_start_factor))
    scaler = torch.amp.GradScaler(enabled=config.amp and device.type == "cuda")
    ema_decay = resolve_ema_decay(config, total_steps)
    ema = ModelEmaV3(model, decay=ema_decay) if ema_decay is not None else None
    mixup = Mixup(mixup_alpha=config.mixup_alpha, cutmix_alpha=config.cutmix_alpha,
                  switch_prob=config.mixup_switch_prob, label_smoothing=config.label_smoothing,
                  num_classes=len(classes))
    loss_fn = SoftTargetCrossEntropy()

    start_epoch, history, best_top1 = 1, [], 0.0
    if resume is not None:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        if ema is not None and checkpoint["ema"] is not None:
            ema.module.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch, history, best_top1 = checkpoint["epoch"] + 1, checkpoint["history"], checkpoint["best_top1"]
        out_dir = Path(resume).parent
    else:
        run_name = config.run_name or f"{config.model}_{config.dataset}_{time.strftime('%Y%m%d-%H%M%S')}"
        out_dir = Path(config.output_dir) / run_name
        os.makedirs(out_dir, exist_ok=True)

    params = sum(p.numel() for p in model.parameters())
    tokens = (model.config.img_size // model.config.patch_size) ** 2 + model.num_prefix_tokens
    print(f"model={config.model} loop={model.schedule} params={params:,} dataset={spec.name} "
          f"tokens={tokens} steps/epoch={steps_per_epoch} "
          f"total_steps={total_steps} accum={config.accum_steps} ema_decay={ema_decay}")
    print(f"output: {out_dir}")
    (out_dir / "config.json").write_text(json.dumps(
        {**asdict(config), "resolved_ema_decay": ema_decay, "loop_resolved": str(model.schedule),
         "params": params}, indent=2))

    for epoch in range(start_epoch, config.epochs + 1):
        stats = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, mixup, loss_fn, ema,
                                config, device, desc=f"epoch {epoch}/{config.epochs}")
        record = {"epoch": epoch, "lr": scheduler.get_last_lr()[0], **stats}
        if epoch % config.eval_interval == 0 or epoch == config.epochs:
            metrics = evaluate(model, val_loader, device, amp=config.amp, max_batches=config.max_eval_batches)
            record.update({f"val_{k}": v for k, v in metrics.items()})
            if ema is not None:
                ema_metrics = evaluate(ema.module, val_loader, device, amp=config.amp,
                                       max_batches=config.max_eval_batches)
                record.update({f"ema_val_{k}": v for k, v in ema_metrics.items()})
            score = record.get("ema_val_top1", record["val_top1"])
            if score >= best_top1:
                best_top1 = score
                save_checkpoint(out_dir / "best.pth", model=model, ema=ema, optimizer=optimizer,
                                scheduler=scheduler, scaler=scaler, epoch=epoch, config=config, spec=spec,
                                classes=classes, history=history + [record], best_top1=best_top1)
        history.append(record)
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()),
              flush=True)
        save_checkpoint(out_dir / "last.pth", model=model, ema=ema, optimizer=optimizer, scheduler=scheduler,
                        scaler=scaler, epoch=epoch, config=config, spec=spec, classes=classes,
                        history=history, best_top1=best_top1)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    # per-step accuracy of the final (EMA) model, paper Fig. 2a
    final = ema.module if ema is not None else model
    metrics = evaluate(final, val_loader, device, amp=config.amp, max_batches=config.max_eval_batches,
                       all_steps=True)
    (out_dir / "final_eval.json").write_text(json.dumps(metrics, indent=2))
    print(f"final: top1={metrics['top1']:.4f} per-step top1="
          f"{[round(v, 4) for v in metrics['top1_per_step']]}")


def eval_only(path: str, all_steps: bool, use_ema: bool, num_workers: int, batch_size: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(path, use_ema=use_ema, device=device)
    config = Config(**checkpoint["config"])
    _, val_loader, _ = build_loaders(get_spec(checkpoint["dataset"], config.data_root), batch_size=batch_size,
                                     num_workers=num_workers)
    metrics = evaluate(model, val_loader, device, all_steps=all_steps)
    print(json.dumps(metrics, indent=2))


def parse_args() -> tuple[Config, argparse.Namespace]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for field in fields(Config):
        flag = "--" + field.name.replace("_", "-")
        if field.type == "bool":
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=None)
        else:
            caster = {"int": int, "float": float}.get(field.type.split(" |")[0], str)
            parser.add_argument(flag, type=caster, default=None)
    parser.add_argument("--resume", default=None, help="last.pth to continue training from")
    parser.add_argument("--smoke", action="store_true", help="2 epochs x 8 batches, eval 4 batches")
    parser.add_argument("--eval-only", default=None, metavar="CKPT")
    parser.add_argument("--all-steps", action="store_true", help="eval-only: per-step accuracy")
    parser.add_argument("--no-ema", action="store_true", help="eval-only: use raw weights instead of EMA")
    args = parser.parse_args()

    if args.resume is not None:
        base = Config(**torch.load(args.resume, map_location="cpu", weights_only=False)["config"])
    else:
        base = Config()
    overrides = {f.name: getattr(args, f.name) for f in fields(Config) if getattr(args, f.name) is not None}
    config = replace(base, **overrides)
    if config.model not in PRESETS:
        raise ValueError(f"unknown model {config.model} (available: {list(PRESETS)})")
    if args.smoke:
        config = replace(config, epochs=2, warmup_epochs=1, max_train_batches=8, max_eval_batches=4,
                         effective_batch=2 * config.batch_size, num_workers=min(config.num_workers, 4),
                         output_dir=args.output_dir or "./outputs/bvit_smoke")
    return config, args


def main() -> None:
    config, args = parse_args()
    if args.eval_only is not None:
        eval_only(args.eval_only, args.all_steps, not args.no_ema, config.num_workers, 2 * config.batch_size)
        return
    print(asdict(config))
    train(config, args.resume)


if __name__ == "__main__":
    main()
