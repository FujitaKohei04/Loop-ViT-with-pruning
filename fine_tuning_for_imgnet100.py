"""Fine-tune SReT models on ImageNet-100 (./data/ImageNet) with a 100-class head.

The 1000-class head is replaced by a 100-class head initialized with the rows of the
corresponding ImageNet-1k classes, then one of the following is trained:
    head: only the classification head (the backbone is frozen)
    full: all parameters

With --epochs 0 the model is only converted (head replaced, no training) and saved.

Usage examples:
    python fine_tuning_for_imgnet100.py --models sret_s --mode head
    python fine_tuning_for_imgnet100.py --models sret_s sret_t sret_lt --mode full
    python fine_tuning_for_imgnet100.py --models sret_s --mode head --epochs 0   # conversion only
    python fine_tuning_for_imgnet100.py --models sret_s --mode full --lr 1e-5 --batch-size 32
"""

import argparse
from dataclasses import asdict, dataclass, replace
import math
import os
from pathlib import Path
import random
from typing import Callable

import numpy as np
import torch
from timm.data import create_transform, resolve_model_data_config
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

import SReT
import SReT_no_loop
from imagenet_labels import build_imagenet_target_map
from main import evaluate_topk

Batch = tuple[torch.Tensor, torch.Tensor]


# * ===== Settings (edit here) =====

# Model name -> (constructor, pretrained ImageNet-1k checkpoint).
# To add a model, append an entry here.
MODEL_BUILDERS: dict[str, tuple[Callable[..., nn.Module], str]] = {
    "sret_s":         (SReT.SReT_S, "./pre-trained/SReT_S.pth"),
    "sret_t":         (SReT.SReT_T, "./pre-trained/SReT_T.pth"),
    "sret_lt":        (SReT.SReT_LT, "./pre-trained/SReT_LT_8-4-1_2-1-1.pth"),
    "sret_s_no_loop": (SReT_no_loop.SReT_S_no_loop, "./pre-trained/SReT_S.pth"),
}


@dataclass(frozen=True)
class Config:
    # * data
    image_net_root: str = "./data/ImageNet"  # ImageNet-100 in ImageFolder layout (train/ val/)
    batch_size: int = 64                     # lower this if CUDA runs out of memory
    num_workers: int = 4
    seed: int = 42

    # * training (defaults depend on the mode, see MODE_DEFAULTS)
    mode: str = "full"             # head: train only the head / full: train all parameters
    epochs: int = 5
    lr: float = 5e-5
    weight_decay: float = 0.05
    warmup_steps: int = 500        # linear warmup, then cosine decay
    label_smoothing: float = 0.1

    # * output
    output_dir: str = "./fine-tuned"


# Default hyper parameters of each mode (overridden by command line arguments).
# head: only a linear layer is trained, so a larger LR converges quickly.
# full: small LR so that the pretrained features are not destroyed.
MODE_DEFAULTS: dict[str, dict[str, float]] = {
    "head": {"epochs": 3, "lr": 1e-3, "weight_decay": 1e-4, "warmup_steps": 100},
    "full": {"epochs": 5, "lr": 5e-5, "weight_decay": 0.05, "warmup_steps": 500},
}


def output_path(config: Config, model_name: str) -> Path:
    """e.g. ./fine-tuned/sret_s_imagenet100_full.pth"""
    return Path(config.output_dir) / f"{model_name}_imagenet100_{config.mode}.pth"


# * ===== Setup =====

def set_seed(seed: int) -> None:
    """Seed everything (without forcing deterministic kernels, which SReT's backward does not support)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(model_name: str) -> nn.Module:
    """Build the model and load its ImageNet-1k pretrained weights."""
    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"unknown model: {model_name} (available: {list(MODEL_BUILDERS)})")
    constructor, checkpoint_path = MODEL_BUILDERS[model_name]
    model = constructor(pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False)["model"])
    return model


def build_loaders(config: Config, model: nn.Module) -> tuple[DataLoader[Batch], DataLoader[Batch], list[str]]:
    """ImageNet-100 loaders with labels 0..99 (ImageFolder order). The train loader is shuffled."""
    root = Path(config.image_net_root).expanduser()
    data_config = resolve_model_data_config(model)
    train_dataset = datasets.ImageFolder(root / "train", transform=create_transform(**data_config, is_training=True))
    val_dataset = datasets.ImageFolder(root / "val", transform=create_transform(**data_config, is_training=False))
    if train_dataset.classes != val_dataset.classes:
        raise ValueError("train and val must have the same classes")

    kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": config.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **kwargs)
    return train_loader, val_loader, train_dataset.classes


def replace_head_for_subset(model: nn.Module, classes: list[str]) -> None:
    """Replace the 1000-class head by a 100-class head, initialized with the rows of the
    corresponding ImageNet-1k classes (so the model starts at its 1k accuracy restricted to 100 classes)."""
    target_map = build_imagenet_target_map(classes)
    indices = torch.tensor([target_map[i] for i in range(len(classes))])
    old_head = model.head
    new_head = nn.Linear(old_head.in_features, len(classes))
    with torch.no_grad():
        new_head.weight.copy_(old_head.weight[indices])
        new_head.bias.copy_(old_head.bias[indices])
    model.head = new_head
    model.num_classes = len(classes)


def set_trainable(model: nn.Module, mode: str) -> list[nn.Parameter]:
    """Freeze the parameters that are not trained in this mode and return the trainable ones."""
    if mode == "full":
        for parameter in model.parameters():
            parameter.requires_grad = True
    elif mode == "head":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.head.parameters():
            parameter.requires_grad = True
    else:
        raise ValueError(f"unknown mode: {mode}")
    return [p for p in model.parameters() if p.requires_grad]


def count_parameters(parameters: list[nn.Parameter]) -> int:
    return sum(p.numel() for p in parameters)


# * ===== Training =====

def warmup_cosine(total_steps: int, warmup_steps: int) -> Callable[[int], float]:
    """LR multiplier: linear warmup, then cosine decay to 0."""
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return factor


def set_train_mode(model: nn.Module, mode: str) -> None:
    """full: whole model in train mode.
    head: backbone stays in eval mode (no dropout / drop path), so it behaves as a fixed feature extractor."""
    if mode == "full":
        model.train()
    else:
        model.eval()
        model.head.train()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[Batch],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    loss_fn: nn.Module,
    config: Config,
    device: torch.device,
    desc: str,
) -> dict[str, float]:
    """One pass over the train data with AMP. Returns the mean loss and the train top-1 accuracy."""
    set_train_mode(model, config.mode)
    total_loss, correct, total = 0.0, 0, 0
    bar = tqdm(loader, desc=desc, leave=False)
    for images, targets in bar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(images)
            loss = loss_fn(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * targets.numel()
        correct += int(logits.argmax(dim=1).eq(targets).sum().item())
        total += targets.numel()
        bar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{scheduler.get_last_lr()[0]:.1e}")
    return {"train_loss": total_loss / max(total, 1), "train_top1": correct / max(total, 1)}


def fine_tune(
    model: nn.Module,
    train_loader: DataLoader[Batch],
    val_loader: DataLoader[Batch],
    config: Config,
    device: torch.device,
) -> list[dict[str, float]]:
    """Train for config.epochs and evaluate on val after every epoch."""
    parameters = set_trainable(model, config.mode)
    print(f"trainable parameters: {count_parameters(parameters):,} / {count_parameters(list(model.parameters())):,}")
    if config.epochs == 0:
        return []

    optimizer = torch.optim.AdamW(parameters, lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine(len(train_loader) * config.epochs, config.warmup_steps))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_fn = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    history = []
    for epoch in range(1, config.epochs + 1):
        train_stats = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, loss_fn, config, device,
                                      desc=f"{config.mode} FT {epoch}/{config.epochs}")
        metrics = evaluate_topk(model, val_loader, device)
        history.append({"epoch": epoch, **train_stats, **metrics})
        print(f"epoch {epoch}: loss={train_stats['train_loss']:.4f} train_top1={train_stats['train_top1']:.2%} "
              f"val_top1={metrics['top1']:.2%} val_top5={metrics['top5']:.2%}")
    return history


# * ===== Save / load =====

def save_checkpoint(path: Path, model: nn.Module, model_name: str, classes: list[str], config: Config,
                    before: dict[str, float], history: list[dict[str, float]]) -> None:
    """Weights with a 100-class head + everything needed to reproduce / reuse them."""
    os.makedirs(path.parent, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "model_name": model_name,
        "classes": classes,              # class i of the head = classes[i] (WNID)
        "config": asdict(config),
        "metrics_before_ft": before,     # 1k head restricted to the 100 classes, no training
        "history": history,
    }, path)
    print(f"saved: {path}")


def load_fine_tuned(path: str | Path, device: torch.device | str = "cpu") -> tuple[nn.Module, list[str]]:
    """Rebuild a model saved by this script. Returns (model, classes)."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    constructor, _ = MODEL_BUILDERS[checkpoint["model_name"]]
    model = constructor(pretrained=False)
    model.head = nn.Linear(model.head.in_features, len(checkpoint["classes"]))
    model.num_classes = len(checkpoint["classes"])
    model.load_state_dict(checkpoint["model"])
    return model.to(device), checkpoint["classes"]


# * ===== Main =====

def run(model_name: str, config: Config, device: torch.device) -> None:
    """Convert the head, fine-tune and save one model."""
    print(f"========== {model_name} ({config.mode}) ==========")
    set_seed(config.seed)
    model = build_model(model_name)
    train_loader, val_loader, classes = build_loaders(config, model)
    replace_head_for_subset(model, classes)
    model.to(device)

    before = evaluate_topk(model, val_loader, device)
    print(f"before fine-tuning (1k head restricted to 100 classes): "
          f"top1={before['top1']:.2%} top5={before['top5']:.2%}")

    history = fine_tune(model, train_loader, val_loader, config, device)
    save_checkpoint(output_path(config, model_name), model, model_name, classes, config, before, history)

    del model, train_loader, val_loader
    torch.cuda.empty_cache()


def parse_args() -> tuple[list[str], Config]:
    parser = argparse.ArgumentParser(description="Fine-tune SReT models on ImageNet-100 with a 100-class head.")
    parser.add_argument("--models", nargs="+", choices=list(MODEL_BUILDERS), default=["sret_s"])
    parser.add_argument("--mode", choices=list(MODE_DEFAULTS), default=Config.mode,
                        help="head: train only the classification head / full: train all parameters")
    parser.add_argument("--epochs", type=int, default=None, help="0 = convert the head only (default: MODE_DEFAULTS)")
    parser.add_argument("--lr", type=float, default=None, help="default: MODE_DEFAULTS")
    parser.add_argument("--weight-decay", type=float, default=None, help="default: MODE_DEFAULTS")
    parser.add_argument("--warmup-steps", type=int, default=None, help="default: MODE_DEFAULTS")
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--num-workers", type=int, default=Config.num_workers)
    parser.add_argument("--output-dir", default=Config.output_dir)
    args = parser.parse_args()

    # mode defaults first, then the values given on the command line
    overrides = dict(MODE_DEFAULTS[args.mode])
    for key in ("epochs", "lr", "weight_decay", "warmup_steps"):
        if getattr(args, key) is not None:
            overrides[key] = getattr(args, key)
    config = replace(Config(), mode=args.mode, batch_size=args.batch_size, num_workers=args.num_workers,
                     output_dir=args.output_dir, **overrides)
    return args.models, config


def main() -> None:
    model_names, config = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(asdict(config))
    for model_name in model_names:
        run(model_name, config, device)


if __name__ == "__main__":
    main()
