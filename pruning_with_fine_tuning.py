"""Iterative Taylor pruning with fine-tuning (Molchanov et al., ICLR 2017, arXiv:1611.06440)
applied to SReT-S on ImageNet-100.

Procedure (Sec. 2 of the paper):
    1) Fine-tune the network on the target task (ImageNet-100, 100-class head).
    2) Alternate pruning and fine-tuning:
         compute Taylor scores -> prune the least important weights -> fine-tune a few updates
    3) Stop at the target pruning rate.

The pruning unit is a single weight (unstructured), scored by |w * dC/dw| as in
compare_acc_by_each_pruning.py, so the results can be compared with the one-shot pruning.

Usage examples:
    python pruning_with_fine_tuning.py                      # iterative pruning + fine-tuning
    python pruning_with_fine_tuning.py --mode oneshot       # baseline: one-shot pruning, no fine-tuning
    python pruning_with_fine_tuning.py --updates-per-step 1000
    python pruning_with_fine_tuning.py --force-initial-ft   # redo step 1 even if the checkpoint exists
"""

import argparse
from dataclasses import asdict, dataclass, replace
import math
import os
from pathlib import Path
import pickle
import random
from typing import Iterator

import numpy as np
import torch
from timm.data import create_transform, resolve_model_data_config
from torch import nn
from torch.nn.utils import prune
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

import SReT
from imagenet_labels import build_imagenet_target_map
from main import evaluate_topk

Batch = tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class Config:
    # * data
    image_net_root: str = "./data/ImageNet"  # ImageNet-100 in ImageFolder layout (train/ val/)
    batch_size: int = 64                     # lower this if CUDA runs out of memory
    num_workers: int = 4
    seed: int = 42

    # * step 1: initial fine-tuning on ImageNet-100
    pretrained_path: str = "./pre-trained/SReT_S.pth"
    finetuned_path: str = "./fine-tuned/sret_s_imagenet100.pth"  # reused if it exists
    initial_epochs: int = 5
    initial_lr: float = 5e-5
    initial_weight_decay: float = 0.05
    warmup_steps: int = 500
    label_smoothing: float = 0.1

    # * step 2: iterative pruning
    mode: str = "iterative"         # iterative: prune -> fine-tune repeatedly / oneshot: no fine-tuning
    prune_step: float = 0.05        # pruning rate added per iteration
    max_pruning_rate: float = 0.95
    calibration_batches: int = 16   # batches used to compute the Taylor scores
    layer_l2_normalize: bool = False  # paper Sec. 2.3; False = same criterion as the one-shot results
    updates_per_step: int = 300     # fine-tuning mini-batch updates after each pruning step
    step_lr: float = 1e-5
    step_weight_decay: float = 1e-4

    # * output (same format as compare_acc_by_each_pruning.py -> usable by show_prun_acc_graph.py)
    output_root: str = "./prun_acc_figure"
    model_name: str = "sret_s"

    @property
    def output_dir(self) -> Path:
        return Path(self.output_root) / f"{self.model_name}_in100_{self.mode}"


# * ===== Setup =====

def set_seed(seed: int) -> None:
    """Seed everything (without forcing deterministic kernels, which SReT's backward does not support)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def infinite_batches(loader: DataLoader[Batch]) -> Iterator[Batch]:
    """Iterate over the loader forever (a new shuffle every epoch)."""
    while True:
        yield from loader


def build_model(config: Config) -> nn.Module:
    model = SReT.SReT_S(pretrained=False)
    model.load_state_dict(torch.load(config.pretrained_path, map_location="cpu", weights_only=False)["model"])
    return model


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


# * ===== Training =====

def train_updates(
    model: nn.Module,
    batches: Iterator[Batch],
    num_updates: int,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    desc: str = "train",
) -> float:
    """Run `num_updates` mini-batch updates with AMP. Pruning masks stay active (pruned weights remain 0).
    Returns the mean loss."""
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    total_loss = 0.0
    bar = tqdm(range(num_updates), desc=desc, leave=False)
    for _ in bar:
        images, targets = next(batches)
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            loss = loss_fn(model(images), targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        bar.set_postfix(loss=f"{loss.item():.3f}")
    return total_loss / max(num_updates, 1)


def warmup_cosine(total_steps: int, warmup_steps: int):
    """LR multiplier: linear warmup, then cosine decay to 0."""
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return factor


def initial_fine_tune(
    model: nn.Module,
    train_loader: DataLoader[Batch],
    val_loader: DataLoader[Batch],
    config: Config,
    device: torch.device,
) -> list[dict[str, float]]:
    """Step 1 of the paper: fine-tune the whole network on ImageNet-100."""
    steps_per_epoch = len(train_loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.initial_lr, weight_decay=config.initial_weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine(steps_per_epoch * config.initial_epochs, config.warmup_steps))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    batches = infinite_batches(train_loader)

    history = []
    for epoch in range(1, config.initial_epochs + 1):
        loss = train_updates(model, batches, steps_per_epoch, optimizer, loss_fn, device, scheduler,
                             desc=f"initial FT {epoch}/{config.initial_epochs}")
        metrics = evaluate_topk(model, val_loader, device)
        history.append({"epoch": epoch, "loss": loss, **metrics})
        print(f"[initial FT] epoch {epoch}: loss={loss:.4f} top1={metrics['top1']:.2%} top5={metrics['top5']:.2%}")
    return history


def load_or_fine_tune(
    model: nn.Module,
    train_loader: DataLoader[Batch],
    val_loader: DataLoader[Batch],
    classes: list[str],
    config: Config,
    device: torch.device,
    force: bool,
) -> None:
    """Load the fine-tuned checkpoint if it exists; otherwise run step 1 and save it."""
    path = Path(config.finetuned_path)
    if path.exists() and not force:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint["classes"] != classes:
            raise ValueError(f"classes in {path} differ from the dataset; rerun with --force-initial-ft")
        model.load_state_dict(checkpoint["model"])
        print(f"loaded fine-tuned model: {path}")
        return

    history = initial_fine_tune(model, train_loader, val_loader, config, device)
    os.makedirs(path.parent, exist_ok=True)
    torch.save({"model": model.state_dict(), "classes": classes, "config": asdict(config), "history": history}, path)
    print(f"saved fine-tuned model: {path}")


# * ===== Taylor pruning =====

def iter_prunable_linear_layers(model: nn.Module) -> Iterator[tuple[str, nn.Linear]]:
    """All Linear layers (same target as compare_acc_by_each_pruning.py, head included)."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            yield name, module


def weight_grad(module: nn.Linear) -> torch.Tensor | None:
    """Gradient w.r.t. the weight. After torch pruning, the leaf parameter is `weight_orig`
    and its gradient is already multiplied by the mask."""
    parameter = module.weight_orig if prune.is_pruned(module) else module.weight
    return parameter.grad


def collect_taylor_scores(
    model: nn.Module,
    batches: Iterator[Batch],
    num_batches: int,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Accumulate first-order Taylor scores |w * dC/dw| of the current (masked) weights."""
    layers = dict(iter_prunable_linear_layers(model))
    scores = {name: torch.zeros_like(module.weight, dtype=torch.float32) for name, module in layers.items()}
    model.train()
    for _ in tqdm(range(num_batches), desc="Taylor scores", leave=False):
        images, targets = next(batches)
        model.zero_grad(set_to_none=True)
        loss = loss_fn(model(images.to(device)), targets.to(device))
        loss.backward()
        for name, module in layers.items():
            grad = weight_grad(module)
            if grad is not None:
                scores[name] += (module.weight.detach() * grad.detach()).abs().float()
    model.zero_grad(set_to_none=True)
    return scores


def normalize_scores(scores: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Layer-wise L2 normalization (paper Sec. 2.3)."""
    return {name: s / s.norm().clamp_min(1e-12) for name, s in scores.items()}


def current_masks(model: nn.Module) -> dict[str, torch.Tensor]:
    """Current mask of every prunable layer (all ones if not pruned yet)."""
    return {
        name: module.weight_mask.detach() if prune.is_pruned(module) else torch.ones_like(module.weight)
        for name, module in iter_prunable_linear_layers(model)
    }


def global_masks(scores: dict[str, torch.Tensor], old_masks: dict[str, torch.Tensor],
                 pruning_rate: float) -> dict[str, torch.Tensor]:
    """Prune the lowest scores globally until `pruning_rate` of all prunable weights are 0.
    Already pruned weights always stay pruned (their score is set below every other score)."""
    names = list(scores)
    flat = torch.cat([torch.where(old_masks[n].bool(), scores[n], -1.0).reshape(-1).cpu() for n in names])
    num_to_prune = int(flat.numel() * pruning_rate)
    flat_mask = torch.ones_like(flat)
    if num_to_prune > 0:
        flat_mask[torch.topk(flat, k=num_to_prune, largest=False).indices] = 0.0

    masks, offset = {}, 0
    for name in names:
        numel = scores[name].numel()
        masks[name] = flat_mask[offset:offset + numel].reshape_as(scores[name]).to(scores[name].device)
        offset += numel
    return masks


def apply_masks(model: nn.Module, masks: dict[str, torch.Tensor]) -> None:
    """Set the pruning masks. The first call registers torch pruning; later calls overwrite the mask."""
    for name, module in iter_prunable_linear_layers(model):
        mask = masks[name].to(module.weight.dtype)
        if prune.is_pruned(module):
            module.weight_mask.copy_(mask)
        else:
            prune.custom_from_mask(module, "weight", mask=mask)


def count_parameters(masks: dict[str, torch.Tensor]) -> tuple[int, int]:
    """(remaining, total) prunable parameters."""
    total = sum(m.numel() for m in masks.values())
    remaining = int(sum(m.sum().item() for m in masks.values()))
    return remaining, total


def prune_step(model: nn.Module, batches: Iterator[Batch], pruning_rate: float, config: Config,
               device: torch.device) -> tuple[int, int]:
    """Compute Taylor scores on the current model and prune up to `pruning_rate`."""
    scores = collect_taylor_scores(model, batches, config.calibration_batches, nn.CrossEntropyLoss(), device)
    if config.layer_l2_normalize:
        scores = normalize_scores(scores)
    masks = global_masks(scores, current_masks(model), pruning_rate)
    apply_masks(model, masks)
    return count_parameters(masks)


# * ===== Experiment =====

def pruning_schedule(config: Config) -> list[float]:
    """Pruning rates of each iteration: prune_step, 2 * prune_step, ..., max_pruning_rate."""
    n = int(round(config.max_pruning_rate / config.prune_step))
    return [round(config.prune_step * i, 6) for i in range(1, n + 1)]


def new_record() -> dict[str, dict[str, list[float]]]:
    """Same keys as compare_acc_by_each_pruning.py (+ accuracy right after pruning, before fine-tuning)."""
    return {
        "x": {"pruning_rate": [], "parameter": []},
        "y": {"imagenet_top1": [], "imagenet_top5": [],
              "imagenet_top1_before_ft": [], "imagenet_top5_before_ft": []},
    }


def add_record(record: dict, rate: float, parameter: int, metrics: dict[str, float],
               metrics_before_ft: dict[str, float]) -> None:
    record["x"]["pruning_rate"].append(rate)
    record["x"]["parameter"].append(float(parameter))
    record["y"]["imagenet_top1"].append(metrics["top1"])
    record["y"]["imagenet_top5"].append(metrics["top5"])
    record["y"]["imagenet_top1_before_ft"].append(metrics_before_ft["top1"])
    record["y"]["imagenet_top5_before_ft"].append(metrics_before_ft["top5"])


def save_record(record: dict, config: Config) -> None:
    """Saved after every iteration so that partial results survive a crash."""
    os.makedirs(config.output_dir, exist_ok=True)
    with open(config.output_dir / "data.pkl", "wb") as f:
        pickle.dump(record, f)
    with open(config.output_dir / "hyper_parameter.txt", "w") as f:
        print(asdict(config), file=f)


def run_iterative(model: nn.Module, train_loader: DataLoader[Batch], val_loader: DataLoader[Batch],
                  config: Config, device: torch.device, record: dict) -> None:
    """Step 2: prune -> fine-tune -> evaluate, repeated with increasing pruning rate."""
    batches = infinite_batches(train_loader)
    loss_fn = nn.CrossEntropyLoss()
    for rate in tqdm(pruning_schedule(config), desc="pruning"):
        remaining, total = prune_step(model, batches, rate, config, device)
        metrics_before = evaluate_topk(model, val_loader, device)

        # build the optimizer after pruning: torch pruning renames `weight` to `weight_orig`
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.step_lr, weight_decay=config.step_weight_decay)
        train_updates(model, batches, config.updates_per_step, optimizer, loss_fn, device, desc=f"FT @{rate:.2f}")
        metrics = evaluate_topk(model, val_loader, device)

        add_record(record, rate, remaining, metrics, metrics_before)
        save_record(record, config)
        print(f"rate={rate:.2f} params={remaining}/{total} "
              f"top1: {metrics_before['top1']:.2%} -> {metrics['top1']:.2%} (after FT)")


def run_oneshot(model: nn.Module, train_loader: DataLoader[Batch], val_loader: DataLoader[Batch],
                config: Config, device: torch.device, record: dict) -> None:
    """Baseline: scores of the dense fine-tuned model, each rate pruned at once, no fine-tuning."""
    scores = collect_taylor_scores(model, infinite_batches(train_loader), config.calibration_batches,
                                   nn.CrossEntropyLoss(), device)
    if config.layer_l2_normalize:
        scores = normalize_scores(scores)
    dense_masks = {name: torch.ones_like(s) for name, s in scores.items()}
    for rate in tqdm(pruning_schedule(config), desc="pruning"):
        masks = global_masks(scores, dense_masks, rate)
        apply_masks(model, masks)
        remaining, total = count_parameters(masks)
        metrics = evaluate_topk(model, val_loader, device)
        add_record(record, rate, remaining, metrics, metrics)
        save_record(record, config)
        print(f"rate={rate:.2f} params={remaining}/{total} top1={metrics['top1']:.2%}")


def parse_args() -> tuple[Config, bool]:
    default = Config()
    parser = argparse.ArgumentParser(description="Iterative Taylor pruning + fine-tuning of SReT-S on ImageNet-100.")
    parser.add_argument("--mode", choices=["iterative", "oneshot"], default=default.mode)
    parser.add_argument("--initial-epochs", type=int, default=default.initial_epochs)
    parser.add_argument("--updates-per-step", type=int, default=default.updates_per_step)
    parser.add_argument("--prune-step", type=float, default=default.prune_step)
    parser.add_argument("--max-pruning-rate", type=float, default=default.max_pruning_rate)
    parser.add_argument("--batch-size", type=int, default=default.batch_size)
    parser.add_argument("--layer-l2-normalize", action="store_true", help="layer-wise L2 normalization (paper)")
    parser.add_argument("--force-initial-ft", action="store_true", help="redo step 1 even if the checkpoint exists")
    args = parser.parse_args()
    config = replace(
        default,
        mode=args.mode,
        initial_epochs=args.initial_epochs,
        updates_per_step=args.updates_per_step,
        prune_step=args.prune_step,
        max_pruning_rate=args.max_pruning_rate,
        batch_size=args.batch_size,
        layer_l2_normalize=args.layer_l2_normalize,
    )
    return config, args.force_initial_ft


def main() -> None:
    config, force_initial_ft = parse_args()
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(asdict(config))

    model = build_model(config)
    train_loader, val_loader, classes = build_loaders(config, model)
    replace_head_for_subset(model, classes)
    model.to(device)

    print(f"before fine-tuning (1k head restricted to 100 classes): {evaluate_topk(model, val_loader, device)}")
    load_or_fine_tune(model, train_loader, val_loader, classes, config, device, force_initial_ft)

    # pruning rate 0 = the fine-tuned dense model
    record = new_record()
    dense = evaluate_topk(model, val_loader, device)
    _, total = count_parameters(current_masks(model))
    add_record(record, 0.0, total, dense, dense)
    save_record(record, config)
    print(f"rate=0.00 params={total} top1={dense['top1']:.2%}")

    if config.mode == "iterative":
        run_iterative(model, train_loader, val_loader, config, device, record)
    else:
        run_oneshot(model, train_loader, val_loader, config, device, record)
    print(f"saved: {config.output_dir / 'data.pkl'}")


if __name__ == "__main__":
    main()
