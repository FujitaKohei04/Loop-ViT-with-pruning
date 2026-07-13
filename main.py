from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torch.nn.utils import prune
from torch.utils.data import DataLoader
import timm
from timm.data import create_transform, resolve_model_data_config

import SReT

import random
import numpy as np

Batch = tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class TaylorPruningConfig:
    """Configuration for first-order Taylor pruning."""

    amount: float = 0.3
    prune_bias: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.amount < 1.0:
            raise ValueError("amount must satisfy 0.0 <= amount < 1.0")


def iter_prunable_linear_layers(model: nn.Module) -> Iterable[tuple[str, nn.Linear]]:
    """Yield Linear layers that can be safely masked without changing ViT shapes."""

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            yield name, module


def collect_taylor_scores(
    model: nn.Module,
    batches: Iterable[Batch],
    loss_fn: nn.Module,
    device: torch.device,
    *,
    prune_bias: bool = False,
) -> dict[tuple[str, str], torch.Tensor]:
    """Accumulate first-order Taylor scores, abs(parameter * gradient)."""

    scores: dict[tuple[str, str], torch.Tensor] = {}
    model.train()
    model.zero_grad(set_to_none=True)

    for images, targets in batches:
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, targets)
        loss.backward()

        for name, module in iter_prunable_linear_layers(model):
            if module.weight.grad is not None:
                key = (name, "weight")
                score = (module.weight.detach() * module.weight.grad.detach()).abs()
                scores[key] = scores.get(key, torch.zeros_like(score)) + score

            if prune_bias and module.bias is not None and module.bias.grad is not None:
                key = (name, "bias")
                score = (module.bias.detach() * module.bias.grad.detach()).abs()
                scores[key] = scores.get(key, torch.zeros_like(score)) + score

        model.zero_grad(set_to_none=True)

    return scores


def global_mask_from_taylor_scores(
    scores: dict[tuple[str, str], torch.Tensor],
    amount: float,
) -> dict[tuple[str, str], torch.Tensor]:
    """Create binary masks that prune the lowest Taylor scores globally."""

    if amount == 0.0 or not scores:
        return {key: torch.ones_like(score) for key, score in scores.items()}

    flat_scores = torch.cat([score.reshape(-1).float().cpu() for score in scores.values()])
    num_to_prune = int(flat_scores.numel() * amount)
    if num_to_prune < 1:
        return {key: torch.ones_like(score) for key, score in scores.items()}

    prune_indices = torch.topk(flat_scores, k=num_to_prune, largest=False).indices
    flat_mask = torch.ones(flat_scores.numel())
    flat_mask[prune_indices] = False

    masks: dict[tuple[str, str], torch.Tensor] = {}
    offset = 0
    for key, score in scores.items():
        numel = score.numel()
        masks[key] = flat_mask[offset : offset + numel].reshape_as(score).to(score.device)
        offset += numel

    return masks


def apply_taylor_pruning(
    model: nn.Module,
    batches: Iterable[Batch],
    loss_fn: nn.Module,
    config: TaylorPruningConfig,
    device: torch.device,
) -> dict[str, float]:
    """Apply global Taylor pruning masks to all Linear layers in the model."""

    scores = collect_taylor_scores(
        model,
        batches,
        loss_fn,
        device,
        prune_bias=config.prune_bias,
    )
    masks = global_mask_from_taylor_scores(scores, config.amount)
    modules = dict(iter_prunable_linear_layers(model))

    pruned = 0
    total = 0
    for (module_name, parameter_name), mask in masks.items():
        module = modules[module_name]
        prune.custom_from_mask(module, parameter_name, mask=mask)
        pruned += mask.numel() - int(mask.sum().item())
        total += mask.numel()

    sparsity = pruned / total if total else 0.0
    return {
        "pruned_parameters": float(pruned),
        "total_prunable_parameters": float(total),
        "sparsity": sparsity,
    }


def build_imagenet_loaders(
    root: str | Path,
    model: nn.Module,
    *,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader[Batch], DataLoader[Batch]]:
    """Build ImageNet train/val loaders for a standard ImageFolder layout."""

    from torchvision import datasets

    root = Path(root).expanduser()
    train_dir = root / "train"
    val_dir = root / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(
            "ImageNet root must contain 'train' and 'val' directories, "
            f"but got: {root}"
        )

    data_config = resolve_model_data_config(model)
    train_transform = create_transform(**data_config, is_training=True)
    val_transform = create_transform(**data_config, is_training=False)

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    train_loader: DataLoader[Batch] = DataLoader(
        train_dataset,
        shuffle=False,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader: DataLoader[Batch] = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    return train_loader, val_loader


def build_imagenet_val_loader(
    root: str | Path,
    model: nn.Module,
    *,
    batch_size: int,
    num_workers: int,
) -> DataLoader[Batch]:
    """Build an ImageNet validation loader for test-only evaluation."""

    from torchvision import datasets

    root = Path(root).expanduser()
    val_dir = root / "val"
    if not val_dir.is_dir():
        raise FileNotFoundError(
            "ImageNet root must contain a 'val' directory, "
            f"but got: {root}"
        )

    data_config = resolve_model_data_config(model)
    val_transform = create_transform(**data_config, is_training=False)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    return DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


@torch.inference_mode()
def evaluate_topk(
    model: nn.Module,
    batches: Iterable[Batch],
    device: torch.device,
    *,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Evaluate top-1 and top-5 accuracy."""

    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    eval_batches = islice(batches, max_batches) if max_batches is not None else batches

    for images, targets in eval_batches:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)

        max_k = min(5, logits.shape[-1])
        predictions = logits.topk(max_k, dim=1).indices
        correct = predictions.eq(targets[:, None])
        correct_top1 += int(correct[:, :1].sum().item())
        correct_top5 += int(correct[:, :max_k].sum().item())
        total += targets.numel()

    if total == 0:
        return {"top1": 0.0, "top5": 0.0}

    return {
        "top1": correct_top1 / total,
        "top5": correct_top5 / total,
    }


def make_random_calibration_batches(
    *,
    num_batches: int,
    batch_size: int,
    num_classes: int,
    image_size: int,
    device: torch.device,
) -> Iterable[Batch]:
    """Small demo calibration stream. Replace this with real train batches."""

    generator = torch.Generator()
    generator.manual_seed(0)
    for _ in range(num_batches):
        images = torch.randn(
            batch_size,
            3,
            image_size,
            image_size,
            generator=generator,
        ).to(device)
        targets = torch.randint(
            0,
            num_classes,
            (batch_size,),
            generator=generator,
        ).to(device)
        yield images, targets

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run timm ViT with optional Taylor pruning.")
    parser.add_argument("--model-name", default="vit_base_patch16_224")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--calibration-batches", type=int, default=1)
    parser.add_argument("--taylor-prune-amount", type=float, default=0.3)
    parser.add_argument("--prune-bias", action="store_true")
    parser.add_argument(
        "--imagenet-root",
        default=None,
        help="Path containing ImageNet train/ and val/ directories.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=None,
        help="Limit ImageNet validation batches for a quick smoke test.",
    )
    parser.add_argument(
        "--skip-baseline-eval",
        action="store_true",
        help="Skip pre-pruning ImageNet validation.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only evaluate the pretrained model on ImageNet val; skip pruning.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(42)
    device = torch.device(args.device)

    if(args.model_name != "sret_s"):
        model = timm.create_model(
            args.model_name,
            pretrained=args.pretrained,
            num_classes=args.num_classes,
        ).to(device)
    else:
        model = SReT.SReT_S(pretrained=False)
        model.load_state_dict(torch.load('./pre-trained/SReT_S.pth')['model'])
        model.to(device)

    if args.eval_only:
        if args.imagenet_root is None:
            raise ValueError("--eval-only requires --imagenet-root")
        val_loader = build_imagenet_val_loader(
            args.imagenet_root,
            model,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        metrics = evaluate_topk(
            model,
            val_loader,
            device,
            max_batches=args.eval_batches,
        )
        print(f"ImageNet val top-1: {metrics['top1']:.2%}")
        print(f"ImageNet val top-5: {metrics['top5']:.2%}")
        return

    val_loader = None
    if args.imagenet_root is not None:
        train_loader, val_loader = build_imagenet_loaders(
            args.imagenet_root,
            model,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        calibration_batches = islice(train_loader, args.calibration_batches)
    else:
        calibration_batches = make_random_calibration_batches(
            num_batches=args.calibration_batches,
            batch_size=args.batch_size,
            num_classes=args.num_classes,
            image_size=args.image_size,
            device=device,
        )

    baseline_metrics = None
    if val_loader is not None and not args.skip_baseline_eval:
        baseline_metrics = evaluate_topk(
            model,
            val_loader,
            device,
            max_batches=args.eval_batches,
        )

    pruning_stats = apply_taylor_pruning(
        model,
        calibration_batches,
        nn.CrossEntropyLoss(),
        TaylorPruningConfig(
            amount=args.taylor_prune_amount,
            prune_bias=args.prune_bias,
        ),
        device,
    )

    if val_loader is None:
        model.eval()
        with torch.inference_mode():
            output = model(torch.randn(1, 3, args.image_size, args.image_size, device=device))
        print(output.shape)

    print(
        "Taylor pruning sparsity: "
        f"{pruning_stats['sparsity']:.2%} "
        f"({int(pruning_stats['pruned_parameters'])}/"
        f"{int(pruning_stats['total_prunable_parameters'])})"
    )
    if val_loader is not None:
        metrics = evaluate_topk(
            model,
            val_loader,
            device,
            max_batches=args.eval_batches,
        )
        if baseline_metrics is not None:
            print(
                "ImageNet val top-1: "
                f"{baseline_metrics['top1']:.2%} -> {metrics['top1']:.2%} "
                f"({metrics['top1'] - baseline_metrics['top1']:+.2%})"
            )
            print(
                "ImageNet val top-5: "
                f"{baseline_metrics['top5']:.2%} -> {metrics['top5']:.2%} "
                f"({metrics['top5'] - baseline_metrics['top5']:+.2%})"
            )
        else:
            print(f"ImageNet val top-1: {metrics['top1']:.2%}")
            print(f"ImageNet val top-5: {metrics['top5']:.2%}")


if __name__ == "__main__":
    main()
