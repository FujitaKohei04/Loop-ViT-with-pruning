from __future__ import annotations

import argparse
import os
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

import LoopViT
import SReT

import random
import numpy as np

from fine_tuning import FineTuningConfig, fine_tune_model
from imagenet_labels import remap_imagenet_targets
from analysis.statistics import collect_weights
from ptflops import get_model_complexity_info
import time

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib-cache"))
import matplotlib.pyplot as plt

Batch = tuple[torch.Tensor, torch.Tensor]

@dataclass
class Analyzing_data:
    weights: dict[str, torch.Tensor]
    activations: dict[str, torch.Tensor] | None = None


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


def _is_activation_target(module: nn.Module, target: str) -> bool:
    if target == "linear":
        return isinstance(module, nn.Linear)
    if target == "conv":
        return isinstance(module, nn.Conv2d)
    if target == "linear-conv":
        return isinstance(module, (nn.Linear, nn.Conv2d))
    if target == "all-leaf":
        return len(list(module.children())) == 0
    raise ValueError(f"Unknown activation target: {target}")


def iter_activation_layers(
    model: nn.Module,
    target: str,
) -> Iterable[tuple[str, nn.Module]]:
    """Yield modules whose forward outputs should be collected."""

    for name, module in model.named_modules():
        if name and _is_activation_target(module, target):
            yield name, module


def _flatten_activation_output(output: object) -> torch.Tensor | None:
    if isinstance(output, torch.Tensor):
        return output.detach().flatten().cpu()
    if isinstance(output, (list, tuple)):
        tensors = [
            item.detach().flatten().cpu()
            for item in output
            if isinstance(item, torch.Tensor)
        ]
        if tensors:
            return torch.cat(tensors)
    return None


@torch.inference_mode()
def collect_activations(
    model: nn.Module,
    batches: Iterable[Batch],
    device: torch.device,
    *,
    target: str = "linear",
    max_elements_per_layer: int = 200_000,
) -> dict[str, torch.Tensor]:
    """Collect sampled forward activations for selected modules."""

    activations: dict[str, list[torch.Tensor]] = {}
    counts: dict[str, int] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> None:
            flat = _flatten_activation_output(output)
            if flat is None:
                return

            remaining = max_elements_per_layer - counts.get(name, 0)
            if remaining <= 0:
                return

            sample = flat[:remaining]
            activations.setdefault(name, []).append(sample)
            counts[name] = counts.get(name, 0) + sample.numel()

        return hook

    for name, module in iter_activation_layers(model, target):
        handles.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    try:
        for images, _targets in batches:
            images = images.to(device, non_blocking=True)
            model(images)
    finally:
        for handle in handles:
            handle.remove()

    return {
        name: torch.cat(values)
        for name, values in activations.items()
        if values
    }


def _safe_plot_name(name: str) -> str:
    return name.replace("/", "_").replace(".", "_")


def save_tensor_histograms(
    tensors: dict[str, torch.Tensor],
    output_dir: str | Path,
    *,
    prefix: str,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, tensor in tensors.items():
        plt.figure()
        if xlim is not None:
            plt.xlim(*xlim)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.hist(
            tensor.flatten().numpy(),
            bins=100,
        )
        plt.title(name)
        plt.savefig(output_dir / f"{prefix}_{_safe_plot_name(name)}.png")
        plt.close()


def save_weight_histograms(
    weights: dict[str, torch.Tensor],
    model_name: str,
) -> None:
    save_tensor_histograms(
        weights,
        Path("analysis-data") / "weights" / model_name,
        prefix="weight",
        xlim=(-1, 1),
        ylim=(0, 100000),
    )


def save_activation_outputs(
    activations: dict[str, torch.Tensor],
    model_name: str,
    *,
    save_tensors: bool,
) -> None:
    output_dir = Path("analysis-data") / "activations" / model_name
    save_tensor_histograms(
        activations,
        output_dir,
        prefix="activation",
    )
    if save_tensors:
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(activations, output_dir / "activations.pt")


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
    remap_imagenet_targets(train_dataset)
    remap_imagenet_targets(val_dataset)

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
    remap_imagenet_targets(val_dataset)

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

def measure_complexity(model, image_size):
    model.eval()

    macs, params = get_model_complexity_info(
        model,
        (3, image_size, image_size),
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False,
    )

    gflops = macs * 2 / 1e9

    return {
        "params": params,
        "gflops": gflops,
    }

def measure_latency(
    model,
    device,
    image_size,
    batch_size,
    repeat=100,
):
    model.eval()

    x = torch.randn(
        batch_size,
        3,
        image_size,
        image_size,
        device=device,
    )

    # Warm-up
    with torch.inference_mode():
        for _ in range(20):
            model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        for _ in range(repeat):
            model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    end = time.perf_counter()

    latency = (end - start) * 1000 / repeat

    return latency

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
    parser.add_argument("--model-name-normal")
    parser.add_argument("--model-name-comp-loop")
    parser.add_argument("--model-name-loop")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--calibration-batches", type=int, default=1)
    parser.add_argument("--taylor-prune-amount", type=float, default=0.3)
    parser.add_argument("--prune-bias", action="store_true")
    parser.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=0,
        help="Fine-tune for this many epochs after Taylor pruning.",
    )
    parser.add_argument(
        "--fine-tune-lr",
        type=float,
        default=1e-5,
        help="Learning rate used for post-pruning fine-tuning.",
    )
    parser.add_argument(
        "--fine-tune-weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay used for post-pruning fine-tuning.",
    )
    parser.add_argument(
        "--fine-tune-batches",
        type=int,
        default=None,
        help="Limit fine-tuning batches per epoch for quick experiments.",
    )
    parser.add_argument(
        "--collect-activations",
        action="store_true",
        help="Collect baseline forward activations for each model.",
    )
    parser.add_argument(
        "--activation-batches",
        type=int,
        default=1,
        help="Number of batches used for activation collection.",
    )
    parser.add_argument(
        "--activation-target",
        choices=("linear", "conv", "linear-conv", "all-leaf"),
        default="linear",
        help="Module type whose forward activations are collected.",
    )
    parser.add_argument(
        "--activation-max-elements-per-layer",
        type=int,
        default=200_000,
        help="Maximum number of flattened activation values kept per layer.",
    )
    parser.add_argument(
        "--save-activation-tensors",
        action="store_true",
        help="Also save collected activation tensors as activations.pt.",
    )
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

def run_model(model, device, args, model_name) -> Analyzing_data | None:

    print(f"\n\n===== Running model =====: {model_name}")
    baseline_complexity = measure_complexity(
        model,
        args.image_size,
    )

    baseline_latency = measure_latency(
        model,
        device,
        args.image_size,
        args.batch_size,
    )

    if args.eval_only:
        if args.imagenet_root is None:
            raise ValueError("--eval-only requires --imagenet-root")
        val_loader = build_imagenet_val_loader(
            args.imagenet_root,
            model,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        if args.collect_activations:
            activations = collect_activations(
                model,
                islice(val_loader, args.activation_batches),
                device,
                target=args.activation_target,
                max_elements_per_layer=args.activation_max_elements_per_layer,
            )
            save_activation_outputs(
                activations,
                model_name,
                save_tensors=args.save_activation_tensors,
            )
            print(f"Saved activations: {len(activations)} layers")
        metrics = evaluate_topk(
            model,
            val_loader,
            device,
            max_batches=args.eval_batches,
        )
        print(f"ImageNet val top-1: {metrics['top1']:.2%}")
        print(f"ImageNet val top-5: {metrics['top5']:.2%}")
        return

    train_loader = None
    val_loader = None
    activation_batches = None
    if args.imagenet_root is not None:
        train_loader, val_loader = build_imagenet_loaders(
            args.imagenet_root,
            model,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        calibration_batches = islice(train_loader, args.calibration_batches)
        activation_batches = islice(train_loader, args.activation_batches)
    else:
        calibration_batches = make_random_calibration_batches(
            num_batches=args.calibration_batches,
            batch_size=args.batch_size,
            num_classes=args.num_classes,
            image_size=args.image_size,
            device=device,
        )
        activation_batches = make_random_calibration_batches(
            num_batches=args.activation_batches,
            batch_size=args.batch_size,
            num_classes=args.num_classes,
            image_size=args.image_size,
            device=device,
        )
    if args.fine_tune_epochs > 0 and train_loader is None:
        raise ValueError("--fine-tune-epochs requires --imagenet-root")

    baseline_metrics = None
    if val_loader is not None and not args.skip_baseline_eval:
        baseline_metrics = evaluate_topk(
            model,
            val_loader,
            device,
            max_batches=args.eval_batches,
        )
    
    weights = collect_weights(model)
    save_weight_histograms(weights, model_name)

    activations = None
    if args.collect_activations:
        activations = collect_activations(
            model,
            activation_batches,
            device,
            target=args.activation_target,
            max_elements_per_layer=args.activation_max_elements_per_layer,
        )
        save_activation_outputs(
            activations,
            model_name,
            save_tensors=args.save_activation_tensors,
        )
        print(f"Saved activations: {len(activations)} layers")

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

    pruned_complexity = measure_complexity(
        model,
        args.image_size,
    )

    pruned_latency = measure_latency(
        model,
        device,
        args.image_size,
        args.batch_size,
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
    pruned_metrics = None
    if val_loader is not None:
        pruned_metrics = evaluate_topk(
            model,
            val_loader,
            device,
            max_batches=args.eval_batches,
        )
        if baseline_metrics is not None:
            print(
                "ImageNet val top-1: "
                f"{baseline_metrics['top1']:.2%} -> {pruned_metrics['top1']:.2%} "
                f"({pruned_metrics['top1'] - baseline_metrics['top1']:+.2%})"
            )
            print(
                "ImageNet val top-5: "
                f"{baseline_metrics['top5']:.2%} -> {pruned_metrics['top5']:.2%} "
                f"({pruned_metrics['top5'] - baseline_metrics['top5']:+.2%})"
            )
            print("========== Complexity ==========")

            print(
                f"Params : "
                f"{baseline_complexity['params']/1e6:.2f} M"
            )

            print(
                f"GFLOPs : "
                f"{baseline_complexity['gflops']:.2f}"
            )

            print(
                f"Latency : "
                f"{baseline_latency:.2f} ms"
            )

            print()

            print("========== After Pruning ==========")

            print(
                f"Params : "
                f"{pruned_complexity['params']/1e6:.2f} M"
            )

            print(
                f"GFLOPs : "
                f"{pruned_complexity['gflops']:.2f}"
            )

            print(
                f"Latency : "
                f"{pruned_latency:.2f} ms"
            )
        else:
            print(f"ImageNet val top-1: {pruned_metrics['top1']:.2%}")
            print(f"ImageNet val top-5: {pruned_metrics['top5']:.2%}")

    if args.fine_tune_epochs > 0:
        print("========== Fine-tuning ==========")
        fine_tune_history = fine_tune_model(
            model,
            train_loader,
            nn.CrossEntropyLoss(),
            device,
            FineTuningConfig(
                epochs=args.fine_tune_epochs,
                lr=args.fine_tune_lr,
                weight_decay=args.fine_tune_weight_decay,
                max_batches_per_epoch=args.fine_tune_batches,
            ),
        )
        for epoch_stats in fine_tune_history:
            print(
                "Fine-tune epoch "
                f"{int(epoch_stats['epoch'])}/{args.fine_tune_epochs}: "
                f"loss={epoch_stats['loss']:.4f}, "
                f"top-1={epoch_stats['top1']:.2%}"
            )
        if val_loader is not None:
            fine_tuned_metrics = evaluate_topk(
                model,
                val_loader,
                device,
                max_batches=args.eval_batches,
            )
            print("========== After Fine-tuning ==========")
            if pruned_metrics is not None:
                print(
                    "ImageNet val top-1: "
                    f"{pruned_metrics['top1']:.2%} -> {fine_tuned_metrics['top1']:.2%} "
                    f"({fine_tuned_metrics['top1'] - pruned_metrics['top1']:+.2%})"
                )
                print(
                    "ImageNet val top-5: "
                    f"{pruned_metrics['top5']:.2%} -> {fine_tuned_metrics['top5']:.2%} "
                    f"({fine_tuned_metrics['top5'] - pruned_metrics['top5']:+.2%})"
                )
            else:
                print(f"ImageNet val top-1: {fine_tuned_metrics['top1']:.2%}")
                print(f"ImageNet val top-5: {fine_tuned_metrics['top5']:.2%}")

    return Analyzing_data(weights=weights, activations=activations)

def main() -> None:
    args = parse_args()
    set_seed(42)
    device = torch.device(args.device)
    model_normal = model_comp_loop = model_loop = None

    if(args.model_name_normal == "vit_base_patch16_224"):
        model_normal = timm.create_model(
            args.model_name_normal,
            pretrained=args.pretrained,
            num_classes=args.num_classes,
        ).to(device)
    if(args.model_name_comp_loop == "sret_s"):
        model_comp_loop = SReT.SReT_S(pretrained=False)
        model_comp_loop.load_state_dict(torch.load('./pre-trained/SReT_S.pth')['model'])
        model_comp_loop.to(device)

    if(args.model_name_loop == "loop_vit"):
        model_loop = LoopViT.build_loop_vit_base_patch16_224(
            pretrained=False,
            use_iteration_embedding=False,
        ).to(device)

    if model_loop is not None:
        analysis_data_loop = run_model(model_loop, device, args, args.model_name_loop)
    if model_normal is not None:
        analysis_data_normal = run_model(model_normal, device, args, args.model_name_normal)
    if model_comp_loop is not None:
        analysis_data_comp_loop = run_model(model_comp_loop, device, args, args.model_name_comp_loop)


if __name__ == "__main__":
    main()
