"""LoRA-only ImageNet-100 fine-tuning for a pretrained ViT-Large.

The training set is split into train.X1 ... train.X4 (25 classes each).
Labels are remapped to the 100-class index used by val.X before concatenation.
"""

from __future__ import annotations

import csv
import math
import os
import random
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import timm
from timm.data.imagenet_info import ImageNetInfo
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.auto import tqdm


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_NAME = "vit_large_patch16_224"
NUM_CLASSES = 100
EPOCHS = 40  # Longer schedule for LoRA-only recovery after 24-to-12 block compression.
WARMUP_EPOCHS = 2
EARLY_STOPPING_PATIENCE = 5  # Stop after this many epochs without a Top-1 improvement.
EARLY_STOPPING_MIN_DELTA = 0.0
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 16  # Effective batch size: 64.
LORA_RANK = 16
LORA_ALPHA = 16.0
LORA_DROPOUT = 0.05
LORA_LEARNING_RATE = 1.0e-4
FULL_FINE_TUNE_LEARNING_RATE = 1.0e-5
HEAD_LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 0.01
NUM_WORKERS = 2  # Reduce WSL RAM usage during parallel JPEG decoding.
SEED = 42
USE_AMP = True

# The first path is used on Windows and the second path is used from WSL.
DATA_ROOT_CANDIDATES = (
    Path(r"C:\Users\take\Desktop\imagenet100"),
    Path("/mnt/c/Users/take/Desktop/imagenet100"),
)
TRAIN_PARTITIONS = ("train.X1", "train.X2", "train.X3", "train.X4")
VALIDATION_PARTITION = "val.X"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# Dataset handling
# -----------------------------------------------------------------------------
def resolve_data_root() -> Path:
    """Find ImageNet-100 on either Windows or WSL."""
    for root in DATA_ROOT_CANDIDATES:
        if (root / VALIDATION_PARTITION).is_dir() and all(
            (root / partition).is_dir() for partition in TRAIN_PARTITIONS
        ):
            return root
    searched = "\n  ".join(str(path) for path in DATA_ROOT_CANDIDATES)
    raise FileNotFoundError(
        "ImageNet-100 was not found. Expected val.X and train.X1 ... train.X4 under:\n"
        f"  {searched}"
    )


class ClassMappedImageFolder(datasets.ImageFolder):
    """ImageFolder whose local class IDs are remapped to a global class map."""

    def __init__(
        self,
        root: Path,
        class_to_idx: dict[str, int],
        transform: transforms.Compose,
    ) -> None:
        super().__init__(root, transform=transform)
        unknown_classes = set(self.classes) - set(class_to_idx)
        if unknown_classes:
            raise ValueError(f"Unknown classes in {root}: {sorted(unknown_classes)}")

        local_classes = list(self.classes)
        self.samples = [
            (path, class_to_idx[local_classes[local_target]])
            for path, local_target in self.samples
        ]
        self.imgs = self.samples
        self.targets = [target for _, target in self.samples]
        self.classes = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
        self.class_to_idx = dict(class_to_idx)


def build_datasets() -> tuple[Dataset, Dataset, list[str]]:
    data_root = resolve_data_root()
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    validation_dataset = datasets.ImageFolder(
        data_root / VALIDATION_PARTITION, transform=evaluation_transform
    )
    if len(validation_dataset.classes) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} validation classes, found {len(validation_dataset.classes)}."
        )

    train_datasets = [
        ClassMappedImageFolder(
            data_root / partition,
            validation_dataset.class_to_idx,
            train_transform,
        )
        for partition in TRAIN_PARTITIONS
    ]
    train_dataset: Dataset = ConcatDataset(train_datasets)
    return train_dataset, validation_dataset, validation_dataset.classes

def imagenet100_to_imagenet1k_indices(class_names: list[str]) -> list[int]:
    """Map ImageFolder synset names to the pretrained ImageNet-1K logit order."""
    imagenet1k_synsets = ImageNetInfo("imagenet-1k").label_names()
    synset_to_index = {synset: index for index, synset in enumerate(imagenet1k_synsets)}
    missing = [class_name for class_name in class_names if class_name not in synset_to_index]
    if missing:
        raise ValueError(f"Classes are not ImageNet-1K synsets: {missing}")
    return [synset_to_index[class_name] for class_name in class_names]

# -----------------------------------------------------------------------------
# LoRA
# -----------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """Frozen Linear layer plus trainable low-rank update."""

    def __init__(self, base_linear: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.rank = min(rank, self.in_features, self.out_features)
        self.scaling = alpha / self.rank
        self.base = base_linear
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

        self.lora_A = nn.Parameter(torch.empty(self.rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base(x)
        lora_output = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return base_output + self.scaling * lora_output


def replace_block_linears_with_lora(module: nn.Module) -> None:
    """Replace all Linear layers inside transformer blocks with LoRA adapters."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, LORA_RANK, LORA_ALPHA, LORA_DROPOUT))
        else:
            replace_block_linears_with_lora(child)




def build_even_block_model(use_lora: bool, train_blocks: bool = False) -> nn.Module:
    """Build a 12-block ViT-Large from original even-indexed blocks (0, 2, ..., 22)."""
    model = timm.create_model(MODEL_NAME, pretrained=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    original_blocks = list(model.blocks)
    if len(original_blocks) != 24:
        raise ValueError(f"Expected a 24-block ViT-Large, found {len(original_blocks)} blocks.")
    model.blocks = nn.Sequential(*(original_blocks[index] for index in range(0, 24, 2)))
    if use_lora:
        replace_block_linears_with_lora(model.blocks)
    elif train_blocks:
        for parameter in model.blocks.parameters():
            parameter.requires_grad_(True)
    return model


def trainable_parameter_groups(model: nn.Module, learning_rate: float) -> list[dict]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("No trainable parameters were found.")
    return [{"params": parameters, "lr": learning_rate}]
# -----------------------------------------------------------------------------
# Training and evaluation
# -----------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
    )


def autocast_context():
    if DEVICE.type == "cuda" and USE_AMP:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    class_indices: list[int] | None = None,
    description: str = "Validation",
) -> tuple[float, float]:
    """Evaluate either a 100-class model or an ImageNet-1K model restricted to these 100 classes."""
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc=description, leave=False):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        with autocast_context():
            logits = model(images)
        if class_indices is not None:
            logits = logits[:, class_indices]
        loss_sum += F.cross_entropy(logits.float(), labels, reduction="sum").item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.numel()
    return loss_sum / total, 100.0 * correct / total


def clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
def parameter_counts(model: nn.Module) -> tuple[int, int]:
    return (
        sum(parameter.numel() for parameter in model.parameters()),
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    )


def output_paths(experiment_key: str) -> tuple[Path, Path, Path, Path]:
    output_dir = Path(__file__).resolve().parent / "results_imagenet100_lora"
    output_dir.mkdir(parents=True, exist_ok=True)
    return (
        output_dir / f"history_{experiment_key}.csv",
        output_dir / f"learning_curves_{experiment_key}.png",
        output_dir / f"best_{experiment_key}.pt",
        output_dir / "comparison_latest.csv",
    )


def save_history(
    history: dict[str, list[float]], csv_path: Path, figure_path: Path,
    original_accuracy: float, compressed_accuracy: float, label: str,
) -> None:
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=("epoch", "train_loss", "val_loss", "val_accuracy", "learning_rate"))
        writer.writeheader()
        for values in zip(history["epoch"], history["train_loss"], history["val_loss"], history["val_accuracy"], history["learning_rate"]):
            writer.writerow(dict(zip(writer.fieldnames, values)))

    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    loss_axis.plot(history["epoch"], history["train_loss"], label="train")
    loss_axis.plot(history["epoch"], history["val_loss"], label="validation")
    loss_axis.set(xlabel="Epoch", ylabel="Cross-entropy loss", title=f"{label}: loss")
    loss_axis.grid(alpha=0.3)
    loss_axis.legend()
    accuracy_axis.axhline(original_accuracy, color="black", linestyle=":", label="Original 24-block")
    accuracy_axis.axhline(compressed_accuracy, color="gray", linestyle="--", label="Even-block 12-layer")
    accuracy_axis.plot(history["epoch"], history["val_accuracy"], label=label)
    accuracy_axis.set(xlabel="Epoch", ylabel="Top-1 accuracy (%)", title=f"{label}: validation accuracy")
    accuracy_axis.grid(alpha=0.3)
    accuracy_axis.legend(fontsize=8)
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)


def save_comparison(rows: list[dict[str, object]], path: Path) -> None:
    fields = ("model", "blocks", "total_parameters", "trainable_parameters", "top1_accuracy")
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def print_comparison(rows: list[dict[str, object]]) -> None:
    print("\nModel comparison (ImageNet-100 validation)")
    print(f"{'Model':<32} {'Blocks':>6} {'Total params':>16} {'Trainable':>16} {'Top-1':>9}")
    print("-" * 88)
    for row in rows:
        print(f"{str(row['model']):<32} {int(row['blocks']):>6} {int(row['total_parameters']):>16,} {int(row['trainable_parameters']):>16,} {float(row['top1_accuracy']):>8.2f}%")


def train_experiment(
    model: nn.Module, experiment_key: str, label: str, learning_rate: float,
    train_loader: DataLoader, validation_loader: DataLoader, class_indices: list[int],
    class_names: list[str], original_accuracy: float, compressed_accuracy: float,
) -> tuple[float, int, int]:
    model = model.to(DEVICE)
    total_parameters, trainable_parameters = parameter_counts(model)
    optimizer = torch.optim.AdamW(trainable_parameter_groups(model, learning_rate), weight_decay=WEIGHT_DECAY)
    updates_per_epoch = math.ceil(len(train_loader) / GRADIENT_ACCUMULATION_STEPS)
    total_updates = EPOCHS * updates_per_epoch
    warmup_updates = WARMUP_EPOCHS * updates_per_epoch

    def lr_multiplier(step: int) -> float:
        if step < warmup_updates:
            return (step + 1) / max(1, warmup_updates)
        progress = (step - warmup_updates) / max(1, total_updates - warmup_updates)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda" and USE_AMP)
    csv_path, figure_path, checkpoint_path, _ = output_paths(experiment_key)
    initial_loss, initial_accuracy = evaluate(model, validation_loader, class_indices, f"{label} epoch 0")
    history: dict[str, list[float]] = {
        "epoch": [0.0], "train_loss": [float("nan")], "val_loss": [initial_loss],
        "val_accuracy": [initial_accuracy], "learning_rate": [0.0],
    }
    best_accuracy = initial_accuracy

    def save_checkpoint(epoch: int, accuracy: float) -> None:
        torch.save(
            {
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "val_accuracy": accuracy,
                "class_names": class_names,
                "config": {"model": MODEL_NAME, "classes": NUM_CLASSES, "blocks": 12,
                           "selected_blocks": list(range(0, 24, 2)), "experiment": experiment_key},
            }, checkpoint_path,
        )

    save_history(history, csv_path, figure_path, original_accuracy, compressed_accuracy, label)
    save_checkpoint(0, initial_accuracy)
    print(f"{label} | epoch  0/{EPOCHS} | val loss {initial_loss:.4f} | val acc {initial_accuracy:.2f}%")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for batch_index, (images, labels) in enumerate(tqdm(train_loader, desc=f"{label} [{epoch}/{EPOCHS}]")):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            with autocast_context():
                logits = model(images)[:, class_indices]
                loss = F.cross_entropy(logits, labels)
            loss_sum += loss.detach().item()
            scaler.scale(loss / GRADIENT_ACCUMULATION_STEPS).backward()
            if (batch_index + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or batch_index + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        train_loss = loss_sum / max(1, len(train_loader))
        val_loss, val_accuracy = evaluate(model, validation_loader, class_indices, f"{label} validation")
        history["epoch"].append(float(epoch))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["learning_rate"].append(scheduler.get_last_lr()[0])
        save_history(history, csv_path, figure_path, original_accuracy, compressed_accuracy, label)
        print(f"{label} | epoch {epoch:>2}/{EPOCHS} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | val acc {val_accuracy:.2f}% | lr {scheduler.get_last_lr()[0]:.2e}")
        if val_accuracy > best_accuracy + EARLY_STOPPING_MIN_DELTA:
            best_accuracy = val_accuracy
            epochs_without_improvement = 0
            save_checkpoint(epoch, val_accuracy)
            print(f"  Saved best checkpoint: {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            print(
                f"  No Top-1 improvement for {epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE} epoch(s)."
            )
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"  Early stopping at epoch {epoch}; best Top-1: {best_accuracy:.2f}%")
                break

    print(f"{label} best Top-1: {best_accuracy:.2f}%")
    print(f"  Learning curves: {figure_path}")
    print(f"  History CSV: {csv_path}")
    return best_accuracy, total_parameters, trainable_parameters


def main() -> None:
    seed_everything(SEED)
    print(f"Device: {DEVICE}")
    train_dataset, validation_dataset, class_names = build_datasets()
    print(f"Training images: {len(train_dataset):,}; validation images: {len(validation_dataset):,}")
    print(f"Classes: {len(class_names)}; epochs per experiment: {EPOCHS}")
    train_loader = make_loader(train_dataset, TRAIN_BATCH_SIZE, shuffle=True)
    validation_loader = make_loader(validation_dataset, EVAL_BATCH_SIZE, shuffle=False)
    class_indices = imagenet100_to_imagenet1k_indices(class_names)

    original = timm.create_model(MODEL_NAME, pretrained=True).to(DEVICE)
    for parameter in original.parameters():
        parameter.requires_grad_(False)
    original_total, original_trainable = parameter_counts(original)
    _, original_accuracy = evaluate(original, validation_loader, class_indices, "Original 24-block")
    del original
    clear_cuda_cache()

    compressed = build_even_block_model(use_lora=False, train_blocks=False).to(DEVICE)
    compressed_total, compressed_trainable = parameter_counts(compressed)
    _, compressed_accuracy = evaluate(compressed, validation_loader, class_indices, "Even-block 12-layer")
    del compressed
    clear_cuda_cache()
    print(f"Original 24-block Top-1: {original_accuracy:.2f}%")
    print(f"Even-block 12-layer Top-1: {compressed_accuracy:.2f}%")

    lora_accuracy, lora_total, lora_trainable = train_experiment(
        build_even_block_model(use_lora=True), "lora_only", "12-block LoRA only", LORA_LEARNING_RATE,
        train_loader, validation_loader, class_indices, class_names, original_accuracy, compressed_accuracy,
    )
    clear_cuda_cache()
    full_accuracy, full_total, full_trainable = train_experiment(
        build_even_block_model(use_lora=False, train_blocks=True), "full_finetune", "12-block full fine-tuning",
        FULL_FINE_TUNE_LEARNING_RATE, train_loader, validation_loader, class_indices, class_names,
        original_accuracy, compressed_accuracy,
    )

    rows: list[dict[str, object]] = [
        {"model": "Original ViT-Large", "blocks": 24, "total_parameters": original_total, "trainable_parameters": original_trainable, "top1_accuracy": original_accuracy},
        {"model": "Even-block 12-layer", "blocks": 12, "total_parameters": compressed_total, "trainable_parameters": compressed_trainable, "top1_accuracy": compressed_accuracy},
        {"model": "12-layer + LoRA only", "blocks": 12, "total_parameters": lora_total, "trainable_parameters": lora_trainable, "top1_accuracy": lora_accuracy},
        {"model": "12-layer full fine-tuning", "blocks": 12, "total_parameters": full_total, "trainable_parameters": full_trainable, "top1_accuracy": full_accuracy},
    ]
    _, _, _, comparison_path = output_paths("lora_only")
    print_comparison(rows)
    save_comparison(rows, comparison_path)
    print(f"Comparison CSV: {comparison_path}")


if __name__ == "__main__":
    main()


