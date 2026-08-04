"""Evaluate the saved ImageNet-100 12-layer LoRA checkpoint without training."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
TRAINING_SCRIPT = HERE / "evaluate_vit12 LoRAave本家.py"
RESULTS_DIR = HERE / "results_imagenet100_lora"
CHECKPOINT = RESULTS_DIR / "best_lora_imagenet100.pt"
CSV_PATH = RESULTS_DIR / "evaluation_comparison.csv"
FIGURE_PATH = RESULTS_DIR / "evaluation_comparison.png"


def load_training_module():
    spec = importlib.util.spec_from_file_location("imagenet100_lora_training", TRAINING_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load training script: {TRAINING_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parameter_counts(model):
    return (
        sum(parameter.numel() for parameter in model.parameters()),
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    )


def save_results(rows, plt):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fields = ("model", "blocks", "total_parameters", "trainable_parameters", "top1_accuracy")
    with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    labels = [row["model"] for row in rows]
    accuracies = [row["top1_accuracy"] for row in rows]
    figure, axis = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    bars = axis.bar(labels, accuracies, color=("#444444", "#999999", "#377eb8"))
    axis.set_ylabel("ImageNet-100 Top-1 accuracy (%)")
    axis.set_title("24-block vs. even-block 12-layer LoRA recovery")
    axis.set_ylim(0, min(100, max(accuracies) + 8))
    axis.grid(axis="y", alpha=0.3)
    for bar, accuracy in zip(bars, accuracies):
        axis.text(bar.get_x() + bar.get_width() / 2, accuracy, f"{accuracy:.2f}%", ha="center", va="bottom")
    figure.savefig(FIGURE_PATH, dpi=180)
    plt.close(figure)


def main():
    if not TRAINING_SCRIPT.is_file():
        raise FileNotFoundError(f"Training script not found: {TRAINING_SCRIPT}")
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"LoRA checkpoint not found: {CHECKPOINT}")

    experiment = load_training_module()
    experiment.NUM_WORKERS = 2
    _, validation_dataset, class_names = experiment.build_datasets()
    validation_loader = experiment.make_loader(validation_dataset, experiment.EVAL_BATCH_SIZE, shuffle=False)
    class_indices = experiment.imagenet100_to_imagenet1k_indices(class_names)

    print(f"Device: {experiment.DEVICE}")
    print("Evaluating original 24-block ViT-Large...")
    original = experiment.timm.create_model(experiment.MODEL_NAME, pretrained=True).to(experiment.DEVICE)
    for parameter in original.parameters():
        parameter.requires_grad_(False)
    original_total, original_trainable = parameter_counts(original)
    _, original_accuracy = experiment.evaluate(
        original, validation_loader, class_indices=class_indices, description="Original 24-block"
    )
    del original
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Evaluating even-block 12-layer ViT-Large...")
    compressed = experiment.build_model().to(experiment.DEVICE)
    compressed_total, compressed_trainable = parameter_counts(compressed)
    _, compressed_accuracy = experiment.evaluate(
        compressed, validation_loader, class_indices=class_indices, description="Even-block 12-layer"
    )

    print("Loading and evaluating the best LoRA checkpoint...")
    try:
        checkpoint = torch.load(CHECKPOINT, map_location=experiment.DEVICE, weights_only=False)
    except TypeError:
        checkpoint = torch.load(CHECKPOINT, map_location=experiment.DEVICE)
    compressed.load_state_dict(checkpoint["model_state_dict"], strict=True)
    _, lora_accuracy = experiment.evaluate(
        compressed, validation_loader, class_indices=class_indices, description="12-layer + LoRA"
    )

    rows = [
        {"model": "Original ViT-Large", "blocks": 24, "total_parameters": original_total, "trainable_parameters": original_trainable, "top1_accuracy": original_accuracy},
        {"model": "Even-block 12-layer", "blocks": 12, "total_parameters": compressed_total, "trainable_parameters": compressed_trainable, "top1_accuracy": compressed_accuracy},
        {"model": "12-layer + LoRA only", "blocks": 12, "total_parameters": compressed_total, "trainable_parameters": compressed_trainable, "top1_accuracy": lora_accuracy},
    ]
    print("\nModel comparison (ImageNet-100 validation)")
    print(f"{'Model':<28} {'Blocks':>6} {'Total params':>16} {'Trainable':>16} {'Top-1':>9}")
    print("-" * 82)
    for row in rows:
        print(f"{row['model']:<28} {row['blocks']:>6} {row['total_parameters']:>16,} {row['trainable_parameters']:>16,} {row['top1_accuracy']:>8.2f}%")
    save_results(rows, experiment.plt)
    print(f"\nComparison CSV: {CSV_PATH}")
    print(f"Comparison graph: {FIGURE_PATH}")


if __name__ == "__main__":
    main()

