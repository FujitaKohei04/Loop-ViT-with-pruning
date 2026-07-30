from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice

import torch
from torch import nn

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for minimal environments
    def tqdm(iterable, **kwargs):
        return iterable

Batch = tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class FineTuningConfig:
    epochs: int = 0
    lr: float = 1e-5
    weight_decay: float = 1e-4
    max_batches_per_epoch: int | None = None

    def __post_init__(self) -> None:
        if self.epochs < 0:
            raise ValueError("fine-tune epochs must be >= 0")
        if self.lr <= 0:
            raise ValueError("fine-tune learning rate must be > 0")
        if self.weight_decay < 0:
            raise ValueError("fine-tune weight decay must be >= 0")
        if self.max_batches_per_epoch is not None and self.max_batches_per_epoch < 1:
            raise ValueError("fine-tune batches must be >= 1")


def fine_tune_model(
    model: nn.Module,
    batches: Iterable[Batch],
    loss_fn: nn.Module,
    device: torch.device,
    config: FineTuningConfig,
) -> list[dict[str, float]]:
    """Fine-tune a pruned model while PyTorch pruning masks remain active."""

    if config.epochs == 0:
        return []

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    history: list[dict[str, float]] = []

    scaler = torch.amp.GradScaler("cuda")

    for epoch in tqdm(
        range(1, config.epochs + 1),
        desc="Fine-tuning",
        unit="epoch",
        leave=True,
    ):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_batches = (
            islice(batches, config.max_batches_per_epoch)
            if config.max_batches_per_epoch is not None
            else batches
        )

        batch_iter = tqdm(
            epoch_batches,
            total=config.max_batches_per_epoch,
            desc=f"Epoch {epoch}",
            leave=False,
            unit="batch",
        )
        for images, targets in batch_iter:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = loss_fn(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = targets.numel()
            running_loss += float(loss.item()) * batch_size
            correct += int(logits.argmax(dim=1).eq(targets).sum().item())
            total += batch_size

            if total:
                batch_iter.set_postfix(loss=running_loss / total)

        epoch_stats = {
            "epoch": float(epoch),
            "loss": running_loss / total if total else 0.0,
            "top1": correct / total if total else 0.0,
        }
        history.append(epoch_stats)

    return history
