"""Datasets for training RecurrentViT from scratch.

Nothing is downloaded here (download=False everywhere). Missing data raises
FileNotFoundError with the command to prepare it (see prepare_data.py).

Labels are 0..C-1 in the dataset's own order (ImageFolder: sorted WNIDs), not
ImageNet-1k indices, because the models are trained from scratch.

Augmentation profiles:
    paper: RandomResizedCrop(scale 0.08-1, ratio 3/4-4/3), flip, RandAugment,
           Random Erasing (paper Sec. 3.2). RandAugment n=2 m=9 (torchvision
           default) and erasing p=0.25 (DeiT) are assumptions.
    small: RandomCrop(padding=img/8), flip, RandAugment, Random Erasing.
           Deviation from the paper for 32/64 px images, where a 0.08 scale
           crop is too destructive.
    auto:  small for images <= 64 px, paper otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

Augment = Literal["auto", "paper", "small", "none"]

DATA_ROOT = Path("./data")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    num_classes: int
    img_size: int
    patch_size: int           # default patch size for RecurrentViT on this dataset
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    root: Path
    kind: Literal["cifar10", "cifar100", "imagefolder"]
    prepare_hint: str

    @property
    def num_tokens(self) -> int:
        return (self.img_size // self.patch_size) ** 2 + 1


DATASETS: dict[str, DatasetSpec] = {
    "cifar10": DatasetSpec(
        "cifar10", 10, 32, 4, (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616),
        DATA_ROOT, "cifar10", "uv run python prepare_data.py --dataset cifar10"),
    "cifar100": DatasetSpec(
        "cifar100", 100, 32, 4, (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762),
        DATA_ROOT, "cifar100", "uv run python prepare_data.py --dataset cifar100"),
    "tiny_imagenet": DatasetSpec(
        "tiny_imagenet", 200, 64, 8, IMAGENET_MEAN, IMAGENET_STD,
        DATA_ROOT / "tiny-imagenet-200", "imagefolder",
        "uv run python prepare_data.py --dataset tiny_imagenet"),
    "imagenet100": DatasetSpec(
        "imagenet100", 100, 224, 16, IMAGENET_MEAN, IMAGENET_STD,
        DATA_ROOT / "ImageNet", "imagefolder",
        "place ImageNet-100 as data/ImageNet/{train,val}/<wnid>/*.JPEG"),
    "imagenet1k": DatasetSpec(
        "imagenet1k", 1000, 224, 16, IMAGENET_MEAN, IMAGENET_STD,
        DATA_ROOT / "ImageNet1k", "imagefolder",
        "uv run python prepare_data.py --dataset imagenet1k  (see its --help)"),
}


def get_spec(name: str, root: str | Path | None = None) -> DatasetSpec:
    if name not in DATASETS:
        raise ValueError(f"unknown dataset: {name} (available: {list(DATASETS)})")
    spec = DATASETS[name]
    if root is not None:
        spec = DatasetSpec(**{**spec.__dict__, "root": Path(root)})
    return spec


def build_transform(spec: DatasetSpec, train: bool, augment: Augment = "auto",
                    re_prob: float = 0.25) -> transforms.Compose:
    normalize = [transforms.ToTensor(), transforms.Normalize(spec.mean, spec.std)]
    small = spec.img_size <= 64
    if not train or augment == "none":
        if small:
            return transforms.Compose(normalize)
        # assumption: standard Resize(img/0.875) -> CenterCrop evaluation
        return transforms.Compose([
            transforms.Resize(int(spec.img_size / 0.875), interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(spec.img_size),
            *normalize,
        ])

    if augment == "auto":
        augment = "small" if small else "paper"
    if augment == "paper":
        crop = transforms.RandomResizedCrop(spec.img_size, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3),
                                            interpolation=InterpolationMode.BILINEAR)
    elif augment == "small":
        crop = transforms.RandomCrop(spec.img_size, padding=spec.img_size // 8, padding_mode="reflect")
    else:
        raise ValueError(f"unknown augment profile: {augment}")
    return transforms.Compose([
        crop,
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        *normalize,
        transforms.RandomErasing(p=re_prob),
    ])


def build_dataset(spec: DatasetSpec, split: Literal["train", "val"], transform) -> Dataset:
    if spec.kind in ("cifar10", "cifar100"):
        cls = datasets.CIFAR10 if spec.kind == "cifar10" else datasets.CIFAR100
        try:
            return cls(str(spec.root), train=split == "train", download=False, transform=transform)
        except RuntimeError as error:  # torchvision: "Dataset not found or corrupted"
            raise FileNotFoundError(f"{spec.name} not found under {spec.root}. Prepare it with:\n"
                                    f"  {spec.prepare_hint}") from error

    directory = spec.root / split
    if not directory.is_dir():
        raise FileNotFoundError(f"{spec.name}: {directory} does not exist. Prepare it with:\n"
                                f"  {spec.prepare_hint}")
    return datasets.ImageFolder(directory, transform=transform)


def class_names(dataset: Dataset) -> list[str]:
    return list(dataset.classes)


def build_loaders(
    spec: DatasetSpec,
    *,
    batch_size: int,
    num_workers: int,
    augment: Augment = "auto",
    re_prob: float = 0.25,
    eval_batch_size: int | None = None,
    train_eval_transform: bool = False,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """(shuffled train loader, val loader, class names).

    train_eval_transform=True uses the deterministic eval transform for the train
    split too (calibration data for pruning importance)."""
    train_transform = (build_transform(spec, train=False) if train_eval_transform
                       else build_transform(spec, train=True, augment=augment, re_prob=re_prob))
    train_dataset = build_dataset(spec, "train", train_transform)
    val_dataset = build_dataset(spec, "val", build_transform(spec, train=False))
    if class_names(train_dataset) != class_names(val_dataset):
        raise ValueError(f"{spec.name}: train and val must have the same classes")

    kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size or batch_size, shuffle=False,
                            drop_last=False, **kwargs)
    return train_loader, val_loader, class_names(train_dataset)
