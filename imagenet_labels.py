from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from timm.data.imagenet_info import ImageNetInfo


class ImageFolderLike(Protocol):
    classes: list[str]
    target_transform: object


def build_imagenet_target_map(classes: Sequence[str]) -> dict[int, int]:
    """Map ImageFolder class indices to ImageNet-1k classifier indices."""

    wnid_to_imagenet = {
        wnid: imagenet_index
        for imagenet_index, wnid in enumerate(ImageNetInfo("imagenet-1k").label_names())
    }
    missing_wnids = [wnid for wnid in classes if wnid not in wnid_to_imagenet]
    if missing_wnids:
        preview = ", ".join(missing_wnids[:5])
        raise ValueError(
            "ImageFolder classes must be ImageNet WNIDs. "
            f"Missing from ImageNet-1k mapping: {preview}"
        )

    return {
        imagefolder_index: wnid_to_imagenet[wnid]
        for imagefolder_index, wnid in enumerate(classes)
    }


def remap_imagenet_targets(dataset: ImageFolderLike) -> dict[int, int]:
    """Make an ImageFolder dataset return ImageNet-1k target indices."""

    target_map = build_imagenet_target_map(dataset.classes)
    dataset.target_transform = target_map.__getitem__
    return target_map
