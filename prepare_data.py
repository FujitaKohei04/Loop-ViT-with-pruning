"""Download / reorganize datasets for train_bvit.py. Downloads happen ONLY here.

    uv run python prepare_data.py --dataset cifar100        # ~160 MB, a few minutes
    uv run python prepare_data.py --dataset cifar10         # ~160 MB, a few minutes
    uv run python prepare_data.py --dataset tiny_imagenet   # ~240 MB zip, ~10 min incl. unzip
    uv run python prepare_data.py --dataset imagenet1k      # no download, extracts local tars

ImageNet-1k cannot be downloaded automatically (registration required).
Get these files from https://image-net.org (ILSVRC2012) or Kaggle
"ImageNet Object Localization Challenge" and put them in data/ImageNet1k/:
    ILSVRC2012_img_train.tar     (~138 GB)
    ILSVRC2012_img_val.tar       (~6.3 GB)
    ILSVRC2012_devkit_t12.tar.gz (~2.5 MB, needed to label val images)
This script then extracts them into data/ImageNet1k/{train,val}/<wnid>/ using
torchvision.datasets.ImageNet (takes 1-2 hours).
"""

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

from torchvision import datasets

from dl_dataset.registry import get_spec

TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def prepare_cifar(name: str, root: Path) -> None:
    cls = datasets.CIFAR10 if name == "cifar10" else datasets.CIFAR100
    for train in (True, False):
        cls(str(root), train=train, download=True)


def prepare_tiny_imagenet(root: Path, download: bool) -> None:
    """Download + unzip, then move val/images/*.JPEG into val/<wnid>/ (ImageFolder layout)."""
    data_root = root.parent
    zip_path = data_root / "tiny-imagenet-200.zip"
    if not root.is_dir():
        if not zip_path.exists():
            if not download:
                raise FileNotFoundError(f"{zip_path} not found (run without --no-download)")
            print(f"downloading {TINY_IMAGENET_URL} -> {zip_path}")
            urllib.request.urlretrieve(TINY_IMAGENET_URL, zip_path)
        print(f"extracting {zip_path}")
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(data_root)

    val_dir = root / "val"
    annotations = val_dir / "val_annotations.txt"
    if not annotations.exists():
        print("val is already in ImageFolder layout")
        return
    for line in annotations.read_text().splitlines():
        file_name, wnid = line.split("\t")[:2]
        (val_dir / wnid).mkdir(exist_ok=True)
        source = val_dir / "images" / file_name
        if source.exists():
            shutil.move(source, val_dir / wnid / file_name)
    (val_dir / "images").rmdir()
    annotations.rename(root / "val_annotations.txt")
    # test/ has no labels; val is used for evaluation.
    print(f"tiny_imagenet ready: {root}")


def prepare_imagenet1k(root: Path) -> None:
    required = ["ILSVRC2012_img_train.tar", "ILSVRC2012_img_val.tar", "ILSVRC2012_devkit_t12.tar.gz"]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"put {missing} in {root} first (see this script's docstring)")
    for split in ("val", "train"):
        print(f"extracting {split} (this takes a while)")
        datasets.ImageNet(str(root), split=split)
    print(f"imagenet1k ready: {root}/{{train,val}}  (the tars can be deleted)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=["cifar10", "cifar100", "tiny_imagenet", "imagenet1k"])
    parser.add_argument("--root", default=None, help="default: the dataset's root in dl_dataset/registry.py")
    parser.add_argument("--no-download", action="store_true", help="tiny_imagenet: only reorganize an existing zip")
    args = parser.parse_args()

    root = get_spec(args.dataset, args.root).root
    root.mkdir(parents=True, exist_ok=True)
    if args.dataset in ("cifar10", "cifar100"):
        prepare_cifar(args.dataset, root)
    elif args.dataset == "tiny_imagenet":
        prepare_tiny_imagenet(root, download=not args.no_download)
    else:
        prepare_imagenet1k(root)


if __name__ == "__main__":
    main()
