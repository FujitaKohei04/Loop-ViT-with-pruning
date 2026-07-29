from itertools import islice

from timm.models import dataclass
import torch
from torch import nn

import SReT
from main import set_seed
from main_measure import apply_taylor_pruning

@dataclass(frozen=True)
class Config:
    image_net_root: str = './data/imagenet'  # Path to the ImageNet dataset
    batch_size: int = 32  # Batch size for data loaders
    calibration_batches: int = 16  # Number of batches for calibration (used in Taylor pruning)
    taylor_prune_amount: float = 0.2  # Amount of pruning for Taylor pruning
    prune_bias: bool = False  # Whether to prune bias parameters
    eval_batches: int = 32  # Number of batches for evaluation

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
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader: DataLoader[Batch] = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    test_loader: DataLoader[Batch] = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    return train_loader, val_loader, test_loader

def compare_acc_by_each_pruning() -> None:
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #define the model
    model = SReT.SReT_X(pretrained=False)
    model.load_state_dict(torch.load('./pre-trained/SReT_S.pth')['model'])
    model.to(device)

    #make data loaders
    train_loader, val_loader, test_loader = None, None, None
    if Config.image_net_root is not None:
        train_loader, val_loader, test_loader = build_imagenet_loaders(
            Config.image_net_root,
            model,
            batch_size = Config.batch_size,
        )
        calibration_batches = islice(train_loader, Config.calibration_batches) #for Taylor pruning

    else:
        print("Please provide the path to the ImageNet dataset.")

    #Taylor pruning
    pruning_stats = apply_taylor_pruning(
        model,
        calibration_batches,
        nn.CrossEntropyLoss(),
        TaylorPruningConfig(
            amount=Config.taylor_prune_amount,
            prune_bias=Config.prune_bias,
        ),
        device,
    )

    pruned_metrics = evaluate_topk(
        model,
        val_loader,
        device,
        max_batches=Config.eval_batches,
    )
    
    print(f"ImageNet val top-1: {pruned_metrics['top1']:.2%}")
    print(f"ImageNet val top-5: {pruned_metrics['top5']:.2%}")


if __name__ == "__main__":
    compare_acc_by_each_pruning()