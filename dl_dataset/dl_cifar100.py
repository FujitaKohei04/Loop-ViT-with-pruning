from torch import nn
import torch
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from timm.data import create_transform, resolve_model_data_config

from fine_tuning import Batch

def build_cifar_loaders(
        model: nn.Module,
        *,
        batch_size: int,
        num_workers: int,
) -> tuple[DataLoader[Batch], DataLoader[Batch]]:
    # data setting for model
    data_config = resolve_model_data_config(model)
    train_transform = create_transform(**data_config, is_training=True)
    val_transform = create_transform(**data_config, is_training=False)

    # load dataset
    train_dataset = CIFAR100(root='./data', train=True,  download=True, transform=train_transform)
    val_dataset   = CIFAR100(root='./data', train=False, download=True, transform=val_transform)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }

    # make dataloader
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
    return train_loader, val_loader

