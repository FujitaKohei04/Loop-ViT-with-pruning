from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import timm
import torch
from timm.data import create_transform, resolve_model_data_config
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import SReT
from imagenet_labels import remap_imagenet_targets
from main import evaluate_topk, set_seed
from main_measure import TaylorPruningConfig, apply_taylor_pruning, build_imagenet_loaders

Batch = tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class Config:
    image_net_root: str = './data/ImageNet'  # Path to the ImageNet dataset
    batch_size: int = 32  # Batch size for data loaders
    calibration_batches: int = 16  # Number of batches for calibration (used in Taylor pruning)
    taylor_prune_amount: float = 0.2  # Amount of pruning for Taylor pruning
    prune_bias: bool = False  # Whether to prune bias parameters
    eval_batches: int = 32  # Number of batches for evaluation
    model_name: str = "sret_s"

def print_config() -> None:
    print("=== Setting ===")
    print(f"batch size: {Config.batch_size}")
    print(f"calibration batches: {Config.calibration_batches}")
    print(f"prune bias: {Config.prune_bias}")
    print(f"eval batches: {Config.eval_batches}")
    print()

def plt_pru_acc(
        pruning_rate: list[float],
        accuracy: list[float],
        model_name: str,
        title: str,
) -> None:
    plt.plot(pruning_rate, accuracy, marker='o')
    plt.xlabel("pruning rate")
    plt.ylabel("accuracy")
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.grid(True)
    plt.title(title)
    plt.savefig(f"./prun_acc_figure/{model_name}/{title}")
    plt.show()

def run_model(
        pruning_rate: float,
        device: torch.device,
) -> tuple[float, float, float]:
    # * define the model
    if (Config.model_name == "sret_s"):
        model = SReT.SReT_S(pretrained=False)
        model.load_state_dict(torch.load('./pre-trained/SReT_S.pth')['model'])
        model.to(device)
    else:
        model = timm.create_model(
            Config.model_name,
            pretrained=True,
        ).to(device)

    # * make data loaders
    train_loader, val_loader = None, None
    if Config.image_net_root is not None:
        train_loader, val_loader, _ = build_imagenet_loaders(
            Config.image_net_root,
            model,
            batch_size = Config.batch_size,
            num_workers=1,
        )
        calibration_batches = islice(train_loader, Config.calibration_batches) #for Taylor pruning

    else:
        print("Please provide the path to the ImageNet dataset.")

    # * Taylor pruning
    pruning_stats = apply_taylor_pruning(
        model,
        calibration_batches,
        nn.CrossEntropyLoss(),
        TaylorPruningConfig(
            amount=pruning_rate,
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
    print(f"parameter: {pruning_stats['total_prunable_parameters']} -> {pruning_stats['total_prunable_parameters'] - pruning_stats['pruned_parameters']}")

    return pruning_stats['total_prunable_parameters'] - pruning_stats['pruned_parameters'], pruned_metrics['top1'], pruned_metrics['top5']
        
def compare_acc_by_each_pruning() -> None:
    set_seed(42)
    print_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pru_rate = []
    acc_top1 = []
    acc_top5 = []

    for rate in tqdm(np.arange(0.0, 1.0, 0.05)):
        temp_p, temp_t1, temp_t5= run_model(pruning_rate=rate, device=device)
        pru_rate.append(temp_p)
        acc_top1.append(temp_t1)
        acc_top5.append(temp_t5)

    np.save(
        f"./prun_acc_figure/{Config.model_name}/data.npy",
        np.array(
            [
                [pru_rate],
                [acc_top1],
                [acc_top5],
            ]
        ),
    )
    
    plt_pru_acc(pruning_rate=pru_rate, accuracy=acc_top1, model_name=Config.model_name, title="top1")
    plt_pru_acc(pruning_rate=pru_rate, accuracy=acc_top5, model_name=Config.model_name, title="top5")

if __name__ == "__main__":
    compare_acc_by_each_pruning()