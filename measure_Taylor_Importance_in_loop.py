import pickle
from dataclasses import asdict, dataclass
from itertools import islice
import os
from pathlib import Path
import time

import timm
import torch
from torch import nn

from SReT import SReT
from compare_acc_by_each_pruning import print_config
import cp_pre_w
from dl_dataset.dl_cifar100 import build_cifar_loaders
from main_measure import build_imagenet_loaders, collect_taylor_scores, set_seed
from models_activation_weights import mkdir_time

@dataclass(frozen=True)
class Config:
    dataset: str = 'imagenet100' #imagenet100, cifar100
    image_net_root: str = './data/ImageNet'  # Path to the ImageNet dataset
    batch_size: int = 32  # Batch size for data loaders
    calibration_batches: int = 16  # Number of batches for calibration (used in Taylor pruning)
    taylor_prune_amount: float = 0.2  # Amount of pruning for Taylor pruning
    prune_bias: bool = False  # Whether to prune bias parameters
    eval_batches: int = 64  # Number of batches for evaluation
    model_name: str = "sret_s_no_loop" 
    # sret_s, sret_lt, sret_t
    # sret_s_no_loop
    # vit_base_patch16_224, vit_small_patch16_224, vit_tiny_patch16_224

def mkdir_time():
    t = time.time()
    path = f"./TaylorImportanceInLoop/{t}"
    Path(path).mkdir(exist_ok=True)
    return path

def main() -> None:
    DIR_PATH = mkdir_time()
    set_seed(42)
    print_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # * define the model
    if (Config.model_name == "sret_s"):
        model = SReT.SReT_S(pretrained=False)
        model.load_state_dict(torch.load('./pre-trained/SReT_S.pth')['model'])
        model.to(device)
    elif (Config.model_name == "sret_t"):
        model = SReT.SReT_T(pretrained=False)
        model.load_state_dict(torch.load('./pre-trained/SReT_T.pth')['model'])
        model.to(device)
    elif (Config.model_name == "sret_lt"):
        model = SReT.SReT_LT(pretrained=False)
        model.load_state_dict(torch.load('./pre-trained/SReT_LT_8-4-1_2-1-1.pth', weights_only=False)['model'])
        model.to(device)
    elif (Config.model_name == "sret_s_no_loop"):
        model = cp_pre_w.load_SReT_S_no_loop()
        model.to(device)
    elif (Config.model_name == "vit_base_patch16_224" 
            or Config.model_name == "vit_small_patch16_224"
            or Config.model_name == "vit_tiny_patch16_224"):
        model = timm.create_model(
            Config.model_name,
            pretrained=True,
        ).to(device)
    elif (Config.model_name == "loop_vit_large_lora"):
        model = None
    else:
        print("the model name is not found.")

    # * make data loaders
    train_loader, val_loader = None, None
    if Config.dataset == "imagenet100" and Config.image_net_root is not None:
        train_loader, val_loader = build_imagenet_loaders(
            Config.image_net_root,
            model,
            batch_size = Config.batch_size,
            num_workers=1,
        )
        calibration_batches = islice(train_loader, Config.calibration_batches) #for Taylor pruning
    elif Config.dataset == "cifar100":
        train_loader, val_loader = build_cifar_loaders(
            model,
            batch_size = Config.batch_size,
            num_workers=1,
        )
        calibration_batches = islice(train_loader, Config.calibration_batches) #for Taylor pruning
    else:
        print("Please provide the path to the ImageNet dataset.")


    scores = collect_taylor_scores(
        model,
        calibration_batches,
        nn.CrossEntropyLoss(),
        device,
        prune_bias=Config.prune_bias,
    )


    # make dir by "model_name"
    os.makedirs(f"./{DIR_PATH}/{Config.model_name}", exist_ok=True)

    with open(f"./{DIR_PATH}/{Config.model_name}/data.pkl", "wb") as f:
        pickle.dump(
            {
                "TaylorImportance": scores
            },
            f
        )

    # save the hyper parameter
    with open(f"./prun_acc_figure/{Config.model_name}/hyper_parameter.txt", "w") as o:
        print(asdict(Config()), file=o)

if __name__ == "__main__":
    main()