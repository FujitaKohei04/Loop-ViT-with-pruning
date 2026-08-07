from itertools import islice
import json
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib-cache"))

import time

import timm
import torch

import SReT
from analysis.statistics import collect_weights
from main_measure import build_imagenet_val_loader, collect_activations, make_random_calibration_batches, parse_args, save_activation_outputs, save_weight_histograms, set_seed

def mkdir_time():
    t = time.time()
    path = f"./activation_weight/{t}"
    Path(path).mkdir(exist_ok=True)
    return path

def print_hypr(pr: dict, dir_path: Path = None):
    with open(Path(".") / str(dir_path) / "data.json", "w") as f:
        json.dump(pr, f, indent=4)

def main():
    DIR_PATH = mkdir_time()
    args = parse_args()
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


    # モデル作成
    if args.model_name_normal == "vit_base_patch16_224":
        model = timm.create_model(
            args.model_name_normal,
            pretrained=True,
            num_classes=args.num_classes,
        ).to(device)

    elif args.model_name_comp_loop == "sret_s":
        model = SReT.SReT_S(pretrained=False)
        model.load_state_dict(torch.load("./pre-trained/SReT_S.pth")["model"])
        model.to(device)

    # elif args.model_name_loop == "loop_vit":
    #     model = LoopViT.build_loop_vit_base_patch16_224(
    #         pretrained=False,
    #         use_iteration_embedding=False,
    #     ).to(device)

    else:
        raise ValueError("Unknown model")

    model_name = (
        args.model_name_normal
        or args.model_name_comp_loop
        or args.model_name_loop
    )

    print_hypr(
        {
            "dir path": DIR_PATH,
            "device": str(device),
            "model": model_name,
            "num classes": args.num_classes,
            "imagenet root": args.imagenet_root,
            "batch size": args.batch_size,
            "num workers": args.num_workers,
            "activation batches": args.activation_batches,
            "activation target": args.activation_target,
            "activation max elements per layer": args.activation_max_elements_per_layer
        },
        dir_path=DIR_PATH
    )

    # Weight保存
    weights = collect_weights(model)
    save_weight_histograms(weights, model_name, dir_path=DIR_PATH)

    # Activation保存
    if args.collect_activations:

        if args.imagenet_root is not None:
            loader = build_imagenet_val_loader(
                args.imagenet_root,
                model,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            batches = islice(loader, args.activation_batches)
        else:
            print("Error: imagenet is None")

        activations = collect_activations(
            model,
            batches,
            device,
            target=args.activation_target,
            max_elements_per_layer=args.activation_max_elements_per_layer,
        )

        save_activation_outputs(
            activations,
            model_name,
            save_tensors=args.save_activation_tensors,
            dir_path=DIR_PATH,
        )

        print(f"Saved activations: {len(activations)} layers")

if __name__ == "__main__":
    main()