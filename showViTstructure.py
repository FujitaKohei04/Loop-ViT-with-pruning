import argparse
import timm
import torch
import SReT
import LoopViT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run timm ViT with optional Taylor pruning.")
    parser.add_argument("--model-name", default="vit_base_patch16_224")
    return parser.parse_args()

args = parse_args()
if(args.model_name == "vit"):
    model = timm.create_model(
        "vit_base_patch16_224"
    )
elif(args.model_name == "sret_s"):
    model = SReT.SReT_S(pretrained=False)
    model.load_state_dict(torch.load('./pre-trained/SReT_S.pth')['model'])
elif(args.model_name == "loop_vit"):
    model = LoopViT.build_loop_vit_base_patch16_224(
        pretrained=True
    )

print(model)