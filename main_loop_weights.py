from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

from SReT import SReT_S
from analysis.activation import ActivationRecorder
from analysis.attention import plot_attention
from analysis.gradients import register_gradient_hooks
from analysis.hooks import register_activation_hooks, register_attention_hooks, register_mlp_activation_hooks, remove_hooks
from analysis.mlp_activation_analyzer import MLPActivationAnalyzer
from analysis.recorder import AnalysisRecorder
from analysis.weight_change import compute_weight_change


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze SReT loop-wise behavior")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num-loops", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--imagenet-root", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def build_loader(root: str | None, batch_size: int, num_workers: int):
    if root is None:
        return None
    from torchvision import datasets
    from torchvision import transforms as T

    root_path = Path(root)
    dataset_root = root_path / "train" if (root_path / "train").exists() else root_path

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(dataset_root, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    model = SReT_S(pretrained=False).to(device)
    model.num_loops = args.num_loops
    model.train()

    output_dir = Path(args.output_dir)
    recorder = AnalysisRecorder(output_dir=output_dir)
    activation_recorder = ActivationRecorder(output_dir=output_dir / "activation")
    mlp_analyzer = MLPActivationAnalyzer()

    handles = []
    handles.extend(register_activation_hooks(model, recorder, activation_recorder))
    handles.extend(register_attention_hooks(model, recorder))
    handles.extend(register_gradient_hooks(model, recorder))
    handles.extend(register_mlp_activation_hooks(model, mlp_analyzer))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    loader = build_loader(args.imagenet_root, args.batch_size, args.num_workers)
    if loader is None:
        raise ValueError("--imagenet-root is required for loop-wise analysis")

    weight_before = deepcopy(model.state_dict())

    for epoch in range(args.epochs):
        model.current_epoch = epoch
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            for loop in range(model.num_loops):
                model.current_loop = loop
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = loss_fn(logits, targets)
                loss.backward()
                optimizer.step()

            break

        recorder.save_epoch_gradients(epoch)
        activation_recorder.save(epoch)
        recorder.save_attentions()

    compute_weight_change(weight_before, model.state_dict(), recorder)
    recorder.save_weight_changes()

    torch.save(dict(mlp_analyzer.data), output_dir / "mlp_activation.pt")

    remove_hooks(handles)

    print(f"Saved analysis outputs to {output_dir}")


if __name__ == "__main__":
    main()
