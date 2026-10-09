"""Standalone architecture definition for the "2 shared blocks x 12 loops,
select-type" compressed ViT-Large -- the SAME architecture used to train
vit2_select_interleaved.pt (best_no_lora_layer1_13_interleaved.pt in the
original project, ~72.4% Top-1 on ImageNet-100).

This file intentionally contains ONLY what is needed to reconstruct the model
and load the checkpoint -- no dataset handling, no training loop, no LoRA
(this variant does not use LoRA). It has no dependency on any other file from
the original project.

How the model is built (for context):
    - Start from a standard, ImageNet-1K-pretrained ViT-Large
      (timm's "vit_large_patch16_224"), which has 24 transformer blocks.
    - Keep only block #1 (index 0) and block #13 (index 12), reusing their
      pretrained weights as-is (no averaging, no reinitialization).
    - Everything else (patch embedding, position embedding, final LayerNorm,
      classification head) stays exactly as in the original pretrained model.
    - At inference, these two blocks are applied in INTERLEAVED order, 12
      times each: block_A, block_B, block_A, block_B, ... (24 block
      applications total -- the same depth as the original 24-block model).
    - Both blocks' own weights were then fully fine-tuned (not frozen) on
      ImageNet-100; no LoRA adapters are used anywhere.

Requires: torch, torchvision, timm (pip install torch torchvision timm).
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn
from torchvision import transforms

MODEL_NAME = "vit_large_patch16_224"
RECURSION_LOOPS = 12
SELECTED_LAYER_INDICES = (0, 12)  # 0-indexed: original layer 1 and layer 13.


class RecursiveSharedBlocks(nn.Module):
    """Apply two shared transformer blocks in interleaved order, `loops` times
    each: block_A, block_B, block_A, block_B, ... (2 * loops applications
    total). Both blocks' weights are shared/reused across every repetition --
    there is no per-loop adapter of any kind."""

    def __init__(self, blocks: list[nn.Module], loops: int = RECURSION_LOOPS) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.loops = loops

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.loops):
            for block in self.blocks:
                x = block(x)
        return x


def build_model(pretrained: bool = True) -> nn.Module:
    """Build the (untrained) architecture. Call load_checkpoint() afterwards
    to load the actual fine-tuned weights -- `pretrained` only controls
    whether timm downloads its own ImageNet-1K weights first, which doesn't
    matter once a checkpoint is loaded, but keeps this function usable
    standalone (e.g. for inspecting the architecture) without a checkpoint."""
    import timm  # Imported lazily so `pip install timm` is only needed here.

    model = timm.create_model(MODEL_NAME, pretrained=pretrained)
    original_blocks = list(model.blocks)
    if len(original_blocks) != 24:
        raise ValueError(f"Expected a 24-block ViT-Large, found {len(original_blocks)} blocks.")

    selected_blocks = [original_blocks[index] for index in SELECTED_LAYER_INDICES]
    model.blocks = RecursiveSharedBlocks(selected_blocks, RECURSION_LOOPS)
    return model


def imagenet100_to_imagenet1k_indices(class_names: list[str]) -> list[int]:
    """The classification head was never replaced, so the model still outputs
    1000 ImageNet-1K logits. Map each ImageNet-100 synset name to its position
    in that 1000-way output, so those 100 columns can be sliced out."""
    from timm.data.imagenet_info import ImageNetInfo

    imagenet1k_synsets = ImageNetInfo("imagenet-1k").label_names()
    synset_to_index = {synset: index for index, synset in enumerate(imagenet1k_synsets)}
    missing = [name for name in class_names if name not in synset_to_index]
    if missing:
        raise ValueError(f"Classes are not ImageNet-1K synsets: {missing}")
    return [synset_to_index[name] for name in class_names]


def load_checkpoint(checkpoint_path: str, device: str | torch.device = "cpu") -> tuple[nn.Module, list[str], list[int]]:
    """Build the architecture and load a trained checkpoint (e.g.
    vit2_select_interleaved.pt) into it. Returns (model, class_names,
    class_indices):
        - class_names[i] is the ImageNet-100 synset name for restricted class i.
        - class_indices is the list of positions in the model's raw 1000-way
          output that correspond to class_names, in the same order -- the
          model's classification head is still the original 1000-class
          ImageNet-1K head (it was never replaced), so this restriction must
          be applied before interpreting predictions. See predict()."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # pretrained=False: the checkpoint's own weights overwrite everything
    # anyway, so there is no need to also download timm's ImageNet-1K weights.
    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()
    class_names = checkpoint["class_names"]
    class_indices = imagenet100_to_imagenet1k_indices(class_names)
    return model, class_names, class_indices


# Standard ImageNet-style preprocessing, matching what the model was
# evaluated with during training (Resize(256) -> CenterCrop(224)).
EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


def autocast_context(device: torch.device):
    """Same rule the original training/evaluation code used: fp16 mixed
    precision on CUDA, full precision everywhere else (CPU/MPS). Using this
    (rather than always running in plain fp32) matters because the trained
    weights were themselves validated under this exact numeric behavior --
    matching it here reproduces that inference path precisely, rather than a
    slightly different one that happens to also work."""
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


@torch.inference_mode()
def predict(
    model: nn.Module,
    image_tensor: torch.Tensor,
    class_names: list[str],
    class_indices: list[int],
    top_k: int = 5,
):
    """image_tensor: a single preprocessed image, shape (3, 224, 224) or
    (1, 3, 224, 224). class_names/class_indices: as returned by
    load_checkpoint(). Returns a list of (class_name, probability) tuples,
    the top_k most likely classes among the 100 ImageNet-100 classes."""
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    device = next(model.parameters()).device
    with autocast_context(device):
        logits = model(image_tensor.to(device))
    logits = logits[:, class_indices].float()  # Restrict 1000 -> 100 classes.
    probabilities = logits.softmax(dim=-1)[0]
    top_probabilities, top_indices = probabilities.topk(min(top_k, len(class_names)))
    return [
        (class_names[index], probability.item())
        for index, probability in zip(top_indices.tolist(), top_probabilities)
    ]
