"""What happens when a multigrid loop ViT goes back from the coarse grid to the fine grid?

For every coarse -> fine move of a trained model this measures, on validation images,
    norm_fine:   mean L2 norm of the saved fine patch tokens
    norm_update: mean L2 norm of the update that the coarse grid sends up (prolong(c - c0))
    sibling_cos_before / after: mean cosine similarity between the 4 patch tokens that share
                 a coarse parent, before and after the update is added (1 = identical tokens)
and writes outputs/mgloop/transfer_diagnosis.json.

Usage:
    uv run python mgloop_diagnose.py w_cycle v_cycle coarse_to_fine w_cycle_deep
"""

from __future__ import annotations

import json
import sys

import torch
import torch.nn.functional as F

from dl_dataset.registry import build_loaders, get_spec
from mgloop_experiments import DATASET, EXPERIMENTS, OUTPUT_DIR
from train_bvit import load_model


def sibling_cosine(patches: torch.Tensor, size: int) -> float:
    """Mean pairwise cosine similarity inside every 2x2 group of a (B, size*size, C) grid."""
    B, _, C = patches.shape
    groups = patches.view(B, size // 2, 2, size // 2, 2, C).permute(0, 1, 3, 2, 4, 5).reshape(-1, 4, C)
    groups = F.normalize(groups.float(), dim=-1)
    similarity = groups @ groups.transpose(1, 2)
    return float((similarity.sum(dim=(1, 2)) - 4).mean() / 12)


@torch.inference_mode()
def diagnose(name: str, images: torch.Tensor) -> list[dict]:
    model, _ = load_model(EXPERIMENTS[name].run_dir / "last.pth", device=images.device)
    model.eval()
    prefix = model.num_prefix_tokens
    x, level, saved, records = model.embed(images), 0, [], []
    for step in model.schedule.unroll():
        target = model.step_levels[step.global_step]
        while level < target:
            coarse = model.restrict(x, level)
            saved.append((x, coarse))
            x, level = coarse, level + 1
        while level > target:
            fine, coarse_before = saved.pop()
            update = model.prolong(x - coarse_before, level - 1)
            size = model.grid_size >> (level - 1)
            x, level = fine + update, level - 1
            records.append({
                "before_step": step.global_step + 1,
                "norm_fine": float(fine[:, prefix:].float().norm(dim=-1).mean()),
                "norm_update": float(update[:, prefix:].float().norm(dim=-1).mean()),
                "sibling_cos_before": sibling_cosine(fine[:, prefix:], size),
                "sibling_cos_after": sibling_cosine(x[:, prefix:], size),
            })
        x = model.blocks[step.block_id](x)
    return records


def main() -> None:
    names = sys.argv[1:] or ["w_cycle", "v_cycle", "coarse_to_fine", "w_cycle_deep"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader, _ = build_loaders(get_spec(DATASET), batch_size=1000, num_workers=2)
    images = next(iter(val_loader))[0].to(device)
    results = {name: diagnose(name, images) for name in names}
    (OUTPUT_DIR / "transfer_diagnosis.json").write_text(json.dumps(results, indent=2))
    for name, records in results.items():
        print(f"== {name}")
        for r in records:
            print(f"  before step {r['before_step']:2d}: |fine|={r['norm_fine']:7.2f} |update|={r['norm_update']:7.2f} "
                  f"ratio={r['norm_update'] / r['norm_fine']:5.2f} sibling cos {r['sibling_cos_before']:.3f} -> "
                  f"{r['sibling_cos_after']:.3f}")


if __name__ == "__main__":
    main()
