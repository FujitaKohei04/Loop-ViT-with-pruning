"""Statistics of step-conditioned importance and masks (paper Fig. 4, 5, 6 bottom, 15)."""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.stats import spearmanr


def step_bins(num_uses: int, num_bins: int = 3) -> list[tuple[int, int]]:
    """Split 1..num_uses into equal ranges: 12 -> [(1, 4), (5, 8), (9, 12)]."""
    chunks = np.array_split(np.arange(1, num_uses + 1), min(num_bins, num_uses))
    return [(int(chunk[0]), int(chunk[-1])) for chunk in chunks]


def persistence(masks: torch.Tensor, num_bins: int = 3) -> dict[str, float]:
    """Fraction of selected units (kept in >= 1 step) that are kept in 1-4 / 5-8 / 9-12 steps (Fig. 6 bottom)."""
    counts = masks.sum(dim=0)
    counts = counts[counts > 0]
    result = {}
    for low, high in step_bins(masks.shape[0], num_bins):
        share = ((counts >= low) & (counts <= high)).float().mean().item() if counts.numel() else float("nan")
        result[f"steps_{low}-{high}"] = share
    return result


def chance_persistence(num_uses: int, keep_fraction: float, num_bins: int = 3) -> dict[str, float]:
    """Fig. 15: every step keeps a unit independently with probability keep_fraction;
    share of 1-4 / 5-8 / 9-12 among units kept at least once."""
    p = keep_fraction
    probabilities = [math.comb(num_uses, k) * p**k * (1 - p) ** (num_uses - k) for k in range(num_uses + 1)]
    selected = 1 - probabilities[0]
    return {f"steps_{low}-{high}": (sum(probabilities[low:high + 1]) / selected if selected > 0 else float("nan"))
            for low, high in step_bins(num_uses, num_bins)}


def step_similarity(scores: torch.Tensor) -> np.ndarray:
    """Spearman correlation between the importance profiles of every pair of steps (Fig. 4).
    Steps without gradient give NaN (gray entries in the paper)."""
    num_uses = scores.shape[0]
    matrix = np.full((num_uses, num_uses), np.nan)
    for i in range(num_uses):
        for j in range(num_uses):
            matrix[i, j] = _spearman(scores[i], scores[j])
    return matrix


def cls_patch_similarity(cls_scores: torch.Tensor, patch_scores: torch.Tensor) -> list[float]:
    """Spearman correlation between CLS and patch profiles at each step (Fig. 5)."""
    return [_spearman(cls_scores[u], patch_scores[u]) for u in range(cls_scores.shape[0])]


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.numpy(), b.numpy()
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)
