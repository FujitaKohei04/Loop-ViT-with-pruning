"""Token reduction and step reduction for RecurrentViT (no retraining, no new parameters).

These do not remove weights: they shorten the token sequence or the loop, so
they cut the attention cost as well as the Linear cost.

    ToMe   Bolya et al., "Token Merging: Your ViT But Faster", ICLR 2023.
           After the attention of every step, the r most similar token pairs
           (bipartite matching on the mean key vectors) are averaged. Merged
           tokens carry their size, which is added to the attention logits as
           log(size) (proportional attention).
    EViT   Liang et al., "Not All Patches are What You Need: Expediting Vision
           Transformers via Token Reorganizations", ICLR 2022.
           At 1/4, 1/2 and 3/4 of the depth, the patch tokens with the highest
           CLS attention are kept (keep_rate) and the others are fused into one
           token (CLS-attention weighted average).
    steps  Run only the first `max_steps` loop steps (layer dropping / early exit;
           the model already supports it).

[A] ToMe / EViT are applied to a model trained without them (both papers report
this off-the-shelf setting). Register tokens are not supported.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from models.bViT.bViT import RecurrentViT


class StepLimited(nn.Module):
    """model(x, max_steps=n) as a module, so that profilers can call model(x)."""

    def __init__(self, model: RecurrentViT, max_steps: int) -> None:
        super().__init__()
        self.model = model
        self.max_steps = max_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x, max_steps=self.max_steps)


class TokenReducedViT(nn.Module):
    """Runs a RecurrentViT with ToMe (r > 0) or EViT (keep_rate < 1) token reduction."""

    def __init__(self, model: RecurrentViT, *, tome_r: int = 0, evit_keep_rate: float = 1.0,
                 evit_steps: tuple[int, ...] | None = None) -> None:
        super().__init__()
        if model.step_levels or model.reg_tokens is not None:
            raise ValueError("token reduction supports neither multigrid levels nor register tokens")
        if tome_r > 0 and evit_keep_rate < 1.0:
            raise ValueError("use either ToMe or EViT")
        self.model = model
        self.tome_r = tome_r
        self.evit_keep_rate = evit_keep_rate
        num_steps = model.schedule.num_steps
        # EViT: layers 4, 7, 10 of 12 (0-based 3, 6, 9)
        self.evit_steps = evit_steps if evit_steps is not None else tuple(
            dict.fromkeys(num_steps * i // 4 for i in (1, 2, 3)))
        self.token_counts: list[tuple[int, int]] = []  # (tokens in attention, tokens in FFN) of the last forward

    def token_reduction(self) -> float:
        """1 - mean token count over all attention / FFN calls relative to the input (last forward)."""
        if not self.token_counts:
            raise RuntimeError("run a forward pass first")
        full = self.token_counts[0][0]
        return 1 - sum(a + b for a, b in self.token_counts) / (2 * len(self.token_counts) * full)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        model = self.model
        x = model.embed(x)
        size: torch.Tensor | None = None  # (B, N, 1) number of patches merged into each token (ToMe)
        self.token_counts = []
        try:
            for step in model.schedule.unroll():
                model.loop_ctx.step = step
                if model.step_embed is not None:
                    x = x + model.step_embed[step.global_step]
                block = model.blocks[step.block_id]
                tokens_in = x.shape[1]
                prune_here = self.evit_keep_rate < 1.0 and step.global_step in self.evit_steps
                update, keys, cls_attn = _attention(block.attn, block.norm1(x), size, need_cls_attn=prune_here)
                x = x + update
                if self.tome_r > 0:
                    x, size = _tome_merge(x, size, keys, self.tome_r)
                elif prune_here:
                    x = _evit_reorganize(x, cls_attn, self.evit_keep_rate)
                x = x + block.mlp(block.norm2(x))
                self.token_counts.append((tokens_in, x.shape[1]))
        finally:
            model.loop_ctx.step = None
        return model.forward_head(x)


def _attention(attn: nn.Module, x: torch.Tensor, size: torch.Tensor | None, need_cls_attn: bool
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Attention output, mean key per token (B, N, head_dim), CLS attention (B, N - 1) or None."""
    B, N, C = x.shape
    q, k, v = (layer(x).view(B, N, attn.num_heads, C // attn.num_heads).transpose(1, 2)
               for layer in (attn.q, attn.k, attn.v))
    bias = None if size is None else size.log().to(q.dtype).view(B, 1, 1, N)  # proportional attention
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
    cls_attn = None
    if need_cls_attn:  # only the CLS row is needed: (B, heads, N) -> mean over heads
        logits = (q[:, :, :1] @ k.transpose(-2, -1)).squeeze(2).float() * q.shape[-1] ** -0.5
        cls_attn = logits.softmax(dim=-1).mean(dim=1)[:, 1:]
    return attn.proj(out.transpose(1, 2).reshape(B, N, C)), k.mean(dim=1), cls_attn


def _tome_merge(x: torch.Tensor, size: torch.Tensor | None, metric: torch.Tensor, r: int
                ) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Bipartite soft matching (ToMe Sec. 3): tokens alternate between sets A and B, every A
    token proposes its most similar B token, the r best pairs are merged (size-weighted mean).
    The CLS token (index 0, in A) is never merged."""
    B, N, C = x.shape
    r = min(r, (N - 1) // 2)
    if r <= 0:
        return x, size
    if size is None:
        size = x.new_ones(B, N, 1)
    with torch.no_grad():
        metric = F.normalize(metric.float(), dim=-1)
        a, b = metric[:, ::2], metric[:, 1::2]
        scores = a @ b.transpose(-1, -2)
        scores[:, 0] = -math.inf  # CLS
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        unmerged_idx = edge_idx[:, r:].sort(dim=1).values  # sorted: CLS stays first
        src_idx = edge_idx[:, :r]
        dst_idx = node_idx[..., None].gather(dim=1, index=src_idx)

    def merge(t: torch.Tensor) -> torch.Tensor:  # sum of merged tokens
        src, dst = t[:, ::2], t[:, 1::2]
        channels = t.shape[-1]
        unmerged = src.gather(dim=1, index=unmerged_idx.expand(-1, -1, channels))
        moved = src.gather(dim=1, index=src_idx.expand(-1, -1, channels))
        dst = dst.scatter_reduce(1, dst_idx.expand(-1, -1, channels), moved, reduce="sum")
        return torch.cat((unmerged, dst), dim=1)

    x = merge(x * size)
    size = merge(size)
    return x / size, size


def _evit_reorganize(x: torch.Tensor, cls_attn: torch.Tensor, keep_rate: float) -> torch.Tensor:
    """Keep the top keep_rate patch tokens by CLS attention; fuse the rest into one token."""
    B, N, C = x.shape
    keep = math.ceil(keep_rate * (N - 1))
    if keep >= N - 1:
        return x
    order = cls_attn.argsort(dim=1, descending=True)
    keep_idx, drop_idx = order[:, :keep], order[:, keep:]
    patches = x[:, 1:]
    kept = patches.gather(1, keep_idx[..., None].expand(-1, -1, C))
    dropped = patches.gather(1, drop_idx[..., None].expand(-1, -1, C))
    weight = cls_attn.gather(1, drop_idx)[..., None].to(x.dtype)
    fused = (dropped * weight).sum(dim=1, keepdim=True) / weight.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return torch.cat((x[:, :1], kept, fused), dim=1)
