"""RecurrentViT: bViT (single-block recurrent ViT) and its generalizations.

Reference: Byra et al., "Investigating Single-Block Recurrence in Vision
Transformers for Image Recognition" (arXiv:2605.10661).

One class covers every configuration; only the loop schedule differs:
    bvit_b        = RecurrentViT(embed_dim=768, loop="0x12")      1 block x 12 steps, 8.6M params
    vit_b_scratch = RecurrentViT(embed_dim=768, loop="(0-11)x1")  12 blocks x 1 step, 86.6M params
Both have the same FLOPs and are trained with the same recipe from scratch;
"scratch" separates it from timm's pretrained vit_base_patch16_224 (LayerNorm,
IN-21k pretraining, different recipe).

What comes from the paper and what was assumed:
    paper:      pre-norm MHSA + FFN with residuals, RMSNorm, GELU, learnable CLS
                token and positional embedding, linear classifier on the final
                CLS token, steps = depth of the matching ViT, S/B/L widths,
                step embeddings added before the block (App. B, Eq. 8),
                register tokens (App. B)
    assumption: MLP ratio 4 (matches the paper's parameter counts), Linear bias,
                final RMSNorm before the head (as torchvision ViT), RMSNorm
                eps 1e-6, truncated-normal(0.02) init and zero head init,
                no dropout, separate q/k/v Linears (equivalent to a fused qkv;
                needed to analyze Q, K and V separately), per-step logits use
                the shared final norm + head, registers get no positional embedding
    extension:  general loop schedules (see schedule.py), LoopContext
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from timm.layers import trunc_normal_
from timm.models import register_model
from torch import nn

from models.bViT.schedule import LoopContext, LoopSchedule

# size -> (embed_dim, num_heads, depth of the matching ViT)
SIZES: dict[str, tuple[int, int, int]] = {
    "s": (384, 6, 12),
    "b": (768, 12, 12),
    "l": (1024, 16, 24),
}


@dataclass(frozen=True)
class RecurrentViTConfig:
    img_size: int = 224
    patch_size: int = 16
    in_chans: int = 3
    num_classes: int = 1000
    embed_dim: int = 768
    num_heads: int = 12
    mlp_ratio: float = 4.0       # assumption
    qkv_bias: bool = True        # assumption
    norm_eps: float = 1e-6       # assumption
    loop: str = "0x12"           # LoopSchedule spec
    use_step_embed: bool = False  # paper App. B (bViT-B-TE)
    num_registers: int = 0        # paper App. B (bViT-B-R uses 4)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"embed_dim {dim} is not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        # When True, the attention map is computed explicitly and kept in
        # last_attn (B, heads, N, N) for analysis (e.g. pointing game).
        self.record_attn = False
        self.last_attn: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q, k, v = (
            layer(x).view(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
            for layer in (self.q, self.k, self.v)
        )
        if self.record_attn:
            attn = (q @ k.transpose(-2, -1) * q.shape[-1] ** -0.5).softmax(dim=-1)
            self.last_attn = attn.detach()
            x = attn @ v
        else:
            x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    """Pre-norm transformer block: x + MHSA(norm(x)), then x + FFN(norm(x))."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, qkv_bias: bool, norm_eps: float) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=norm_eps)
        self.attn = Attention(dim, num_heads, qkv_bias)
        self.norm2 = nn.RMSNorm(dim, eps=norm_eps)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class RecurrentViT(nn.Module):
    def __init__(self, config: RecurrentViTConfig | None = None, **kwargs) -> None:
        super().__init__()
        if config is None:
            config = RecurrentViTConfig(**kwargs)
        elif kwargs:
            raise ValueError("Pass either config or keyword arguments, not both.")
        if config.img_size % config.patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size")

        self.config = config
        self.schedule = LoopSchedule.parse(config.loop)
        self.loop_ctx = LoopContext()
        self.num_classes = config.num_classes
        self.embed_dim = config.embed_dim
        self.num_prefix_tokens = 1 + config.num_registers
        num_patches = (config.img_size // config.patch_size) ** 2

        self.patch_embed = nn.Conv2d(config.in_chans, config.embed_dim, config.patch_size, config.patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, config.embed_dim))
        if config.num_registers > 0:
            self.reg_tokens = nn.Parameter(torch.zeros(1, config.num_registers, config.embed_dim))
        else:
            self.register_parameter("reg_tokens", None)
        self.blocks = nn.ModuleList(
            Block(config.embed_dim, config.num_heads, config.mlp_ratio, config.qkv_bias, config.norm_eps)
            for _ in range(self.schedule.num_blocks)
        )
        if config.use_step_embed:
            # Eq. 8: x^{t+1} = F(x^t + TE^t), one embedding per global step
            self.step_embed = nn.Parameter(torch.zeros(self.schedule.num_steps, config.embed_dim))
        else:
            self.register_parameter("step_embed", None)
        self.norm = nn.RMSNorm(config.embed_dim, eps=config.norm_eps)
        self.head = nn.Linear(config.embed_dim, config.num_classes) if config.num_classes > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self) -> None:
        for parameter in (self.cls_token, self.pos_embed, self.reg_tokens):
            if parameter is not None:
                trunc_normal_(parameter, std=0.02)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        if isinstance(self.head, nn.Linear):
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    @torch.jit.ignore
    def no_weight_decay(self) -> set[str]:
        return {"cls_token", "pos_embed", "reg_tokens", "step_embed"}

    def reset_classifier(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1) + self.pos_embed
        if self.reg_tokens is not None:
            x = torch.cat((x[:, :1], self.reg_tokens.expand(x.shape[0], -1, -1), x[:, 1:]), dim=1)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.norm(x[:, 0])
        return x if pre_logits else self.head(x)

    def forward_features(
        self,
        x: torch.Tensor,
        *,
        schedule: str | LoopSchedule | None = None,
        max_steps: int | None = None,
        return_all_steps: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run the loop. Returns (final tokens, per-step logits if return_all_steps)."""
        schedule = self.schedule if schedule is None else LoopSchedule.parse(schedule)
        if schedule.num_blocks > len(self.blocks):
            raise ValueError(f"schedule uses {schedule.num_blocks} blocks, model has {len(self.blocks)}")
        steps = schedule.unroll()[:max_steps]
        if self.step_embed is not None and len(steps) > len(self.step_embed):
            raise ValueError(f"step_embed has {len(self.step_embed)} steps, schedule needs {len(steps)}")

        x = self.embed(x)
        per_step_logits: list[torch.Tensor] = []
        try:
            for step in steps:
                self.loop_ctx.step = step
                if self.step_embed is not None:
                    x = x + self.step_embed[step.global_step]
                x = self.blocks[step.block_id](x)
                if return_all_steps:
                    per_step_logits.append(self.forward_head(x))
        finally:
            self.loop_ctx.step = None
        return x, per_step_logits

    def forward(
        self,
        x: torch.Tensor,
        *,
        schedule: str | LoopSchedule | None = None,
        max_steps: int | None = None,
        return_all_steps: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Logits after the last step, or the list of logits after every step
        (Fig. 2a: accuracy when inference is terminated at step t)."""
        x, per_step_logits = self.forward_features(
            x, schedule=schedule, max_steps=max_steps, return_all_steps=return_all_steps)
        return per_step_logits if return_all_steps else self.forward_head(x)


# * ===== Presets =====

# name -> (size, loop spec). Table 3 style splits can be given with --loop.
PRESETS: dict[str, tuple[str, str]] = {
    "bvit_s": ("s", "0x12"),
    "bvit_b": ("b", "0x12"),
    "bvit_l": ("l", "0x24"),
    "vit_s_scratch": ("s", "(0-11)x1"),
    "vit_b_scratch": ("b", "(0-11)x1"),
    "vit_l_scratch": ("l", "(0-23)x1"),
}


def build_recurrent_vit(name: str, **overrides) -> RecurrentViT:
    """Build a preset; any RecurrentViTConfig field (img_size, patch_size,
    num_classes, loop, use_step_embed, ...) can be overridden."""
    if name not in PRESETS:
        raise ValueError(f"unknown preset: {name} (available: {list(PRESETS)})")
    size, loop = PRESETS[name]
    embed_dim, num_heads, _ = SIZES[size]
    config = RecurrentViTConfig(embed_dim=embed_dim, num_heads=num_heads, loop=loop)
    return RecurrentViT(replace(config, **overrides))


def _timm_entry(name: str):
    def create(pretrained: bool = False, **kwargs) -> RecurrentViT:
        if pretrained:
            raise ValueError("no pretrained weights exist for RecurrentViT presets")
        for key in ("pretrained_cfg", "pretrained_cfg_overlay", "cache_dir"):
            kwargs.pop(key, None)
        return build_recurrent_vit(name, **kwargs)

    create.__name__ = name
    create.__qualname__ = name
    return register_model(create)


for _name in PRESETS:
    globals()[_name] = _timm_entry(_name)
