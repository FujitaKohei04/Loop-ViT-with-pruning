from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import timm
import torch
from timm.layers import trunc_normal_
from timm.models import register_model
from timm.models.vision_transformer import global_pool_nlc
from torch import nn

FinetuneMode = Literal["head", "head-norm", "loop", "all"]


@dataclass(frozen=True)
class LoopViTConfig:
    model_name: str = "vit_base_patch16_224"
    pretrained: bool = True
    num_classes: int = 1000
    selected_block_indices: tuple[int, int, int] = (0, 5, 11)
    loop_count: int = 4
    use_iteration_embedding: bool = True
    iteration_embedding_on_cls_token: bool = True
    global_pool: str | None = None


class LoopViT(nn.Module):
    """Minimal looped ViT built from selected timm ViT blocks.

    The default configuration starts from timm's vit_base_patch16_224, keeps
    blocks 1, 6, and 12 (zero-based indices 0, 5, 11), and applies those three
    blocks four times. Parameters are shared across iterations.
    """

    def __init__(self, config: LoopViTConfig | None = None, **kwargs):
        super().__init__()
        if config is None:
            config = LoopViTConfig(**kwargs)
        elif kwargs:
            raise ValueError("Pass either config or keyword arguments, not both.")

        if config.loop_count < 1:
            raise ValueError("loop_count must be >= 1")
        if not config.selected_block_indices:
            raise ValueError("selected_block_indices must not be empty")

        base = timm.create_model(
            config.model_name,
            pretrained=config.pretrained,
            num_classes=config.num_classes,
        )
        num_blocks = len(base.blocks)
        for block_index in config.selected_block_indices:
            if block_index < 0 or block_index >= num_blocks:
                raise ValueError(
                    f"block index {block_index} is out of range for "
                    f"{config.model_name} with {num_blocks} blocks"
                )

        self.config = config
        self.model_name = config.model_name
        self.num_classes = config.num_classes
        self.embed_dim = base.embed_dim
        self.num_features = base.num_features
        self.num_prefix_tokens = getattr(base, "num_prefix_tokens", 1)
        self.global_pool = config.global_pool or getattr(base, "global_pool", "token")
        self.pool_include_prefix = getattr(base, "pool_include_prefix", False)
        self.selected_block_indices = tuple(config.selected_block_indices)
        self.loop_count = config.loop_count
        self.use_iteration_embedding = config.use_iteration_embedding
        self.iteration_embedding_on_cls_token = config.iteration_embedding_on_cls_token

        self.patch_embed = base.patch_embed
        self.cls_token = base.cls_token
        self.reg_token = getattr(base, "reg_token", None)
        self.pos_embed = base.pos_embed
        self.pos_drop = base.pos_drop
        self.patch_drop = base.patch_drop
        self.norm_pre = base.norm_pre
        self.blocks = nn.ModuleList([base.blocks[i] for i in self.selected_block_indices])
        self.norm = base.norm
        self.fc_norm = base.fc_norm
        self.head_drop = base.head_drop
        self.head = base.head

        if self.use_iteration_embedding:
            self.iteration_embed = nn.Parameter(
                torch.zeros(config.loop_count, 1, 1, self.embed_dim)
            )
            trunc_normal_(self.iteration_embed, std=0.02)
        else:
            self.register_parameter("iteration_embed", None)

    @torch.jit.ignore
    def no_weight_decay(self) -> set[str]:
        names = {"pos_embed", "cls_token", "reg_token"}
        if self.use_iteration_embedding:
            names.add("iteration_embed")
        return names

    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool: str | None = None) -> None:
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        if self.cls_token is not None:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            if self.reg_token is not None:
                reg_token = self.reg_token.expand(x.shape[0], -1, -1)
                x = torch.cat((cls_token, reg_token, x), dim=1)
            else:
                x = torch.cat((cls_token, x), dim=1)

        x = x + self.pos_embed
        return self.pos_drop(x)

    def _add_iteration_embedding(
        self,
        x: torch.Tensor,
        iteration_index: int,
    ) -> torch.Tensor:
        if self.iteration_embed is None:
            return x

        emb = self.iteration_embed[iteration_index]
        if self.iteration_embedding_on_cls_token:
            return x + emb

        x = x.clone()
        x[:, self.num_prefix_tokens:] = x[:, self.num_prefix_tokens:] + emb
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        for iteration_index in range(self.loop_count):
            x = self._add_iteration_embedding(x, iteration_index)
            for block in self.blocks:
                x = block(x)

        x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = global_pool_nlc(
            x,
            pool_type=self.global_pool,
            num_prefix_tokens=self.num_prefix_tokens,
            reduce_include_prefix=self.pool_include_prefix,
        )
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.forward_head(x)

    def set_trainable(
        self,
        *,
        patch_embed: bool,
        loop_blocks: bool,
        norm: bool,
        head: bool,
        iteration_embedding: bool | None = None,
    ) -> None:
        if iteration_embedding is None:
            iteration_embedding = self.use_iteration_embedding

        for parameter in self.parameters():
            parameter.requires_grad = False

        self._set_module_trainable(self.patch_embed, patch_embed)
        self._set_module_trainable(self.blocks, loop_blocks)
        self._set_module_trainable(self.norm, norm)
        self._set_module_trainable(self.fc_norm, norm)
        self._set_module_trainable(self.head, head)

        if self.iteration_embed is not None:
            self.iteration_embed.requires_grad = iteration_embedding

    def configure_finetuning(self, mode: FinetuneMode = "head-norm") -> None:
        """Set trainable parameters for common fine-tuning experiments."""

        if mode == "head":
            self.set_trainable(
                patch_embed=False,
                loop_blocks=False,
                norm=False,
                head=True,
                iteration_embedding=True,
            )
        elif mode == "head-norm":
            self.set_trainable(
                patch_embed=False,
                loop_blocks=False,
                norm=True,
                head=True,
                iteration_embedding=True,
            )
        elif mode == "loop":
            self.set_trainable(
                patch_embed=False,
                loop_blocks=True,
                norm=True,
                head=True,
                iteration_embedding=True,
            )
        elif mode == "all":
            for parameter in self.parameters():
                parameter.requires_grad = True
        else:
            raise ValueError(f"Unknown fine-tuning mode: {mode}")

    def get_finetuning_parameter_groups(
        self,
        *,
        base_lr: float,
        head_lr: float | None = None,
        iteration_lr: float | None = None,
        weight_decay: float = 0.05,
    ) -> list[dict[str, object]]:
        """Return optimizer parameter groups for currently trainable params."""

        head_lr = base_lr if head_lr is None else head_lr
        iteration_lr = base_lr if iteration_lr is None else iteration_lr

        groups: list[dict[str, object]] = []
        used: set[int] = set()

        self._append_trainable_group(
            groups,
            self.head.parameters(),
            used,
            lr=head_lr,
            weight_decay=weight_decay,
            name="head",
        )

        if self.iteration_embed is not None:
            self._append_trainable_group(
                groups,
                [self.iteration_embed],
                used,
                lr=iteration_lr,
                weight_decay=0.0,
                name="iteration_embed",
            )

        remaining = [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad and id(parameter) not in used
        ]
        self._append_trainable_group(
            groups,
            remaining,
            used,
            lr=base_lr,
            weight_decay=weight_decay,
            name="backbone",
        )
        return groups

    @staticmethod
    def _set_module_trainable(module: nn.Module, trainable: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad = trainable

    @staticmethod
    def _append_trainable_group(
        groups: list[dict[str, object]],
        parameters: Iterable[nn.Parameter],
        used: set[int],
        *,
        lr: float,
        weight_decay: float,
        name: str,
    ) -> None:
        params = [
            parameter
            for parameter in parameters
            if parameter.requires_grad and id(parameter) not in used
        ]
        if not params:
            return
        used.update(id(parameter) for parameter in params)
        groups.append(
            {
                "params": params,
                "lr": lr,
                "weight_decay": weight_decay,
                "name": name,
            }
        )


def build_loop_vit_base_patch16_224(
    *,
    pretrained: bool = True,
    num_classes: int = 1000,
    use_iteration_embedding: bool = True,
    loop_count: int = 4,
    selected_block_indices: Sequence[int] = (0, 5, 11),
    finetune_mode: FinetuneMode | None = None,
    **kwargs,
) -> LoopViT:
    kwargs.pop("pretrained_cfg", None)
    kwargs.pop("pretrained_cfg_overlay", None)
    kwargs.pop("cache_dir", None)
    config = LoopViTConfig(
        model_name="vit_base_patch16_224",
        pretrained=pretrained,
        num_classes=num_classes,
        use_iteration_embedding=use_iteration_embedding,
        loop_count=loop_count,
        selected_block_indices=tuple(selected_block_indices),
        **kwargs,
    )
    model = LoopViT(config)
    if finetune_mode is not None:
        model.configure_finetuning(finetune_mode)
    return model


@register_model
def loop_vit_base_patch16_224(pretrained: bool = False, **kwargs) -> LoopViT:
    return build_loop_vit_base_patch16_224(pretrained=pretrained, **kwargs)
