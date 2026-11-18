"""Step-conditioned channel pruning of a looped ViT block: units that shrink the layers themselves.

Three kinds of units leave the residual stream (width `dim`) untouched, so removing them needs
no extra layer and every removed unit removes whole rows / columns of two weight matrices:

    FFN neuron   row of mlp.fc1  + column of mlp.fc2
    QK channel   row of attn.q   + row of attn.k      (the channel only enters q . k)
    V channel    row of attn.v   + column of attn.proj

All three are "output channels" of a Linear (fc1 / q / v), so pruning.units.NeuronUnit gates them.
A head whose V channels are all removed is dead; its QK channels are then removed for free
(close_dead_heads).

rotate_heads() re-parametrizes every head without changing the function, so that its channels are
the singular directions of the products W_proj,h W_v,h and W_q,h^T W_k,h (the bViT paper's spectral
basis, applied where a removed direction really makes the layer narrower).

compact_channels() builds the really smaller model: one variant per distinct step mask.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn

from pruning.compact import CompactMlp, StepSwitch, _slice_linear

Masks = dict[str, torch.Tensor]  # layer name -> (num_uses, num_units) bool, True = kept


def _plain(module: nn.Module) -> nn.Linear:
    return getattr(module, "linear", module)  # PrunableLinear -> its Linear


def block_names(masks: Masks) -> list[str]:
    """'blocks.0' for every block that has a mask."""
    return sorted({name.rsplit(".", 2)[0] for name in masks})


# * ===== Function-preserving change of basis =====

@torch.no_grad()
def rotate_heads(model: nn.Module) -> None:
    """In place. Per head h (d = head_dim):
        v' = M v, proj' = proj M^-1   with  proj'_h v'_h = U S^1/2 . S^1/2 W^T  (SVD of proj_h v_h)
        q' = M q, k' = M^-T k         with  q'^T k' = A S^1/2 . S^1/2 B^T       (SVD of q_h^T k_h)
    Channels are sorted by singular value. Biases are transformed too, so the output is unchanged."""
    for block in model.blocks:
        attn = block.attn
        q, k, v, proj = (_plain(layer) for layer in (attn.q, attn.k, attn.v, attn.proj))
        head_dim = proj.in_features // attn.num_heads
        for head in range(attn.num_heads):
            rows = slice(head * head_dim, (head + 1) * head_dim)
            # V / PROJ:  P V = Qp (Rp Rv^T) Qv^T
            P, V = proj.weight[:, rows].double(), v.weight[rows].double()
            Qp, Rp = torch.linalg.qr(P)
            Qv, Rv = torch.linalg.qr(V.T)
            Uc, S, Vch = torch.linalg.svd(Rp @ Rv.T)
            root = S.clamp_min(1e-12).sqrt()
            M = (Qp @ Uc / root).T @ P                       # S^-1/2 U^T P
            v.weight[rows] = (root[:, None] * (Qv @ Vch.T).T).to(v.weight.dtype)
            proj.weight[:, rows] = ((Qp @ Uc) * root).to(proj.weight.dtype)
            if v.bias is not None:
                v.bias[rows] = (M @ v.bias[rows].double()).to(v.bias.dtype)
            # Q / K:  Wq^T Wk = Qa (Ra Rb^T) Qb^T
            Qa, Ra = torch.linalg.qr(q.weight[rows].double().T)
            Qb, Rb = torch.linalg.qr(k.weight[rows].double().T)
            Uc, S, Vch = torch.linalg.svd(Ra @ Rb.T)
            root = S.clamp_min(1e-12).sqrt()
            M = (root[:, None] * Uc.T) @ torch.linalg.inv(Ra.T)   # q' = M q
            M_inv_t = (Uc.T / root[:, None]) @ Ra                 # k' = M^-T k
            q.weight[rows] = (root[:, None] * (Qa @ Uc).T).to(q.weight.dtype)
            k.weight[rows] = (root[:, None] * (Qb @ Vch.T).T).to(k.weight.dtype)
            if q.bias is not None:
                q.bias[rows] = (M @ q.bias[rows].double()).to(q.bias.dtype)
            if k.bias is not None:
                k.bias[rows] = (M_inv_t @ k.bias[rows].double()).to(k.bias.dtype)


# * ===== Masks =====

def close_dead_heads(masks: Masks, num_heads: int) -> Masks:
    """A head without V channels contributes nothing: drop its QK channels too."""
    closed = dict(masks)
    for block in block_names(masks):
        q_name, v_name = f"{block}.attn.q", f"{block}.attn.v"
        if q_name not in masks or v_name not in masks:
            continue
        v_mask = masks[v_name]
        alive = v_mask.view(v_mask.shape[0], num_heads, -1).any(dim=2, keepdim=True)  # (T, H, 1)
        q_mask = masks[q_name].view(v_mask.shape[0], num_heads, -1) & alive
        closed[q_name] = q_mask.reshape(masks[q_name].shape)
    return closed


# * ===== Compact model =====

def _distinct(masks: torch.Tensor, per_step: bool) -> tuple[torch.Tensor, torch.Tensor]:
    if per_step:
        return masks.to(torch.uint8), torch.arange(masks.shape[0])
    return torch.unique(masks.to(torch.uint8), dim=0, return_inverse=True)


class ChannelAttention(nn.Module):
    """Attention whose heads have their own numbers of QK and V channels.

    q / k / v / proj are sliced to the kept channels (this is where the FLOPs go down). Inside
    the attention, heads are zero-padded to the widest kept head so one batched call serves all
    heads; the softmax scale stays that of the original head_dim."""

    def __init__(self, attn: nn.Module, qk_keep: torch.Tensor, v_keep: torch.Tensor) -> None:
        super().__init__()
        q, k, v, proj = (_plain(layer) for layer in (attn.q, attn.k, attn.v, attn.proj))
        device = proj.weight.device
        num_heads, head_dim = attn.num_heads, proj.in_features // attn.num_heads
        self.scale = head_dim ** -0.5
        v_keep = v_keep.view(num_heads, head_dim)
        alive = v_keep.any(dim=1)
        qk_keep = qk_keep.view(num_heads, head_dim) & alive[:, None]
        self.num_heads = int(alive.sum())
        self.q, self.k = (_slice_linear(layer, rows=qk_keep.flatten().to(device)) for layer in (q, k))
        self.v = _slice_linear(v, rows=v_keep.flatten().to(device))
        self.proj = _slice_linear(proj, cols=v_keep.flatten().to(device))
        self.qk_width, qk_index = self._padding(qk_keep[alive])
        self.v_width, v_index = self._padding(v_keep[alive])
        self.register_buffer("qk_index", qk_index.to(device), persistent=False)
        self.register_buffer("v_index", v_index.to(device), persistent=False)
        self.qk_ragged = len(qk_index) != self.num_heads * self.qk_width
        self.v_ragged = len(v_index) != self.num_heads * self.v_width

    @staticmethod
    def _padding(keep: torch.Tensor) -> tuple[int, torch.Tensor]:
        """(padded width, position of every kept channel in the padded (heads * width) layout)."""
        counts = keep.sum(dim=1)
        width = max(int(counts.max()) if len(counts) else 0, 1)
        index = [head * width + c for head, count in enumerate(counts.tolist()) for c in range(count)]
        return width, torch.tensor(index, dtype=torch.long)

    def _heads(self, x: torch.Tensor, width: int, index: torch.Tensor, ragged: bool) -> torch.Tensor:
        B, N, _ = x.shape
        if ragged:
            x = x.new_zeros(B, N, self.num_heads * width).index_copy_(2, index, x)
        return x.view(B, N, self.num_heads, width).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        if self.num_heads == 0:  # only the proj bias is left
            return self.proj(x.new_zeros(B, N, 0))
        q = self._heads(self.q(x), self.qk_width, self.qk_index, self.qk_ragged)
        k = self._heads(self.k(x), self.qk_width, self.qk_index, self.qk_ragged)
        v = self._heads(self.v(x), self.v_width, self.v_index, self.v_ragged)
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(1, 2).reshape(B, N, self.num_heads * self.v_width)
        if self.v_ragged:
            out = out.index_select(2, self.v_index)
        return self.proj(out)


@torch.no_grad()
def compact_channels(model: nn.Module, masks: Masks, *, means: Masks | None = None) -> nn.Module:
    """Copy of `model` whose blocks keep only the masked-in channels (masks: fc1 / q / v names).
    Layers without a mask stay full. The model may have a pruner attached.
    means: {fc1 / v name: (steps, units)} mean outputs (LoopGates); a removed unit then contributes its
    mean through a step-specific bias of fc2 / proj instead of nothing."""
    per_step_variants = means is not None
    small = copy.deepcopy(model)
    loop_ctx = getattr(small, "loop_ctx", None)
    for block_name in block_names(masks):
        block = small.get_submodule(block_name)
        attn, mlp = block.attn, block.mlp
        proj, fc1, fc2 = _plain(attn.proj), _plain(mlp.fc1), _plain(mlp.fc2)
        fc1_mask = masks.get(f"{block_name}.mlp.fc1")
        q_mask, v_mask = masks.get(f"{block_name}.attn.q"), masks.get(f"{block_name}.attn.v")
        if fc1_mask is not None:
            distinct, index = _distinct(fc1_mask, per_step_variants)
            union = int(fc1_mask.any(dim=0).sum())
            stored = union * (fc1.in_features + 1 + fc2.out_features) + fc2.bias.numel()
            block.mlp = StepSwitch([CompactMlp(fc1, mlp.act, fc2, mask.bool()) for mask in distinct],
                                   index.tolist(), loop_ctx, stored)
            if means is not None:
                for step, variant in enumerate(block.mlp.variants):
                    removed = (~fc1_mask[step]).to(fc2.weight.device)
                    mean = means[f"{block_name}.mlp.fc1"][step].to(fc2.weight.device, torch.float32)[removed]
                    variant.fc2.bias += (fc2.weight[:, removed].float() @ mean).to(fc2.bias.dtype)
                block.mlp.stored_params += (len(block.mlp.variants) - 1) * fc2.bias.numel()
        if q_mask is not None or v_mask is not None:
            steps = (q_mask if q_mask is not None else v_mask).shape[0]
            full = torch.ones(steps, proj.in_features, dtype=torch.bool)
            q_mask = full if q_mask is None else q_mask
            v_mask = full if v_mask is None else v_mask
            both = torch.cat((q_mask, v_mask), dim=1).to(torch.uint8)
            distinct, index = _distinct(both, per_step_variants)
            dim = proj.in_features
            stored = (2 * int(q_mask.any(dim=0).sum()) * (dim + 1)                 # q, k rows
                      + int(v_mask.any(dim=0).sum()) * (dim + 1 + proj.out_features)  # v rows, proj columns
                      + proj.bias.numel())
            block.attn = StepSwitch([ChannelAttention(attn, row[:dim].bool(), row[dim:].bool()) for row in distinct],
                                    index.tolist(), loop_ctx, stored)
            if means is not None:
                for step, variant in enumerate(block.attn.variants):
                    removed = (~v_mask[step]).to(proj.weight.device)
                    mean = means[f"{block_name}.attn.v"][step].to(proj.weight.device, torch.float32)[removed]
                    variant.proj.bias += (proj.weight[:, removed].float() @ mean).to(proj.bias.dtype)
                block.attn.stored_params += (len(block.attn.variants) - 1) * proj.bias.numel()
        # the norms and anything else in the block are shared and unchanged
    for module in small.modules():  # masks of layers that were not replaced must not linger
        if hasattr(module, "reset") and hasattr(module, "masks"):
            module.reset()
    return small


class LoopGates:
    """Gates on the three unit kinds of a single-block loop, as hooks on the unchanged model:

        fc1 (FFN neuron)  input of mlp.fc2    x -> mean_t + gate_t (x - mean_t)
        v   (V channel)   input of attn.proj  x -> mean_t + gate_t (x - mean_t)
        q   (QK channel)  output of attn.q    q -> gate_t q

    A closed gate replaces the unit by its mean output at that step (mean ablation); with zero means
    this is plain removal. compact_channels(..., means=...) folds the means into step-specific biases
    of fc2 / proj, so the compact model computes the same function.

    modes: off / mask (0-1 masks, differentiable in the weights) / score (per-sample gates with grad;
    d loss_n / d gate[t, i] = sum over tokens of dL/dx (x - mean), the first-order loss change of
    replacing unit i at step t by its mean)."""

    def __init__(self, model: nn.Module, block_name: str = "blocks.0") -> None:
        if len(model.blocks) != 1:
            raise ValueError("LoopGates supports single-block loops (bViT)")
        block = model.get_submodule(block_name)
        self.model, self.ctx = model, model.loop_ctx
        self.steps = model.schedule.num_steps
        self.layers = {"fc1": block.mlp.fc2, "q": block.attn.q, "v": block.attn.proj}
        self.names = {"fc1": f"{block_name}.mlp.fc1", "q": f"{block_name}.attn.q", "v": f"{block_name}.attn.v"}
        self.units = {"fc1": block.mlp.fc2.in_features, "q": block.attn.q.out_features,
                      "v": block.attn.proj.in_features}
        self.device = block.attn.proj.weight.device
        self.mode = "off"
        self.masks = {key: torch.ones(self.steps, units, device=self.device) for key, units in self.units.items()}
        self.means = {key: torch.zeros(self.steps, self.units[key], device=self.device) for key in ("fc1", "v")}
        self.gates: dict[str, torch.Tensor] = {}
        self._sums: dict[str, torch.Tensor] | None = None
        self._count = 0
        self.handles = [
            self.layers["fc1"].register_forward_pre_hook(lambda m, args: self._input("fc1", args)),
            self.layers["v"].register_forward_pre_hook(lambda m, args: self._input("v", args)),
            self.layers["q"].register_forward_hook(lambda m, args, out: self._output("q", out)),
        ]

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()

    def shapes(self) -> dict[str, tuple[int, int]]:
        return {self.names[key]: (self.steps, units) for key, units in self.units.items()}

    def _gate(self, key: str, x: torch.Tensor) -> torch.Tensor:
        step = self.ctx.use_index
        return (self.gates[key][step] if self.mode == "score" else self.masks[key][step]).to(x.dtype)

    def _input(self, key: str, args: tuple):
        if self.mode == "off":
            return None
        x, step = args[0], self.ctx.use_index
        if self._sums is not None:
            self._sums[key][step] += x.detach().float().sum(dim=(0, 1))
            if key == "fc1":
                self._count[step] += x.shape[0] * x.shape[1]
        mean = self.means[key][step].to(x.dtype)
        return (mean + self._gate(key, x) * (x - mean),)

    def _output(self, key: str, out: torch.Tensor):
        return None if self.mode == "off" else out * self._gate(key, out)

    def set_masks(self, masks: Masks, means: Masks | None = None) -> None:
        """means: {name: (steps, units)} for fc1 / v; None keeps the current means."""
        self.mode = "mask"
        for key, name in self.names.items():
            self.masks[key] = masks[name].to(self.device, torch.float32)
            if means is not None and key in self.means:
                self.means[key] = means[name].to(self.device, torch.float32)

    def zero_means(self) -> None:
        self.means = {key: torch.zeros_like(mean) for key, mean in self.means.items()}

    def get_means(self) -> Masks:
        return {self.names[key]: mean.cpu().clone() for key, mean in self.means.items()}

    @torch.no_grad()
    def measure_means(self, batches: list, passes: int = 3) -> None:
        """Mean output of every unit at every step under the current masks. The means of a step depend on
        the replacements made in earlier steps, hence more than one pass."""
        self.model.eval()
        self.mode = "mask"
        for _ in range(passes):
            self._sums = {key: torch.zeros_like(mean) for key, mean in self.means.items()}
            self._count = torch.zeros(self.steps, device=self.device)
            for images, _ in batches:
                with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
                    self.model(images.to(self.device))
            self.means = {key: total / self._count[:, None] for key, total in self._sums.items()}
            self._sums = None

    def score(self, batches: list) -> dict[str, dict[str, torch.Tensor]]:
        """Per-sample gate gradients g[n, t, i] at the current masks and means, per layer name:
            paper (T, U): mean over batches of |batch-mean g|      (the bViT paper's estimator)
            cross (U, T, T): mean_n g[n, t, i] g[n, s, i]          (diagonal = Fisher)"""
        self.model.eval()
        flags = {p: p.requires_grad for p in self.model.parameters()}
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        stats = {key: {"paper": torch.zeros(self.steps, units, device=self.device),
                       "cross": torch.zeros(units, self.steps, self.steps, device=self.device)}
                 for key, units in self.units.items()}
        images_seen = 0
        try:
            for images, targets in batches:
                images, targets = images.to(self.device), targets.to(self.device)
                self.gates = {key: mask[:, None, None, :].repeat(1, len(images), 1, 1).requires_grad_()
                              for key, mask in self.masks.items()}
                self.mode = "score"
                F.cross_entropy(self.model(images).float(), targets, reduction="sum").backward()
                for key in self.units:
                    grad = self.gates[key].grad.squeeze(2)  # (steps, batch, units)
                    stats[key]["paper"] += grad.mean(dim=1).abs()
                    stats[key]["cross"] += torch.einsum("tbi,sbi->its", grad, grad)
                images_seen += len(images)
        finally:
            self.gates, self.mode = {}, "mask"
            for parameter, flag in flags.items():
                parameter.requires_grad_(flag)
        return {self.names[key]: {"paper": s["paper"].cpu() / len(batches), "cross": s["cross"].cpu() / images_seen}
                for key, s in stats.items()}


__all__ = ["LoopGates", "ChannelAttention", "close_dead_heads", "compact_channels", "rotate_heads", "block_names"]
