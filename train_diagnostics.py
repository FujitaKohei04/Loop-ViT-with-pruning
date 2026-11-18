"""Training diagnostics for train_bvit.py, logged to Weights & Biases.

What is recorded and which failure it helps to find:

    every optimizer step, averaged over `log_interval` steps (x axis: train/optim_step)
        train/loss, train/lr                      divergence, schedule bugs
        train/grad_norm(_max), train/clip_frac    exploding gradients, clipping always on
        train/amp_scale, train/amp_skipped        fp16 overflow (GradScaler skipping steps)
        train/nonfinite_loss                      NaN / inf in the forward pass
        train/logit_*, train/pred_*               logit blow-up, collapse to one class
        sys/*                                     throughput, GPU memory
    every `diag_interval` optimizer steps (heavier, extra no-grad forward on the current batch)
        grad_norm/<group>, param_norm/<group>     which part of the model the gradient lives in
        update_ratio/<group>                      |dW| / |W| per step (healthy ~1e-3)
        loop_rms/tXX, loop_maxabs/tXX             residual stream growth over the loop steps
        loop_delta/tXX                            |x_t - x_{t-1}| / |x_{t-1}|: does the loop still move?
        loop_attn_entropy/tXX                     attention collapse (entropy -> 0) per step
        loop_loss/tXX                             loss if inference stopped after step t
    every eval (x axis: epoch)
        val/*, ema_val/*, val_step/top1_tXX       accuracy, loss, per-step accuracy

Everything is a no-op when wandb is disabled; the cheap scalars (grad norm, skipped AMP
steps, non-finite losses) are still returned in the epoch record printed by train_bvit.py.
"""

from __future__ import annotations

import math
import re
import uuid
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch
from torch import nn

RUN_ID_FILE = "wandb_run_id.txt"


# * ===== W&B run =====

class WandbRun:
    """Thin wrapper: every method does nothing when disabled, so the training code needs no ifs."""

    def __init__(self, *, enabled: bool, out_dir: Path, project: str, entity: str | None, group: str | None,
                 mode: str, config: dict, tags: list[str]) -> None:
        self.enabled = enabled
        self.run = None
        self._alerted: set[str] = set()
        if not enabled:
            return
        import wandb
        self.wandb = wandb
        # the run id is kept next to the checkpoints so that --resume continues the same W&B run
        id_file = out_dir / RUN_ID_FILE
        run_id = id_file.read_text().strip() if id_file.exists() else uuid.uuid4().hex[:8]
        id_file.write_text(run_id)
        self.run = wandb.init(project=project, entity=entity, group=group, name=out_dir.name, id=run_id,
                              resume="allow", dir=str(out_dir), mode=mode, config=config, tags=tags)
        wandb.define_metric("train/optim_step")
        wandb.define_metric("epoch")
        for prefix in ("train", "sys", "grad_norm", "param_norm", "update_ratio", "loop_rms", "loop_maxabs",
                       "loop_delta", "loop_attn_entropy", "loop_loss"):
            wandb.define_metric(f"{prefix}/*", step_metric="train/optim_step")
        for prefix in ("val", "ema_val", "val_step", "ema_val_step", "epoch_stats"):
            wandb.define_metric(f"{prefix}/*", step_metric="epoch")

    def log(self, values: dict) -> None:
        if self.enabled:
            self.run.log(values)

    def summary(self, values: dict) -> None:
        if self.enabled:
            self.run.summary.update(values)

    def alert(self, key: str, title: str, text: str) -> None:
        """Send a W&B alert once per key (e.g. first NaN loss)."""
        if not self.enabled or key in self._alerted:
            return
        self._alerted.add(key)
        try:
            self.run.alert(title=title, text=text, level=self.wandb.AlertLevel.WARN)
        except Exception as error:  # alerts fail in offline mode; the metric is still logged
            print(f"wandb alert failed: {error}")

    def finish(self, exit_code: int = 0) -> None:
        if self.enabled:
            self.run.finish(exit_code=exit_code)


# * ===== Parameter groups =====

def group_name(parameter_name: str) -> str:
    """blocks.3.attn.q.weight -> blocks.3.attn, blocks.0.norm1.weight -> blocks.0.norm,
    patch_embed.weight -> patch_embed, pos_embed -> pos_embed."""
    match = re.match(r"blocks\.(\d+)\.(attn|mlp|norm)", parameter_name)
    if match:
        return f"blocks.{match.group(1)}.{match.group(2)}"
    return parameter_name.split(".")[0]


def group_norms(model: nn.Module) -> tuple[dict[str, float], dict[str, float]]:
    """(grad norm per group, parameter norm per group). Call after unscaling the gradients."""
    grad_sq: dict[str, float] = defaultdict(float)
    param_sq: dict[str, float] = defaultdict(float)
    for name, parameter in model.named_parameters():
        key = group_name(name)
        param_sq[key] += float(parameter.detach().float().pow(2).sum())
        if parameter.grad is not None:
            grad_sq[key] += float(parameter.grad.detach().float().pow(2).sum())
    return ({k: math.sqrt(v) for k, v in grad_sq.items()}, {k: math.sqrt(v) for k, v in param_sq.items()})


def snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}


def update_ratios(model: nn.Module, before: dict[str, torch.Tensor]) -> dict[str, float]:
    """|W_after - W_before| / |W_before| per group (0 when the AMP step was skipped)."""
    delta_sq: dict[str, float] = defaultdict(float)
    weight_sq: dict[str, float] = defaultdict(float)
    for name, parameter in model.named_parameters():
        if name not in before:
            continue
        key = group_name(name)
        delta_sq[key] += float((parameter.detach() - before[name]).float().pow(2).sum())
        weight_sq[key] += float(before[name].float().pow(2).sum())
    return {k: math.sqrt(delta_sq[k]) / max(math.sqrt(weight_sq[k]), 1e-12) for k in delta_sq}


# * ===== Batch statistics =====

@torch.no_grad()
def logit_stats(logits: torch.Tensor, num_classes: int) -> dict[str, float]:
    """Logit scale and prediction diversity of one batch. pred_unique_frac -> 1/num_classes = collapse."""
    logits = logits.detach().float()
    probs = logits.softmax(dim=-1)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
    predictions = logits.argmax(dim=-1)
    return {
        "train/logit_maxabs": float(logits.abs().max()),
        "train/logit_std": float(logits.std()),
        "train/pred_entropy": float(entropy.mean()) / math.log(num_classes),  # 1 = uniform
        "train/pred_unique_frac": predictions.unique().numel() / min(num_classes, predictions.numel()),
    }


@contextmanager
def _record_loop(model: nn.Module) -> Iterator[list[dict]]:
    """Collect token statistics after every loop step of RecurrentViT (blocks are re-used, so the
    hook fires once per step in step order) and the attention entropy of that step."""
    records: list[dict] = []
    attentions = [block.attn for block in model.blocks]
    for attn in attentions:
        attn.record_attn = True

    def hook(module: nn.Module, inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
        x_in, x_out = inputs[0].float(), output.float()
        attn = module.attn.last_attn.float()
        entropy = -(attn * attn.clamp_min(1e-12).log()).sum(dim=-1).mean() / math.log(attn.shape[-1])
        records.append({
            "rms": float(x_out.pow(2).mean().sqrt()),
            "maxabs": float(x_out.abs().max()),
            "delta": float((x_out - x_in).norm() / x_in.norm().clamp_min(1e-12)),
            "attn_entropy": float(entropy),
        })

    handles = [block.register_forward_hook(hook) for block in model.blocks]
    try:
        yield records
    finally:
        for handle in handles:
            handle.remove()
        for attn in attentions:
            attn.record_attn = False
            attn.last_attn = None


@torch.no_grad()
def loop_probe(model: nn.Module, images: torch.Tensor, soft_targets: torch.Tensor, loss_fn: nn.Module,
               autocast) -> dict[str, float]:
    """Extra no-grad forward on the current training batch, per loop step t:
    token RMS / max |x| (residual growth, fp16 overflow), relative update of the step,
    normalized attention entropy and the loss if the loop were stopped after t."""
    was_training = model.training
    model.eval()  # no dropout in this model; eval only to make the probe deterministic
    try:
        with autocast, _record_loop(model) as records:
            per_step_logits = model(images, return_all_steps=True)
        values: dict[str, float] = {}
        for t, (record, logits) in enumerate(zip(records, per_step_logits)):
            values[f"loop_rms/t{t:02d}"] = record["rms"]
            values[f"loop_maxabs/t{t:02d}"] = record["maxabs"]
            values[f"loop_delta/t{t:02d}"] = record["delta"]
            values[f"loop_attn_entropy/t{t:02d}"] = record["attn_entropy"]
            values[f"loop_loss/t{t:02d}"] = float(loss_fn(logits.float(), soft_targets))
        return values
    finally:
        model.train(was_training)


def gpu_memory(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    return {"sys/gpu_mem_peak_gb": torch.cuda.max_memory_allocated(device) / 2**30,
            "sys/gpu_mem_reserved_gb": torch.cuda.memory_reserved(device) / 2**30}
