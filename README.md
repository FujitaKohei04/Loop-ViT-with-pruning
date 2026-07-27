# ViT vs LOOP with pruning

This repository includes a small wrapper for using pretrained Vision
Transformers from `timm`.

## Install

```bash
pip install -e ".[vit]"
```

## Build a pretrained ViT

```python
from vit_vs_loop_with_pruning import PretrainedViTConfig, build_pretrained_vit

model = build_pretrained_vit(
    PretrainedViTConfig(
        model_name="vit_base_patch16_224.augreg_in21k_ft_in1k",
        num_classes=10,
        pretrained=True,
        freeze_backbone=False,
    )
)
```

The default checkpoint is a ViT-B/16 model exposed by `timm`. If
`num_classes` differs from the pretrained classifier, the backbone is loaded
from pretrained weights and the classification head is initialized for the new
task.

## CLI

```bash
pretrained-vit --num-classes 10 --freeze-backbone
```

## Taylor pruning

`main.py` can estimate first-order Taylor importance scores for ViT `Linear`
layers and apply global pruning masks:

```bash
python main.py --no-pretrained --taylor-prune-amount 0.3 --calibration-batches 2
```

The score is `abs(weight * gradient)`, accumulated over calibration batches.
The current CLI uses random calibration data as a runnable demo; for real
experiments, replace `make_random_calibration_batches` with batches from the
training or validation loader.

This implementation applies unstructured masks to `Linear` weights, so layer
shapes, hidden sizes, and attention head counts do not change. The visible
effect is sparsity and accuracy change, not a smaller printed architecture.

## LoopViT

`LoopViT.py` provides a minimal looped ViT for experiments. By default it starts
from `timm`'s `vit_base_patch16_224`, keeps blocks 1, 6, and 12, and repeats
those three blocks four times. Iteration embeddings can be enabled or disabled.

```python
from LoopViT import build_loop_vit_base_patch16_224

model = build_loop_vit_base_patch16_224(
    pretrained=True,
    num_classes=1000,
    use_iteration_embedding=True,
    loop_count=4,
)
```

For fine-tuning, choose a trainable-parameter mode and pass the returned groups
to your optimizer:

```python
model.configure_finetuning("head-norm")
param_groups = model.get_finetuning_parameter_groups(
    base_lr=1e-5,
    head_lr=1e-3,
    iteration_lr=1e-4,
)
```

## ImageNet validation

For ImageNet experiments, prepare the standard `ImageFolder` layout:

```text
/path/to/imagenet/
  train/
    n01440764/
    n01443537/
    ...
  val/
    n01440764/
    n01443537/
    ...
```

To test only a pretrained model on ImageNet validation, skip pruning:

```bash
python main.py \
  --eval-only \
  --imagenet-root /path/to/imagenet \
  --model-name vit_base_patch16_224 \
  --num-classes 1000 \
  --batch-size 32
```

Then run Taylor pruning with real ImageNet calibration batches and evaluate on
the validation split. By default, the script reports validation accuracy before
and after pruning:

```bash
python main.py \
  --imagenet-root /path/to/imagenet \
  --model-name vit_base_patch16_224 \
  --num-classes 1000 \
  --batch-size 32 \
  --calibration-batches 32 \
  --taylor-prune-amount 0.3
```

For a quick smoke test, limit validation:

```bash
python main.py --imagenet-root /path/to/imagenet --eval-batches 10
```

Use `--skip-baseline-eval` when you only want the post-pruning score.
