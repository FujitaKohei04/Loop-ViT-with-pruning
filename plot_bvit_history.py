"""Plot a bViT run's history.json against a reference run (default: bvit_s on CIFAR-100).

    uv run python plot_bvit_history.py outputs/bvit/bvit_s_tiny_imagenet_20261111_190414

Panels
    (a) EMA val top-1 per epoch, both runs
    (b) train loss / ln(num_classes) (mixup soft-target CE, so only the trend is meaningful)
    (c) per-loop-step top-1 on train (eval transform) and val, from <run>/gap_eval.json if present
    (d) target run: val top-1 / top-5 relative to their maximum, second half of training
Output: <run>/history.png
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1, 2
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NUM_CLASSES = {"cifar10": 10, "cifar100": 100, "tiny_imagenet": 200, "imagenet100": 100, "imagenet1k": 1000}


def load(run: Path) -> tuple[list[dict], dict]:
    return json.loads((run / "history.json").read_text()), json.loads((run / "config.json").read_text())


def run_label(config: dict) -> str:
    return f"{config['dataset']} (aug={config.get('augment', 'auto')}, {config['epochs']}ep)"


def load_gap(run: Path) -> dict:
    path = run / "gap_eval.json"
    return json.loads(path.read_text()) if path.exists() else {}


def style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run")
    parser.add_argument("--ref", default="outputs/bvit/bvit_s_cifar100_20261027-140057")
    args = parser.parse_args()
    run, ref = Path(args.run), Path(args.ref)
    jp_font = Path("/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf")
    if jp_font.exists():
        font_manager.fontManager.addfont(jp_font)
    plt.rcParams["font.family"] = ["IPAexGothic", "DejaVu Sans"]

    runs = [(run, *load(run), BLUE), (ref, *load(ref), ORANGE)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), facecolor="#fcfcfb")
    (a, b), (c, d) = axes

    for _, history, config, color in runs:
        label = run_label(config)
        epochs = [r["epoch"] for r in history]
        top1 = [100 * r["ema_val_top1"] for r in history]
        a.plot(epochs, top1, color=color, linewidth=2, label=label)
        best = max(range(len(top1)), key=top1.__getitem__)
        a.annotate(f"best {top1[best]:.1f}% (ep {epochs[best]})\nfinal {top1[-1]:.1f}%",
                   (epochs[-1], top1[-1]), xytext=(-8, -30 if color == BLUE else 8),
                   textcoords="offset points", ha="right", fontsize=9, color=INK)
        norm = math.log(NUM_CLASSES[config["dataset"]])
        b.plot(epochs, [r["train_loss"] / norm for r in history], color=color, linewidth=2, label=label)
    style(a, "(a) val top-1（EMA）", "epoch", "top-1 [%]")
    style(b, "(b) train loss / ln(クラス数)  ※Mixup/CutMix 下の値", "epoch", "正規化 loss")
    a.legend(frameon=False)
    b.legend(frameon=False)

    target_gap = load_gap(run)
    if target_gap:
        for (path, _, config, color) in runs:
            # each run's own gap_eval.json first, else the target's (it may hold the reference too)
            g = load_gap(path).get(config["dataset"]) or target_gap.get(config["dataset"])
            if g is None or (path != run and not load_gap(path) and config["dataset"] == runs[0][2]["dataset"]):
                continue
            steps = range(1, len(g["val"]["top1_per_step"]) + 1)
            c.plot(steps, [100 * v for v in g["train"]["top1_per_step"]], color=color, linewidth=2,
                   linestyle="--", marker="o", markersize=4, label=f"{run_label(config)} train")
            c.plot(steps, [100 * v for v in g["val"]["top1_per_step"]], color=color, linewidth=2,
                   marker="o", markersize=4, label=f"{run_label(config)} val")
            tr, va = 100 * g["train"]["top1"], 100 * g["val"]["top1"]
            c.annotate(f"gap {tr - va:.1f}pt", (len(steps), (tr + va) / 2), xytext=(-6, 0),
                       textcoords="offset points", ha="right", va="center", fontsize=9, color=INK)
        c.legend(frameon=False, fontsize=9, loc="upper left")
    style(c, "(c) ループステップ別 top-1（最終 EMA, 拡張なし）", "ループステップ", "top-1 [%]")

    history = runs[0][1]
    half = [r for r in history if r["epoch"] >= len(history) // 3]
    epochs = [r["epoch"] for r in half]
    for key, linestyle, label in (("ema_val_top1", "-", "top-1"), ("ema_val_top5", "--", "top-5")):
        values = [100 * r[key] for r in half]
        peak = max(values)
        d.plot(epochs, [v - peak for v in values], color=BLUE, linewidth=2, linestyle=linestyle,
               label=f"{label}（最大 {peak:.1f}%）")
    d.axhline(0, color=INK2, linewidth=0.8)
    style(d, f"(d) {runs[0][2]['dataset']}: 最大値からの差（EMA val）", "epoch", "最大値との差 [pt]")
    d.legend(frameon=False)

    fig.suptitle(f"{run.name}  vs  {ref.name}", fontsize=12, color=INK)
    fig.tight_layout()
    out = run / "history.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    print(out)


if __name__ == "__main__":
    main()
