"""Plot Top-1 accuracy against model size / remaining rate from ./prun_acc_figure/*/data.pkl.

Usage examples:
    python show_prun_acc_graph.py                         # all models, x = remaining rate
    python show_prun_acc_graph.py --x parameter           # x = parameter size
    python show_prun_acc_graph.py --x remaining parameter # both, side by side
    python show_prun_acc_graph.py --x remaining parameter --remaining-xlim 0.2 1.0 --parameter-xlim 0 50
    python show_prun_acc_graph.py --models sret_s vit_small_patch16_224
    python show_prun_acc_graph.py --exclude sret_s_no_loop --show
"""

import argparse
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import pickle

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
import numpy as np

DATA_ROOT = Path("./prun_acc_figure")
OUTPUT_DIR = DATA_ROOT / "graph"


# * ===== Style settings (edit here to change colors / names) =====

# Family = group of models drawn in the same color (e.g. SReT family, ViT family).
# To add a family, append an entry here.
# "short" is the abbreviation used in the output file name.
FAMILY_STYLES: dict[str, dict[str, str]] = {
    "sret": {"color": "tab:red", "label": "SReT", "short": "sret"},
    "vit": {"color": "tab:blue", "label": "ViT", "short": "vit"},
}
# Used for models whose family is not registered.
DEFAULT_FAMILY_STYLE: dict[str, str] = {"color": "tab:gray", "label": "Other", "short": "other"}

# Each model: family, line style, marker, display name and abbreviation (used in the file name).
# To add a model, append an entry here (the key is the directory name in DATA_ROOT).
MODEL_STYLES: dict[str, dict[str, str]] = {
    "sret_s":                {"family": "sret", "linestyle": "-",  "marker": "o", "label": "SReT-S",            "short": "s"},
    "sret_t":                {"family": "sret", "linestyle": "--", "marker": "s", "label": "SReT-T",            "short": "t"},
    "sret_lt":               {"family": "sret", "linestyle": ":",  "marker": "^", "label": "SReT-LT",           "short": "lt"},
    "sret_s_no_loop":        {"family": "sret", "linestyle": "-.", "marker": "D", "label": "SReT-S (no loop)",  "short": "nl"},
    "sret_s_fine_tuned":     {"family": "sret", "linestyle": "-.", "marker": "*", "label": "SReT-S (fine-tuned)",  "short": "sf"},
    "vit_base_patch16_224":  {"family": "vit",  "linestyle": "-.",  "marker": "o", "label": "ViT-B",          "short": "b"},
    "vit_small_patch16_224": {"family": "vit",  "linestyle": "-", "marker": "s", "label": "ViT-S",          "short": "s"},
    "vit_tiny_patch16_224":  {"family": "vit",  "linestyle": "--",  "marker": "^", "label": "ViT-Ti",         "short": "ti"},
    "loop_vit_large_lora":   {"family": "vit",  "linestyle": ":", "marker": "D", "label": "Loop ViT-L", "short": "loopl"},
}
# Line styles / markers cycled for models not registered in MODEL_STYLES.
FALLBACK_LINESTYLES = [(0, (5, 1)), (0, (3, 1, 1, 1, 1, 1)), (0, (1, 3))]
FALLBACK_MARKERS = ["v", "P", "X", "*"]

# Axis definitions: display label (used in the graph) and file label (used in the file name).
X_AXES: dict[str, dict[str, str]] = {
    "remaining": {"label": "Remaining Rate (1 - Pruning Rate)\n(a) ", "file": "RemainingRate"},
    "parameter": {"label": "Parameter Size [M]\n(b) ", "file": "ParameterSize"},
}
Y_AXIS: dict[str, str] = {"label": "Top-1 Accuracy [%]", "file": "Top1Acc"}

# Default x range of each graph: (left, right). None = automatic (left is 0 when None).
# Overridden by --<axis>-xlim (e.g. --parameter-xlim 0 50).
X_LIMITS: dict[str, tuple[float | None, float | None]] = {
    "remaining": (None, None),
    "parameter": (None, None),
}

PARAMETER_SCALE = 1e6  # parameter size is shown in units of 1e6 (= M)
LEGEND_LOC = "lower right"
DEFAULT_FONTSIZE = 10.0  # base font size of the whole figure (matplotlib default: 10)
FIGSIZE = (12, 6)
DPI = 200


# * ===== Data =====

@dataclass
class ModelResult:
    name: str
    pruning_rate: np.ndarray
    parameter: np.ndarray
    top1: np.ndarray


def find_available_models(root: Path = DATA_ROOT) -> list[str]:
    """Return the names of the model directories that contain data.pkl.
    Registered models come first in MODEL_STYLES order (= legend order), then the rest."""
    found = {p.parent.name for p in root.glob("*/data.pkl")} # glob return for example "prun_acc_figure/sret_s/data.pkl"
    registered = [m for m in MODEL_STYLES if m in found]
    return registered + sorted(found - set(registered)) # exclude m in registered... What?


def load_result(model_name: str, root: Path = DATA_ROOT) -> ModelResult:
    """Load data.pkl saved by compare_acc_by_each_pruning.py."""
    with open(root / model_name / "data.pkl", "rb") as f:
        data = pickle.load(f)
    return ModelResult(
        name=model_name,
        pruning_rate=np.asarray(data["x"]["pruning_rate"], dtype=float),
        parameter=np.asarray(data["x"]["parameter"], dtype=float),
        top1=np.asarray(data["y"]["imagenet_top1"], dtype=float),
    )


def select_models(available: list[str], include: list[str] | None, exclude: list[str]) -> list[str]:
    """Decide which models to draw from --models / --exclude."""
    selected = available if include is None else include
    missing = [m for m in selected if m not in available]  # conform selected have available model
    if missing:
        raise ValueError(f"data.pkl not found for: {missing} (available: {available})")
    return [m for m in selected if m not in exclude]


# * ===== Style =====

def get_family(model_name: str) -> str | None:
    """Family of the model: MODEL_STYLES first, then prefix match on FAMILY_STYLES."""
    if model_name in MODEL_STYLES:
        return MODEL_STYLES[model_name]["family"]
    for family in FAMILY_STYLES:
        if model_name.startswith(family):
            return family
    return None


def get_family_style(family: str | None) -> dict[str, str]:
    return FAMILY_STYLES.get(family, DEFAULT_FAMILY_STYLE) if family else DEFAULT_FAMILY_STYLE


def get_model_style(model_name: str, fallback_index: int) -> dict:
    """Return the matplotlib kwargs (color, linestyle, marker, label) for the model."""
    family_style = get_family_style(get_family(model_name))
    model_style = MODEL_STYLES.get(model_name, {
        "linestyle": FALLBACK_LINESTYLES[fallback_index % len(FALLBACK_LINESTYLES)],
        "marker": FALLBACK_MARKERS[fallback_index % len(FALLBACK_MARKERS)],
        "label": model_name,
    })
    return {
        "color": family_style["color"],
        "linestyle": model_style["linestyle"],
        "marker": model_style["marker"],
        "label": model_style["label"],
    }


# * ===== Plot =====

def get_x_values(result: ModelResult, x_axis: str) -> np.ndarray:
    if x_axis == "remaining":
        return 1.0 - result.pruning_rate
    if x_axis == "parameter":
        return result.parameter / PARAMETER_SCALE
    raise ValueError(f"unknown x axis: {x_axis}")


def plot_models(ax: Axes, results: list[ModelResult], x_axis: str) -> None:
    """Draw one line per model."""
    fallback_index = 0
    for result in results:
        style = get_model_style(result.name, fallback_index)
        if result.name not in MODEL_STYLES:
            fallback_index += 1
        ax.plot(get_x_values(result, x_axis), result.top1 * 100, markersize=4, **style)


def add_legends(ax: Axes, results: list[ModelResult]) -> None:
    """Model legend (line style) at lower right, family legend (color) stacked above it."""
    model_legend = ax.legend(loc=LEGEND_LOC, title="Model", fontsize="large")
    ax.add_artist(model_legend)

    # families = list(dict.fromkeys(get_family(r.name) for r in results))
    # family_handles = [
    #     Line2D([], [], color=get_family_style(f)["color"], linewidth=3, label=get_family_style(f)["label"])
    #     for f in families
    # ]
    # # Place the family legend right above the model legend so they do not overlap.
    # top = _legend_top_in_axes(ax, model_legend)
    # ax.legend(handles=family_handles, loc=LEGEND_LOC, bbox_to_anchor=(1.0, top),
    #           title="Family", fontsize="large")


def _legend_top_in_axes(ax: Axes, legend: Legend) -> float:
    ax.figure.canvas.draw()
    bbox = legend.get_window_extent().transformed(ax.transAxes.inverted())
    return bbox.y1


def decorate_axes(ax: Axes, x_axis: str, xlim: tuple[float | None, float | None], show_ylabel: bool = True) -> None:
    ax.set_xlabel(X_AXES[x_axis]["label"])
    if show_ylabel:
        ax.set_ylabel(Y_AXIS["label"])
    left, right = xlim
    ax.set_xlim(left=0 if left is None else left, right=right)
    ax.grid(True, linestyle=":", alpha=0.6)


def set_fontsize(fontsize: float) -> None:
    """Set the base font size; labels, ticks and legends scale with it."""
    plt.rcParams["font.size"] = fontsize


def make_figure(results: list[ModelResult], x_axes: list[str],
                xlims: dict[str, tuple[float | None, float | None]]) -> plt.Figure:
    """One subplot per x axis, placed side by side and sharing the y axis.
    xlims gives the x range of each graph by axis name."""
    n = len(x_axes)
    fig, axes = plt.subplots(1, n, figsize=(FIGSIZE[0] * n, FIGSIZE[1]), sharey=True, squeeze=False)
    for i, (ax, x_axis) in enumerate(zip(axes[0], x_axes)):
        plot_models(ax, results, x_axis)
        decorate_axes(ax, x_axis, xlims.get(x_axis, (None, None)), show_ylabel=(i == 0))
        add_legends(ax, results)
    fig.tight_layout()
    return fig


# * ===== Save =====

def build_models_tag(model_names: list[str]) -> str:
    """Abbreviation of the shown models grouped by family, e.g. "sret_s_t_lt_nl-vit_b_s".
    Unregistered models use their full name as the abbreviation."""
    groups: dict[str, list[str]] = {}
    for name in model_names:
        family_short = get_family_style(get_family(name))["short"]
        groups.setdefault(family_short, []).append(MODEL_STYLES.get(name, {}).get("short", name))
    return "-".join(f"{family}_{'_'.join(shorts)}" for family, shorts in groups.items())


def build_output_path(x_axes: list[str], model_names: list[str], output_dir: Path = OUTPUT_DIR) -> Path:
    """File name: <Y axis>_vs_<X axis>[And<X axis>...]_<models>_<YYYYmmdd_HHMMSS>.png"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    x_tag = "And".join(X_AXES[x]["file"] for x in x_axes)
    models_tag = build_models_tag(model_names)
    return output_dir / f"{Y_AXIS['file']}_vs_{x_tag}_{models_tag}_{timestamp}.png"


def save_figure(fig: plt.Figure, path: Path) -> None:
    os.makedirs(path.parent, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    print(f"saved: {path}")


# * ===== Main =====

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Top-1 accuracy of pruned models.")
    parser.add_argument("--x", nargs="+", choices=list(X_AXES), default=["remaining"],
                        help="x axis: remaining (1 - pruning rate) and/or parameter (parameter size). "
                             "Give both (e.g. --x remaining parameter) to draw them side by side in that order")
    for x_axis in X_AXES:
        parser.add_argument(f"--{x_axis}-xlim", nargs=2, type=float, metavar=("LEFT", "RIGHT"), default=None,
                            help=f"x range of the {x_axis} graph (default: X_LIMITS)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="models to show (default: all models in prun_acc_figure)")
    parser.add_argument("--exclude", nargs="+", default=[], help="models to hide")
    parser.add_argument("--fontsize", type=float, default=DEFAULT_FONTSIZE,
                        help=f"base font size of the whole figure (default: {DEFAULT_FONTSIZE})")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR, help="output directory")
    parser.add_argument("--show", action="store_true", help="also show the figure in a window")
    return parser.parse_args()


def get_xlims(args: argparse.Namespace) -> dict[str, tuple[float | None, float | None]]:
    """x range of each graph: --<axis>-xlim if given, otherwise X_LIMITS."""
    xlims = {}
    for x_axis in X_AXES:
        given = getattr(args, f"{x_axis}_xlim")
        xlims[x_axis] = tuple(given) if given is not None else X_LIMITS.get(x_axis, (None, None))
    return xlims


def main() -> None:
    args = parse_args()
    model_names = select_models(find_available_models(), args.models, args.exclude)
    results = [load_result(name) for name in model_names]

    set_fontsize(args.fontsize)
    x_axes = list(dict.fromkeys(args.x))  # drop duplicates, keep order
    fig = make_figure(results, x_axes, get_xlims(args))
    save_figure(fig, build_output_path(x_axes, model_names, args.out_dir))
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
