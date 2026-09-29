"""Poster figures: 300 dpi PNG + PDF, large fonts, colorblind-validated palette (dataviz reference)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from . import config as C  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# Categorical slots, fixed order (validated adjacent-pair CVD-safe on light surface)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "font.size": 16, "axes.titlesize": 18, "axes.labelsize": 17, "xtick.labelsize": 14, "ytick.labelsize": 14,
    "legend.fontsize": 14, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "lines.linewidth": 2.0, "pdf.fonttype": 42,
})


def save(fig, name):
    C.FIGS_DIR.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(C.FIGS_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def pretty(name):
    return name.replace("BB_", "").replace("_", " ").title().replace("Leg Raised Crunch", "Leg-raise crunch")


def pair_heatmap(screen, order=None, highlight=()):
    """screen: DataFrame with class_a, class_b, balanced_acc. Lower triangle, sequential blue."""
    classes = order if order is not None else sorted(set(screen.class_a) | set(screen.class_b))
    idx = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    M = np.full((n, n), np.nan)
    for r in screen.itertuples():
        i, j = sorted((idx[r.class_a], idx[r.class_b]))
        M[j, i] = r.balanced_acc
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP[::-1])  # dark = hard (low accuracy)
    cmap.set_bad(SURFACE)
    M = M[1:, :-1]  # first row / last column of a lower triangle are empty
    fig, ax = plt.subplots(figsize=(12, 10.5))
    lo = np.floor(np.nanmin(M) * 20) / 20
    im = ax.imshow(M, cmap=cmap, vmin=lo, vmax=1.0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    names = [pretty(C.CLASS_NAMES[c]) for c in classes]
    ax.set_xticks(range(n - 1), names[:-1], rotation=45, ha="right")
    ax.set_yticks(range(n - 1), names[1:])
    ax.tick_params(length=0)
    for j in range(n - 1):
        for i in range(j + 1):
            v = M[j, i]
            dark = (v - lo) / (1 - lo) < 0.55
            s = f"{v:.2f}"
            ax.text(i, j, "1.0" if s == "1.00" else s[1:], ha="center", va="center", fontsize=10.5,
                    color="#ffffff" if dark else INK)
    for a, b in highlight:
        i, j = sorted((idx[a], idx[b]))
        ax.add_patch(plt.Rectangle((i - 0.5, j - 1.5), 1, 1, fill=False, ec=INK, lw=2.5))
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Balanced accuracy (new athletes, all data)")
    cb.outline.set_visible(False)
    ax.set_title("How easily can a wrist sensor tell two exercises apart?", loc="left", pad=14)
    return fig
