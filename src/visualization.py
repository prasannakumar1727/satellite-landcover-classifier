"""
visualization.py
----------------
Every image/plot here is computed from the actual intermediate data of the
analysed image - nothing is decorative or pre-rendered.
"""
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import CLASS_META

BG, FG, GRID = "#131A26", "#C9D2E3", "#2A3446"
REGION_COLORS = {"Vegetation": (63, 163, 77), "Water": (58, 134, 255),
                 "Bare soil": (233, 180, 76), "Built-up": (235, 235, 245)}


def region_overlay(base_rgb: np.ndarray, aux: dict) -> np.ndarray:
    """Cleaned segmentation masks (after morphological opening + closing) painted over a dimmed
    copy of the image.  Painting order = priority: built-up < soil < vegetation < water."""
    out = (base_rgb.astype(np.float32) * 0.35)
    for key, name in (("built", "Built-up"), ("soil", "Bare soil"),
                      ("veg", "Vegetation"), ("water", "Water")):
        m = aux[key]
        out[m] = 0.35 * base_rgb[m] + 0.65 * np.array(REGION_COLORS[name], np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def texture_map(local_std: np.ndarray) -> np.ndarray:
    """Local standard deviation as a heat-map (bright = rough, dark = smooth)."""
    norm = np.clip(local_std / 30.0, 0, 1)
    return cv2.cvtColor(cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO),
                        cv2.COLOR_BGR2RGB)


def _style(ax):
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.tick_params(colors=FG, labelsize=8)
    ax.grid(color=GRID, alpha=0.5, linewidth=0.5)


def histogram_figure(before: dict, after: dict, title_after: str) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.9), sharey=True)
    fig.patch.set_facecolor(BG)
    for ax, h, ttl in ((axes[0], before, "Before enhancement"), (axes[1], after, title_after)):
        _style(ax)
        ax.fill_between(range(256), h["gray"], color="#8E9AAF", alpha=0.45, label="Gray")
        for hc, col, lab in zip(h["rgb"], ("#FF5A5F", "#3FA34D", "#3A86FF"), "RGB"):
            ax.plot(hc, color=col, linewidth=1.0, alpha=0.9, label=lab)
        ax.set_title(ttl, color=FG, fontsize=10)
        ax.set_xlim(0, 255)
        ax.set_xlabel("Intensity", color=FG, fontsize=8)
    axes[0].set_ylabel("Pixels", color=FG, fontsize=8)
    axes[1].legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=7, loc="upper right")
    fig.tight_layout()
    return fig


def feature_bar_figure(norm: dict) -> plt.Figure:
    """Normalised (0-1) values of the most decision-relevant features."""
    keys = [("veg_frac", "Vegetation-like area"), ("water_frac", "Water-like area"),
            ("soil_frac", "Bare-soil-like area"), ("builtup_region_frac", "Built-up-like area"),
            ("sat_mean", "Saturation"), ("local_std_mean", "Local variation"),
            ("local_entropy", "Texture entropy"), ("canny_density", "Edge density"),
            ("rectilinearity", "Rectilinearity"), ("flat_fraction", "Flat-patch area"),
            ("veg_continuity", "Vegetation continuity")]
    keys = [(k, l) for k, l in keys if k in norm][::-1]
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    fig.patch.set_facecolor(BG)
    _style(ax)
    vals = [norm[k] for k, _ in keys]
    ax.barh([l for _, l in keys], vals, color="#4C9AFF", height=0.62)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Normalised value (0 – 1)", color=FG, fontsize=8)
    ax.grid(axis="y", visible=False)
    for i, v in enumerate(vals):
        ax.text(min(v + 0.02, 0.9), i, f"{v:.2f}", va="center", color=FG, fontsize=7)
    fig.tight_layout()
    return fig
