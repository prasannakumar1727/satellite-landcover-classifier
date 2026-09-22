"""
segmentation.py
---------------
Colour/threshold-based segmentation of candidate land-cover regions.  Purely
per-pixel rules on the uploaded image (no clustering model, no training):

  vegetation-like : Excess-Green  ExG = 2g - r - b  on chromatic coordinates
                    (an RGB-only index; NOT NDVI, which needs a NIR band)
  water-like      : blue-dominant pixels, OR dark + flat + non-green pixels
  bare-soil-like  : warm (R >= G >= B) moderately saturated pixels
  built-up-like   : low-saturation grey pixels, OR structured (edge-dense) non-vegetated pixels
  flat regions    : pixels whose smoothed gradient is very small (candidate field / water interiors)
"""
import cv2
import numpy as np

from .config import SEGMENTATION as S


def colour_maps(rgb: np.ndarray) -> dict:
    """Per-pixel colour quantities used by segmentation and feature extraction."""
    f = rgb.astype(np.float32) / 255.0
    total = f.sum(axis=2) + 1e-6
    cr, cg, cb = f[:, :, 0] / total, f[:, :, 1] / total, f[:, :, 2] / total
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return {
        "f": f, "cr": cr, "cg": cg, "cb": cb,
        "exg": 2 * cg - cr - cb,
        "hue": hsv[:, :, 0].astype(np.float32) * 2.0,          # degrees 0-358
        "sat": hsv[:, :, 1].astype(np.float32) / 255.0,
        "val": hsv[:, :, 2].astype(np.float32) / 255.0,
    }


def local_std(gray: np.ndarray, window: int) -> np.ndarray:
    """Local standard deviation (grey levels) in a window x window neighbourhood."""
    g = gray.astype(np.float32)
    mean = cv2.boxFilter(g, -1, (window, window))
    sq = cv2.boxFilter(g * g, -1, (window, window))
    return np.sqrt(np.maximum(sq - mean * mean, 0.0))


def vegetation_mask(m: dict) -> np.ndarray:
    f = m["f"]
    return ((m["exg"] > S["veg_exg_min"]) & (f[:, :, 1] >= f[:, :, 0] * 0.97) &
            (f[:, :, 1] > f[:, :, 2]) & (m["sat"] > S["veg_min_sat"]) & (m["val"] > S["veg_min_val"]))


def water_mask(m: dict, lstd: np.ndarray) -> np.ndarray:
    f = m["f"]
    r, b, g = f[:, :, 0], f[:, :, 2], f[:, :, 1]
    blue = ((b >= r + S["water_blue_margin"]) & (b >= g - S["water_blue_green_tol"]) &
            (m["sat"] > S["water_min_sat"]) & (m["val"] < S["water_max_val"]))
    dark = ((m["val"] < S["water_dark_val"]) & (m["sat"] > S["water_dark_min_sat"]) &
            (lstd < S["water_flat_std"]) & (m["exg"] < S["water_dark_max_exg"]) & (b >= r - 0.02))
    return blue | dark


def soil_mask(m: dict) -> np.ndarray:
    f = m["f"]
    return ((f[:, :, 0] >= f[:, :, 1]) & (f[:, :, 1] >= f[:, :, 2] * 0.95) &
            (m["sat"] > S["soil_min_sat"]) & (m["val"] > S["soil_min_val"]))


def gray_mask(m: dict) -> np.ndarray:
    return (m["sat"] < S["gray_max_sat"]) & (m["val"] > S["gray_min_val"])


def builtup_mask(gray_m: np.ndarray, local_edge: np.ndarray, veg: np.ndarray,
                 water: np.ndarray) -> np.ndarray:
    structured = (local_edge > S["builtup_local_edge"]) & ~veg & ~water
    return gray_m | structured


def flat_mask(gray: np.ndarray) -> np.ndarray:
    """Pixels whose Gaussian-smoothed gradient magnitude is tiny: interiors of fields / water."""
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.5)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return np.hypot(gx, gy) < S["flat_gradient_max"]
