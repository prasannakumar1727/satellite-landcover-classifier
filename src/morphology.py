"""
morphology.py
-------------
Morphological clean-up and region analysis of the binary masks.

  opening  (erode -> dilate) removes isolated specks
  closing  (dilate -> erode) fills small holes / gaps
  connected components give region proportion, continuity and fragmentation
"""
import cv2
import numpy as np

from .config import SEGMENTATION as S


def clean_mask(mask: np.ndarray) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (S["morph_kernel"],) * 2)
    m = mask.astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return m > 0


def region_stats(mask: np.ndarray, min_area_frac: float = None) -> dict:
    """Area fraction, largest-region share (continuity) and region count (fragmentation)."""
    total = mask.size
    min_area = max(4, int((min_area_frac or S["min_region_area_frac"]) * total))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    kept = areas[areas >= min_area]
    area_sum = float(mask.sum())
    return {
        "fraction": area_sum / total,
        "continuity": float(kept.max() / area_sum) if kept.size and area_sum > 0 else 0.0,
        "count": int(kept.size),
    }


def patch_analysis(flat: np.ndarray, rgb: np.ndarray) -> dict:
    """Homogeneous patches ("fields"): opened flat regions above a minimum area.
    Returns the patch label map plus size / density / inter-patch tone diversity."""
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    m = cv2.morphologyEx(flat.astype(np.uint8) * 255, cv2.MORPH_OPEN, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
    total = flat.size
    min_area = max(16, int(S["patch_min_area_frac"] * total))
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    if not keep:
        return {"labels": np.zeros(flat.shape, np.int32), "flat_fraction": 0.0,
                "patch_density": 0.0, "patch_color_div": 0.0, "n_patches": 0}
    areas = np.array([stats[i, cv2.CC_STAT_AREA] for i in keep], dtype=np.float64)
    means = np.array([rgb[labels == i].mean(axis=0) for i in keep]) / 255.0   # (n, 3)
    w = areas / areas.sum()
    mu = (means * w[:, None]).sum(axis=0)
    div = float(np.sqrt(((means - mu) ** 2 * w[:, None]).sum(axis=0)).mean())
    lab_out = np.zeros(flat.shape, np.int32)
    for j, i in enumerate(keep, 1):
        lab_out[labels == i] = j
    return {"labels": lab_out, "flat_fraction": float(areas.sum() / total),
            "patch_density": len(keep) / (total / 10000.0),
            "patch_color_div": div, "n_patches": len(keep)}
