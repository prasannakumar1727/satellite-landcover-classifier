"""
enhancement.py
--------------
Stages 4-5 of the DIP pipeline:
  4. Histogram / contrast enhancement  -> enhance_contrast()
  5. Sharpening (unsharp masking)      -> sharpen()

Enhancement is applied to the luminance channel only, so hue/saturation are
not distorted.  (Radiometric colour measurements are taken from the
noise-reduced image, see feature_extraction.py.)
"""
import cv2
import numpy as np

from .config import PROCESSING as P


def enhance_contrast(rgb: np.ndarray, method: str = "CLAHE") -> np.ndarray:
    if method == "Histogram Equalization":
        ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=P["clahe_clip_limit"],
                            tileGridSize=(P["clahe_tile_grid"],) * 2)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def sharpen(rgb: np.ndarray, enabled: bool = True) -> np.ndarray:
    """Unsharp masking: out = img + amount * (img - gaussian(img)); mild by design."""
    if not enabled:
        return rgb.copy()
    blurred = cv2.GaussianBlur(rgb, (0, 0), P["unsharp_sigma"])
    a = P["unsharp_amount"]
    return cv2.addWeighted(rgb, 1.0 + a, blurred, -a, 0)


def gray_histogram(rgb: np.ndarray) -> np.ndarray:
    return np.bincount(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).ravel(), minlength=256)


def channel_histograms(rgb: np.ndarray) -> list:
    return [np.bincount(rgb[:, :, c].ravel(), minlength=256) for c in range(3)]
