"""
preprocessing.py
----------------
Stages 1-3 of the DIP pipeline:
  1. Image acquisition / validation   -> load_image()
  2. Sampling / resizing              -> resize_image()
  3. Noise reduction (median filter)  -> denoise()

All images inside the pipeline are RGB uint8 (H x W x 3).
"""
from dataclasses import dataclass, field
import cv2
import numpy as np

from .config import PROCESSING as P


class ImageValidationError(Exception):
    """Raised for uploads that cannot be analysed (message is user-friendly)."""


@dataclass
class LoadedImage:
    rgb: np.ndarray
    filename: str
    fmt: str
    orig_height: int
    orig_width: int
    orig_channels: int
    bit_depth: str
    is_grayscale: bool
    warnings: list = field(default_factory=list)


def _detect_format(data: bytes, filename: str) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "PNG"
    if data[:3] == b"\xff\xd8\xff":
        return "JPEG"
    if data[:4] in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
        return "TIFF"
    if data[:2] == b"BM":
        return "BMP"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WEBP"
    ext = filename.rsplit(".", 1)[-1].upper() if "." in filename else "UNKNOWN"
    return ext


def _to_uint8(img: np.ndarray, warnings: list) -> tuple:
    """Convert 16-bit / float data to 8-bit with a robust percentile stretch."""
    if img.dtype == np.uint8:
        return img, "8-bit"
    depth = "16-bit" if img.dtype == np.uint16 else str(img.dtype)
    f = img.astype(np.float32)
    lo, hi = np.percentile(f, (0.5, 99.5))
    if hi <= lo:
        hi = lo + 1.0
    warnings.append(f"{depth} image converted to 8-bit using a 0.5-99.5 percentile stretch.")
    return np.clip((f - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8), depth


def load_image(data: bytes, filename: str = "image") -> LoadedImage:
    """Decode + validate an uploaded file.  Never returns garbage: raises ImageValidationError."""
    if not data:
        raise ImageValidationError("The uploaded file is empty.")
    warnings = []
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ImageValidationError(
            "This file could not be read as an image. It may be corrupted or in an unsupported "
            "format. Please upload a JPG, PNG or TIFF image.")
    if img.ndim not in (2, 3) or img.size == 0:
        raise ImageValidationError("Unsupported image layout.")
    h, w = img.shape[:2]
    if min(h, w) < P["min_side_px"]:
        raise ImageValidationError(
            f"Image is too small ({w}x{h}px). At least {P['min_side_px']}px per side is needed "
            "for meaningful texture and edge analysis.")
    if h * w > P["max_decode_pixels"]:
        raise ImageValidationError("Image is extremely large; please downsample it before uploading.")

    channels = 1 if img.ndim == 2 else img.shape[2]
    img, depth = _to_uint8(img, warnings)

    if channels == 1:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif channels == 2:
        rgb = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2RGB)
    elif channels == 3:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif channels == 4:
        alpha = img[:, :, 3]
        if (alpha < 250).mean() > 0.05:
            warnings.append("Transparent (alpha) pixels detected; alpha channel was discarded.")
        rgb = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
    else:
        raise ImageValidationError(
            f"{channels}-band image not supported. Please export an RGB (3-band) version.")

    diff = np.abs(rgb[:, :, 0].astype(np.int16) - rgb[:, :, 1]) + \
        np.abs(rgb[:, :, 1].astype(np.int16) - rgb[:, :, 2])
    is_gray = channels <= 2 or float((diff <= 2).mean()) > 0.995
    if is_gray:
        warnings.append("Grayscale image: colour features carry no information, so confidence "
                        "is reduced.")
    if min(h, w) < P["small_warn_px"]:
        warnings.append(f"Small image ({w}x{h}px): texture and edge measurements are less reliable.")
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if float(g.std()) < 2.0:
        warnings.append("Image is almost uniform (very low contrast); the result is unreliable.")
    elif float(g.mean()) < 25 or float(g.mean()) > 235:
        warnings.append("Image is extremely dark or bright; colour evidence may be distorted.")

    return LoadedImage(rgb=rgb, filename=filename, fmt=_detect_format(data, filename),
                       orig_height=h, orig_width=w, orig_channels=channels, bit_depth=depth,
                       is_grayscale=bool(is_gray), warnings=warnings)


def resize_image(rgb: np.ndarray, target_long_side: int) -> np.ndarray:
    """Resample so the longest side equals `target_long_side` (aspect ratio preserved).
    INTER_AREA (pixel-area averaging) is used for down-sampling because it acts as an
    anti-aliasing low-pass filter; INTER_CUBIC is used for up-sampling."""
    h, w = rgb.shape[:2]
    scale = target_long_side / float(max(h, w))
    if abs(scale - 1.0) < 1e-6:
        return rgb.copy()
    new_w, new_h = max(8, round(w * scale)), max(8, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(rgb, (new_w, new_h), interpolation=interp)


def denoise(rgb: np.ndarray, kernel: int = 3) -> np.ndarray:
    """Median filter: removes impulsive (salt-and-pepper) noise and preserves edges,
    unlike a mean/Gaussian blur which smears boundaries."""
    return cv2.medianBlur(rgb, int(kernel))
