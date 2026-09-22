"""
feature_normalization.py
------------------------
Stage 7: map raw feature values onto a common 0-1 scale so that features with
different units (pixels, bits, fractions) can be combined in one score.

Method: fixed-range min-max scaling with clipping,
        n = clip((x - lo) / (hi - lo), 0, 1)
The (lo, hi) reference ranges live in config.NORMALIZATION_RANGES.  They are
fixed constants reasoned from what each measurement means physically - they
are NOT statistics computed from any set of images.
"""
import numpy as np

from .config import NORMALIZATION_RANGES


def ramp(x: float, lo: float, hi: float) -> float:
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def normalize_features(raw: dict) -> dict:
    return {k: ramp(raw[k], lo, hi) for k, (lo, hi) in NORMALIZATION_RANGES.items() if k in raw}
