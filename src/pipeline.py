"""
pipeline.py
-----------
Runs the complete dataset-free classical DIP pipeline for ONE image:

  1 validate -> 2 resize -> 3 median filter -> 4 contrast enhancement -> 5 sharpen
  -> 6 feature extraction -> 7 normalisation -> 8 rule-based classification
  -> 9 result + explanation

Deterministic: no randomness anywhere.  CLI usage:
    python -m src.pipeline path/to/image.jpg
"""
import os
import sys
import time

import cv2
import numpy as np

from .config import OUTPUT_DIR, PROCESSING as P, RESOLUTION_OPTIONS
from .preprocessing import load_image, resize_image, denoise, LoadedImage
from .enhancement import enhance_contrast, sharpen, gray_histogram, channel_histograms
from .feature_extraction import extract_features
from .classifier import classify_image
from . import visualization as viz

DEFAULT_SETTINGS = {"resolution": RESOLUTION_OPTIONS["Auto (384 px)"], "median_kernel": 3,
                    "contrast": "CLAHE", "sharpen": True}


def analyze(loaded: LoadedImage, settings: dict = None) -> dict:
    s = {**DEFAULT_SETTINGS, **(settings or {})}
    t = {}
    clock = time.perf_counter

    t0 = clock()
    resized = resize_image(loaded.rgb, s["resolution"])
    t["resize"] = clock() - t0
    t0 = clock()
    denoised = denoise(resized, s["median_kernel"])
    t["denoise"] = clock() - t0
    t0 = clock()
    enhanced = enhance_contrast(denoised, s["contrast"])
    t["enhance"] = clock() - t0
    t0 = clock()
    sharpened = sharpen(enhanced, s["sharpen"])
    t["sharpen"] = clock() - t0

    t0 = clock()
    features, aux = extract_features(denoised, sharpened)
    t["features"] = clock() - t0

    small = min(loaded.orig_height, loaded.orig_width) < P["small_warn_px"]
    t0 = clock()
    result = classify_image(features, {"is_grayscale": loaded.is_grayscale, "small_image": small})
    t["classify"] = clock() - t0

    stages = [
        {"id": "01", "name": "Original", "image": loaded.rgb,
         "text": f"Uploaded image, {loaded.orig_width}×{loaded.orig_height} px."},
        {"id": "02", "name": "Resized", "image": resized,
         "text": f"Resampled to {resized.shape[1]}×{resized.shape[0]} px (aspect ratio kept)."},
        {"id": "03", "name": "Noise Reduced", "image": denoised,
         "text": f"{s['median_kernel']}×{s['median_kernel']} median filter removes impulse noise, keeps edges."},
        {"id": "04", "name": "Histogram Enhanced", "image": enhanced,
         "text": ("CLAHE on luminance: local contrast with limited noise gain." if s["contrast"] == "CLAHE"
                  else "Global histogram equalisation of luminance.")},
        {"id": "05", "name": "Sharpened", "image": sharpened,
         "text": ("Mild unsharp mask emphasises boundaries." if s["sharpen"]
                  else "Sharpening disabled - image unchanged.")},
    ]
    return {
        "loaded": loaded, "settings": s, "resized": resized, "denoised": denoised,
        "enhanced": enhanced, "sharpened": sharpened, "stages": stages,
        "hist_before": {"gray": gray_histogram(denoised), "rgb": channel_histograms(denoised)},
        "hist_after": {"gray": gray_histogram(enhanced), "rgb": channel_histograms(enhanced)},
        "features": features, "aux": aux, "result": result, "timings": t,
        "region_overlay": viz.region_overlay(denoised, aux),
        "edge_map": aux["edges"], "texture_map": viz.texture_map(aux["local_std"]),
    }


def run_pipeline(data: bytes, filename: str = "image", settings: dict = None) -> dict:
    """Validate the upload, then analyse.  Raises ImageValidationError for bad files."""
    return analyze(load_image(data, filename), settings)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python -m src.pipeline <image>")
    with open(sys.argv[1], "rb") as fh:
        out = run_pipeline(fh.read(), os.path.basename(sys.argv[1]))
    r = out["result"]
    print(f"Predicted: {r['predicted_class']}   rule-based confidence: {r['confidence']:.0%}"
          f"{'  (LOW CONFIDENCE)' if r['low_confidence'] else ''}")
    for c, v in sorted(r["scores"].items(), key=lambda kv: -kv[1]):
        print(f"  {c:<12}{v:5.1f}")
    for kind, text in r["reasons"]:
        print(f"  [{kind}] {text}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for st in out["stages"]:
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"stage_{st['id']}_{st['name'].replace(' ', '_')}.png"),
                    cv2.cvtColor(st["image"], cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(OUTPUT_DIR, "stage_06_regions.png"),
                cv2.cvtColor(out["region_overlay"], cv2.COLOR_RGB2BGR))
    print(f"Stage images saved to {OUTPUT_DIR}")
