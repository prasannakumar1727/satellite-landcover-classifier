"""
webapp/server.py - Flask backend for the website UI.
Wraps src/pipeline.run_pipeline() only. No classification/feature logic here.
"""
import os
import sys
import base64

import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import CLASSES, CLASS_META, RESOLUTION_OPTIONS, CONTRAST_OPTIONS
from src.preprocessing import ImageValidationError
from src.pipeline import run_pipeline
from src.feature_extraction import FEATURE_GROUPS

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0


def hist_json(h, bins=64):
    """Downsample a 256-bin histogram to `bins` values and return as a plain list."""
    gray = np.asarray(h["gray"], dtype=float).reshape(bins, -1).sum(axis=1)
    rgb = [np.asarray(c, dtype=float).reshape(bins, -1).sum(axis=1).tolist() for c in h["rgb"]]
    return {"gray": gray.tolist(), "rgb": {"r": rgb[0], "g": rgb[1], "b": rgb[2]}}


def img_b64(rgb_array):
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR))
    return "data:image/png;base64," + base64.b64encode(buf).decode()


@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/api/config")
def config():
    return jsonify({
        "classes": [{"name": c, "icon": CLASS_META[c]["icon"], "color": CLASS_META[c]["color"]} for c in CLASSES],
        "resolutions": list(RESOLUTION_OPTIONS.keys()),
        "kernels": [3, 5],
        "contrasts": CONTRAST_OPTIONS,
    })


@app.post("/api/analyze")
def analyze():
    f = request.files.get("image")
    if f is None:
        return jsonify({"error": "No image uploaded"}), 400

    image_bytes = f.read()
    settings = {
        "resolution": RESOLUTION_OPTIONS.get(request.form.get("resolution"), RESOLUTION_OPTIONS["Auto (384 px)"]),
        "median_kernel": int(request.form.get("kernel", 3)),
        "contrast": request.form.get("contrast", "CLAHE"),
        "sharpen": request.form.get("sharpen", "true") == "true",
    }

    try:
        out = run_pipeline(image_bytes, f.filename, settings)
    except ImageValidationError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {e}"}), 500

    loaded, feats, res = out["loaded"], out["features"], out["result"]

    feature_groups = []
    for group, items in FEATURE_GROUPS.items():
        if group in {"TEXTURE", "SPATIAL"}:
            continue
        feature_groups.append({
            "group": group,
            "items": [{"label": label, "value": fmt.format(feats[key])} for label, key, fmt in items],
        })

    return jsonify({
        "warnings": loaded.warnings,
        "input": {
            "image": img_b64(loaded.rgb),
            "filename": loaded.filename,
            "width": loaded.orig_width,
            "height": loaded.orig_height,
            "format": loaded.fmt,
            "channels": loaded.orig_channels,
            "bit_depth": loaded.bit_depth,
            "is_grayscale": loaded.is_grayscale,
            "size_bytes": len(image_bytes),
            "file_size": len(image_bytes),
        },
        "stages": [{"id": s["id"], "name": s["name"], "text": s["text"], "image": img_b64(s["image"])}
                   for s in out["stages"]],
        "hist_before": hist_json(out["hist_before"]),
        "hist_after": hist_json(out["hist_after"]),
        "region_overlay": img_b64(out["region_overlay"]),
        "texture_map": img_b64(cv2.cvtColor(out["texture_map"], cv2.COLOR_GRAY2RGB)
                                if out["texture_map"].ndim == 2 else out["texture_map"]),
        "edge_map": img_b64(cv2.cvtColor(out["edge_map"], cv2.COLOR_GRAY2RGB)
                             if out["edge_map"].ndim == 2 else out["edge_map"]),
        "feature_groups": feature_groups,
        "features_raw": {k: round(float(v), 4) for k, v in feats.items()},
        "result": {
            "predicted_class": res["predicted_class"],
            "icon": CLASS_META[res["predicted_class"]]["icon"],
            "confidence": res["confidence"],
            "low_confidence": res["low_confidence"],
            "scores": res["scores"],
            "reasons": res["reasons"],
            "reason_details": res["reason_details"],
        },
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
