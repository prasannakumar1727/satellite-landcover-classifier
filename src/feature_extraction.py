"""
feature_extraction.py
---------------------
Stage 6: hand-engineered features measured on the uploaded image only.

Radiometric measurements (colour, brightness, histogram, texture amplitude)
are taken from the NOISE-REDUCED image, because contrast enhancement
deliberately changes absolute brightness (it would brighten dark water and
amplify noise).  Structural measurements (edges, orientation, patches) are
taken from the ENHANCED + SHARPENED image, where boundaries are clearest and
global contrast is normalised, so fixed edge thresholds are meaningful.

Feature families
  A colour / spectral-like : RGB & HSV statistics, chromaticity, Excess-Green, mask proportions
  B histogram              : entropies, dynamic range, peak dominance
  C texture                : GLCM (Haralick), local std, local entropy
  D edge / spatial         : Canny density, Sobel gradient, orientation, spatial variation
  E morphological          : cleaned-mask proportions, continuity, patch analysis

NOTE: RGB has no near-infrared band, so NDVI is NOT computed.
"""
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.filters.rank import entropy as rank_entropy
from skimage.morphology import disk

from .config import PROCESSING as P
from . import segmentation as seg
from . import morphology as morph


def _entropy(hist: np.ndarray) -> float:
    p = hist.astype(np.float64)
    p = p[p > 0] / max(p.sum(), 1e-9)
    return float(-(p * np.log2(p)).sum())


# -------------------------------------------------------------- A. colour
def colour_features(m: dict) -> dict:
    f = m["f"]
    d = {}
    for i, c in enumerate("rgb"):
        d[f"mean_{c}"] = float(f[:, :, i].mean())
        d[f"std_{c}"] = float(f[:, :, i].std())
    d["chrom_r"], d["chrom_g"], d["chrom_b"] = (float(m[k].mean()) for k in ("cr", "cg", "cb"))
    d["exg_mean"] = float(m["exg"].mean())
    d["green_dominance"] = float((m["cg"] - np.maximum(m["cr"], m["cb"])).mean())
    d["blue_dominance"] = float((m["cb"] - np.maximum(m["cr"], m["cg"])).mean())
    rad = np.deg2rad(m["hue"])
    w = m["sat"]
    d["hue_mean_deg"] = float(np.rad2deg(np.arctan2((w * np.sin(rad)).mean(),
                                                    (w * np.cos(rad)).mean())) % 360)
    d["sat_mean"], d["sat_std"] = float(m["sat"].mean()), float(m["sat"].std())
    d["val_mean"], d["val_std"] = float(m["val"].mean()), float(m["val"].std())
    return d


# ------------------------------------------------------------ B. histogram
def histogram_features(gray: np.ndarray, m: dict) -> dict:
    gh = np.bincount(gray.ravel(), minlength=256)
    hue_h, _ = np.histogram(m["hue"], bins=36, range=(0, 360), weights=m["sat"])
    sat_h, _ = np.histogram(m["sat"], bins=32, range=(0, 1))
    p5, p95 = np.percentile(gray, (5, 95))
    coarse = gh.reshape(32, 8).sum(axis=1)
    return {"gray_entropy": _entropy(gh), "hue_entropy": _entropy(hue_h),
            "sat_entropy": _entropy(sat_h), "dynamic_range": float(p95 - p5),
            "gray_peak": float(coarse.max() / coarse.sum())}


# --------------------------------------------------------------- C. texture
def texture_features(gray: np.ndarray, lstd: np.ndarray) -> dict:
    q = (gray.astype(np.uint16) * P["glcm_levels"] // 256).astype(np.uint8)
    glcm = graycomatrix(q, [1], list(P["glcm_angles"]), levels=P["glcm_levels"],
                        symmetric=True, normed=True)
    d = {f"glcm_{p}": float(graycoprops(glcm, p).mean())
         for p in ("contrast", "correlation", "energy", "homogeneity")}
    d["local_std_mean"] = float(lstd.mean())
    d["local_entropy"] = float(rank_entropy(gray, disk(P["entropy_disk_radius"])).mean())
    return d


# ---------------------------------------------------------- D. edge / spatial
def edge_features(gray_struct: np.ndarray) -> tuple:
    g = cv2.GaussianBlur(gray_struct, (0, 0), P["canny_blur_sigma"])
    edges = cv2.Canny(g, P["canny_low"], P["canny_high"])
    gf = g.astype(np.float32)
    gx = cv2.Sobel(gf, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(gf, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    mag = np.hypot(gx, gy)
    d = {"canny_density": float((edges > 0).mean()),
         "sobel_mean": float(mag.mean()), "sobel_std": float(mag.std())}

    # orientation statistics on edge pixels (angle folded to [0, 180))
    ys, xs = np.nonzero(edges)
    if len(ys) >= 50:
        ang = np.arctan2(gy[ys, xs], gx[ys, xs]) % np.pi
        wgt = mag[ys, xs]
        hist, _ = np.histogram(ang, bins=P["orientation_bins"], range=(0, np.pi), weights=wgt)
        d["orientation_entropy"] = float(_entropy(hist) / np.log2(P["orientation_bins"]))
        c, s = (wgt * np.cos(2 * ang)).sum(), (wgt * np.sin(2 * ang)).sum()
        theta0 = 0.5 * np.arctan2(s, c)
        dev = np.abs(((ang - theta0) + np.pi / 2) % np.pi - np.pi / 2)          # to main axis
        dev2 = np.abs(((ang - theta0 - np.pi / 2) + np.pi / 2) % np.pi - np.pi / 2)  # to orthogonal
        tol = np.deg2rad(P["rect_tolerance_deg"])
        share = float((wgt * ((dev < tol) | (dev2 < tol))).sum() / wgt.sum())
        chance = 4 * P["rect_tolerance_deg"] / 180.0            # share expected for random angles
        d["rectilinearity"] = float(np.clip((share - chance) / (1 - chance), 0, 1))
        d["dominant_orientation_deg"] = float(np.rad2deg(theta0) % 180)
    else:                                                        # too few edges: no structure
        d.update(orientation_entropy=1.0, rectilinearity=0.0, dominant_orientation_deg=0.0)
    return d, edges


def spatial_features(gray: np.ndarray, lstd: np.ndarray) -> dict:
    n = P["block_grid"]
    h, w = gray.shape
    means, flat = [], []
    for i in range(n):
        for j in range(n):
            sl = (slice(i * h // n, (i + 1) * h // n), slice(j * w // n, (j + 1) * w // n))
            means.append(gray[sl].mean() / 255.0)
            flat.append(lstd[sl].mean() < 4.0)
    return {"spatial_variation": float(np.std(means)), "block_uniformity": float(np.mean(flat))}


# ------------------------------------------------------------- orchestrator
def extract_features(denoised: np.ndarray, structure: np.ndarray) -> tuple:
    """
    denoised  : noise-reduced RGB  (radiometric measurements)
    structure : enhanced + sharpened RGB (edge / boundary measurements)
    Returns (features: dict[str, float], aux: dict of maps/masks for visualisation)
    """
    m = seg.colour_maps(denoised)
    gray_d = cv2.cvtColor(denoised, cv2.COLOR_RGB2GRAY)
    gray_s = cv2.cvtColor(structure, cv2.COLOR_RGB2GRAY)
    lstd = seg.local_std(gray_d, P["local_window"])

    feats = colour_features(m)
    feats.update(histogram_features(gray_d, m))
    feats.update(texture_features(gray_d, lstd))
    edge_d, edges = edge_features(gray_s)
    feats.update(edge_d)
    feats.update(spatial_features(gray_d, lstd))

    # ---- segmentation masks + morphological region analysis
    win = P["edge_density_window"]
    local_edge = cv2.boxFilter((edges > 0).astype(np.float32), -1, (win, win))
    veg, water = seg.vegetation_mask(m), seg.water_mask(m, lstd)
    soil, gray_m = seg.soil_mask(m), seg.gray_mask(m)
    built = seg.builtup_mask(gray_m, local_edge, veg, water)

    feats.update(veg_frac=float(veg.mean()), water_frac=float(water.mean()),
                 soil_frac=float(soil.mean()), gray_frac=float(gray_m.mean()),
                 veg_soil_frac=float(veg.mean() + soil.mean()))

    veg_c, water_c, built_c = (morph.clean_mask(x) for x in (veg, water, built))
    soil_c = morph.clean_mask(soil)
    vs, ws, bs = morph.region_stats(veg_c), morph.region_stats(water_c), morph.region_stats(built_c)
    feats.update(veg_region_frac=vs["fraction"], veg_continuity=vs["continuity"],
                 veg_regions=vs["count"], water_region_frac=ws["fraction"],
                 water_continuity=ws["continuity"], water_regions=ws["count"],
                 builtup_region_frac=bs["fraction"], builtup_regions=bs["count"])

    patches = morph.patch_analysis(seg.flat_mask(gray_d), denoised)
    feats.update(flat_fraction=patches["flat_fraction"], patch_density=patches["patch_density"],
                 patch_color_div=patches["patch_color_div"], n_patches=patches["n_patches"])

    aux = {"edges": edges, "local_std": lstd, "veg": veg_c, "water": water_c,
           "soil": soil_c, "built": built_c, "patch_labels": patches["labels"]}
    return feats, aux


# ----------------------------------------------- UI grouping (label, key, format)
FEATURE_GROUPS = {
    "COLOR": [("Green dominance", "green_dominance", "{:.3f}"),
              ("Blue dominance", "blue_dominance", "{:.3f}"),
              ("Mean saturation", "sat_mean", "{:.2f}"),
              ("Mean brightness", "val_mean", "{:.2f}"),
              ("Vegetation-like area", "veg_frac", "{:.0%}"),
              ("Water-like area", "water_frac", "{:.0%}")],
    "HISTOGRAM": [("Gray entropy (bits)", "gray_entropy", "{:.2f}"),
                  ("Hue entropy (bits)", "hue_entropy", "{:.2f}"),
                  ("Saturation entropy", "sat_entropy", "{:.2f}"),
                  ("Dynamic range (p5-p95)", "dynamic_range", "{:.0f}")],
    "TEXTURE": [("GLCM contrast", "glcm_contrast", "{:.2f}"),
                ("GLCM energy", "glcm_energy", "{:.3f}"),
                ("GLCM homogeneity", "glcm_homogeneity", "{:.2f}"),
                ("Local variation (σ)", "local_std_mean", "{:.1f}"),
                ("Texture entropy", "local_entropy", "{:.2f}")],
    "SPATIAL": [("Edge density", "canny_density", "{:.3f}"),
                ("Mean gradient", "sobel_mean", "{:.1f}"),
                ("Rectilinearity", "rectilinearity", "{:.2f}"),
                ("Uniform blocks", "block_uniformity", "{:.0%}"),
                ("Homogeneous patches", "n_patches", "{:.0f}"),
                ("Vegetation continuity", "veg_continuity", "{:.2f}")],
}
