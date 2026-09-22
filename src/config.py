"""
config.py
---------
Single home for every tunable number in the project (no magic numbers
elsewhere).  Nothing here is learned from data: each value is a documented,
physically/visually motivated constant chosen by the designer.

Sections
  1. Classes and UI metadata
  2. Processing parameters (DIP stages)
  3. Segmentation / morphology parameters
  4. Feature normalisation ranges  (fixed "reference ranges", NOT dataset statistics)
  5. Classification weights, confidence formula
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# ------------------------------------------------------------------ 1. classes
CLASSES = ["Agriculture", "Forest", "Urban", "Water"]
CLASS_META = {
    "Agriculture": {"icon": "🌾", "color": "#E9B44C"},
    "Forest":      {"icon": "🌲", "color": "#3FA34D"},
    "Urban":       {"icon": "🏙️", "color": "#9AA5B8"},
    "Water":       {"icon": "🌊", "color": "#3A86FF"},
}

# ------------------------------------------------------- 2. processing settings
RESOLUTION_OPTIONS = {"Auto (384 px)": 384, "256 px": 256, "512 px": 512}
CONTRAST_OPTIONS = ["CLAHE", "Histogram Equalization"]
MEDIAN_KERNEL_OPTIONS = [3, 5]

PROCESSING = {
    "min_side_px": 32,             # below this an image carries too little structure -> rejected
    "small_warn_px": 128,          # below this texture/edge features are unreliable -> warning
    "max_decode_pixels": 120_000_000,
    "clahe_clip_limit": 2.0,       # low clip => limited noise amplification in flat regions
    "clahe_tile_grid": 8,
    "unsharp_sigma": 1.2,          # Gaussian radius of the blurred copy
    "unsharp_amount": 0.6,         # mild: out = img + amount * (img - blurred)
    "canny_low": 50,               # fixed thresholds are meaningful because the image was
    "canny_high": 130,             # contrast-normalised by the enhancement stage
    "canny_blur_sigma": 1.2,
    "glcm_levels": 32,             # grey-level quantisation for the co-occurrence matrix
    "glcm_angles": (0.0, 0.7853981634, 1.5707963268, 2.3561944902),
    "entropy_disk_radius": 3,
    "local_window": 5,             # window for local mean/variance
    "edge_density_window": 15,     # window for the local edge-density map
    "orientation_bins": 36,        # 5-degree bins over [0, 180)
    "rect_tolerance_deg": 10.0,    # +-tolerance around the two dominant (orthogonal) axes
    "block_grid": 8,               # NxN grid for spatial variation / uniformity
}

# ------------------------------------------------- 3. segmentation / morphology
SEGMENTATION = {
    "veg_exg_min": 0.05,           # Excess-Green  ExG = 2g - r - b  (chromatic coords)
    "veg_min_sat": 0.10,
    "veg_min_val": 0.08,
    "water_blue_margin": 0.03,     # B >= R + margin  (blue dominant)
    "water_blue_green_tol": 0.02,  # B >= G - tol
    "water_min_sat": 0.08,
    "water_max_val": 0.85,
    "water_dark_val": 0.35,        # dark-water branch (teal / deep / turbid water)
    "water_dark_min_sat": 0.12,
    "water_dark_max_exg": 0.10,
    "water_flat_std": 5.0,         # local grey-level std below which a pixel is "flat"
    "soil_min_sat": 0.15,
    "soil_min_val": 0.20,
    "gray_max_sat": 0.16,          # low-saturation (concrete / asphalt-like) pixels
    "gray_min_val": 0.22,
    "builtup_local_edge": 0.12,    # local edge density above which a non-vegetated pixel is "structured"
    "flat_gradient_max": 4.0,      # smoothed-gradient (grey levels / px) below which a pixel is "flat"
    "patch_min_area_frac": 0.003,  # smallest region counted as a homogeneous patch
    "morph_kernel": 5,
    "min_region_area_frac": 0.002,
}

# ---------------------------------------------------- 4. normalisation ranges
# (lo, hi): raw value <= lo -> 0.0, >= hi -> 1.0, linear in between.
# These are fixed reference ranges reasoned from the physical meaning of each
# measurement.  They are NOT computed from any collection of images.
NORMALIZATION_RANGES = {
    "veg_frac":            (0.10, 0.75),
    "veg_soil_frac":       (0.25, 0.80),
    "water_frac":          (0.10, 0.70),
    "soil_frac":           (0.05, 0.40),
    "gray_frac":           (0.10, 0.60),
    "builtup_region_frac": (0.10, 0.60),
    "green_dominance":     (0.00, 0.12),
    "sat_mean":            (0.08, 0.45),
    "val_mean":            (0.15, 0.65),
    "val_std":             (0.03, 0.20),
    "gray_entropy":        (4.5, 7.5),
    "local_std_mean":      (2.0, 22.0),
    "local_entropy":       (1.5, 4.8),
    "glcm_contrast":       (0.5, 18.0),
    "glcm_homogeneity":    (0.35, 0.90),
    "glcm_energy":         (0.01, 0.30),
    "canny_density":       (0.02, 0.20),
    "sobel_mean":          (2.0, 22.0),
    "rectilinearity":      (0.05, 0.45),
    "spatial_variation":   (0.03, 0.22),
    "block_uniformity":    (0.05, 0.90),
    "flat_fraction":       (0.10, 0.70),
    "patch_density":       (0.5, 6.0),     # homogeneous patches per 10 000 px
    "patch_color_div":     (0.02, 0.15),
    "veg_continuity":      (0.30, 0.90),
    "water_continuity":    (0.30, 0.90),
}

# ------------------------------------------------ 5. classification / confidence
# Every class score = 100 * sum(weight_i * term_i) with term_i in [0, 1] and
# weights summing to 1 per class.  Terms are defined in decision_rules.py.
CLASSIFICATION_CONFIG = {
    "weights": {
        "Agriculture": {
            "cultivated_color":  0.20,  # vegetation + bare-soil colours (crop / fallow mixture)
            "color_mixture":     0.05,  # both green crops and soil present -> patchwork
            "field_patches":     0.20,  # large homogeneous patches bounded by edges
            "field_texture":     0.15,  # smooth inside fields, moderate overall variation
            "regular_geometry":  0.15,  # straight field boundaries
            "patch_color_var":   0.15,  # neighbouring fields differ in tone (crop / season)
            "boundary_edges":    0.10,  # moderate edge density (boundaries, not dense built-up)
        },
        "Forest": {
            "vegetation_cover":  0.25,  # share of vegetation-like colour
            "green_dominance":   0.10,  # strength of the green response
            "veg_continuity":    0.15,  # one large connected vegetation region
            "natural_texture":   0.15,  # fine-grained canopy texture
            "irregular_geometry":0.10,  # low geometric regularity
            "few_field_patches": 0.10,  # no field-like homogeneous patches
            "canopy_darkness":   0.10,  # canopy is darker than crops / built-up
            "no_bare_soil":      0.05,
        },
        "Urban": {
            "edge_density":      0.25,
            "structural_geometry":0.10,
            "builtup_cover":     0.20,
            "local_contrast":    0.15,
            "fragmented_patches":0.10,
            "low_vegetation":    0.10,
            "low_saturation":    0.10,
        },
        "Water": {
            "water_color":       0.35,  # water-like colour proportion (primary evidence)
            "water_continuity":  0.15,  # one large connected water body
            "smooth_surface":    0.20,  # very low local variation   (credited only with water colour)
            "low_edges":         0.15,  # almost no edges             (credited only with water colour)
            "not_vegetated":     0.15,  # no vegetation response      (credited only with water colour)
        },
    },
    # bell-shaped ("moderate is best") terms: (low, peak, high) on the *normalised* value
    "bell": {
        "field_texture":  (0.05, 0.30, 0.75),
        "boundary_edges": (0.05, 0.35, 0.80),
    },
    # Confidence = w_margin * min(1, margin / margin_full) + w_strength * min(1, top / strong_score)
    "confidence": {
        "margin_full": 25.0,        # score-point lead of the winner regarded as fully decisive
        "strong_score": 70.0,       # winner score regarded as fully supported
        "w_margin": 0.55,
        "w_strength": 0.45,
        "max_confidence": 0.97,     # never display a fake 100 %
        "min_confidence": 0.05,
        "low_conf_threshold": 0.45,
        "low_margin_points": 6.0,   # winner closer than this to runner-up => ambiguous
        "grayscale_penalty": 0.6,   # colour evidence is absent for grayscale uploads
        "small_image_penalty": 0.85,
    },
    "reason_min_term": 0.55,        # a term must reach this value to be quoted as a "reason"
    "max_reasons": 5,
}
