"""
decision_rules.py
-----------------
The transparent scoring rules.  Each class score is

        score(class) = 100 * sum_i  weight_i * term_i          (term_i in [0, 1])

Weights come from config.CLASSIFICATION_CONFIG (each class sums to 1).
Every term carries a human-readable sentence built from the *measured* values,
which is what the "Why this classification?" panel shows.

Design notes (why each term exists)
  WATER        water is defined mainly by colour; smoothness / low edges / no vegetation
               are only credited in proportion to the water colour evidence ("gated"),
               otherwise a smooth green meadow would look like water.
  FOREST       continuous vegetation cover with fine, irregular canopy texture and no
               field-like patch structure.
  AGRICULTURE  vegetation/soil colours arranged as large homogeneous patches with straight
               boundaries, and neighbouring patches differing in tone.
  URBAN        dense, straight edges, built-up colours, high local contrast, many small patches,
               little vegetation.
"""
from .config import CLASSIFICATION_CONFIG as C

W = C["weights"]


def bell(x: float, lo: float, peak: float, hi: float) -> float:
    """Triangular 'moderate is best' function on [lo, hi] peaking at `peak`."""
    if x <= lo or x >= hi:
        return 0.0
    return (x - lo) / (peak - lo) if x <= peak else (hi - x) / (hi - peak)


def _term(cls, name, value, pos, neg):
    return {"name": name, "weight": W[cls][name], "value": float(max(0.0, min(1.0, value))),
            "pos": pos, "neg": neg}


def water_terms(n, r):
    g = n["water_frac"]                       # gate: structural water evidence only counts with colour
    return [
        _term("Water", "water_color", g,
              "Water-like colour covers {water_frac:.0%} of the image",
              "Very little water-like colour ({water_frac:.0%})"),
        _term("Water", "water_continuity", n["water_continuity"] * g,
              "Water pixels form one large connected body",
              "No large connected water body"),
        _term("Water", "smooth_surface",
              (0.5 * (1 - n["local_std_mean"]) + 0.5 * n["glcm_homogeneity"]) * g,
              "Surface is very smooth (local variation σ = {local_std_mean:.1f}, homogeneity {glcm_homogeneity:.2f})",
              "Surface is textured (local variation σ = {local_std_mean:.1f})"),
        _term("Water", "low_edges", (1 - n["canny_density"]) * g,
              "Almost no edges (edge density {canny_density:.3f})",
              "Too many edges for open water (edge density {canny_density:.3f})"),
        _term("Water", "not_vegetated", (1 - n["veg_frac"]) * g,
              "No vegetation-like colour response",
              "Vegetation-like colour present ({veg_frac:.0%})"),
    ]


def forest_terms(n, r):
    return [
        _term("Forest", "vegetation_cover", n["veg_frac"],
              "Vegetation-like colour covers {veg_frac:.0%} of the image",
              "Limited vegetation-like colour ({veg_frac:.0%})"),
        _term("Forest", "green_dominance", n["green_dominance"],
              "Strong green dominance (green chromaticity exceeds red/blue by {green_dominance:.3f})",
              "Weak green dominance ({green_dominance:.3f})"),
        _term("Forest", "veg_continuity", n["veg_continuity"] * n["veg_frac"],
              "Vegetation forms a continuous region ({veg_continuity:.0%} in its largest piece)",
              "Vegetation is fragmented into separate pieces"),
        _term("Forest", "natural_texture", 0.5 * n["local_entropy"] + 0.5 * n["local_std_mean"],
              "Fine-grained natural canopy texture (texture entropy {local_entropy:.2f})",
              "Texture is too smooth for a forest canopy"),
        _term("Forest", "irregular_geometry", 1 - n["rectilinearity"],
              "Edges are irregular, not geometric (rectilinearity {rectilinearity:.2f})",
              "Edges are geometric/straight (rectilinearity {rectilinearity:.2f})"),
        _term("Forest", "few_field_patches", 1 - n["flat_fraction"],
              "Few smooth field-like patches ({flat_fraction:.0%} flat area)",
              "Large smooth patches present ({flat_fraction:.0%} flat area)"),
        _term("Forest", "canopy_darkness", 1 - n["val_mean"],
              "Dark canopy-like brightness (mean brightness {val_mean:.2f})",
              "Too bright for dense canopy (mean brightness {val_mean:.2f})"),
        _term("Forest", "no_bare_soil", 1 - n["soil_frac"],
              "Little bare-soil colour ({soil_frac:.0%})",
              "Noticeable bare-soil colour ({soil_frac:.0%})"),
    ]


def agriculture_terms(n, r):
    veg, soil = r["veg_frac"], r["soil_frac"]
    share = soil / (veg + soil) if veg + soil > 1e-6 else 0.0
    bt = C["bell"]
    return [
        _term("Agriculture", "cultivated_color", n["veg_soil_frac"],
              "Vegetation and bare-soil colours together cover {veg_soil_frac:.0%} of the image",
              "Few crop / soil colours ({veg_soil_frac:.0%})"),
        _term("Agriculture", "color_mixture", 4 * share * (1 - share) * n["veg_soil_frac"],
              "Mixture of green crops and bare soil (patchwork colouring)",
              "No crop / soil colour mixture"),
        _term("Agriculture", "field_patches", n["flat_fraction"],
              "Large homogeneous field-like patches ({flat_fraction:.0%} of area, {n_patches:.0f} patches)",
              "Few homogeneous field-like patches ({flat_fraction:.0%} of area)"),
        _term("Agriculture", "field_texture", bell(n["local_std_mean"], *bt["field_texture"]),
              "Moderate texture variation typical of cultivated land (σ = {local_std_mean:.1f})",
              "Texture is too smooth or too rough for fields (σ = {local_std_mean:.1f})"),
        _term("Agriculture", "regular_geometry", n["rectilinearity"],
              "Straight, regular boundaries (rectilinearity {rectilinearity:.2f})",
              "Boundaries are not geometrically regular (rectilinearity {rectilinearity:.2f})"),
        _term("Agriculture", "patch_color_var", n["patch_color_div"],
              "Neighbouring patches differ in tone (patch colour diversity {patch_color_div:.3f})",
              "Patches have similar tone (diversity {patch_color_div:.3f})"),
        _term("Agriculture", "boundary_edges", bell(n["canny_density"], *bt["boundary_edges"]),
              "Moderate edge density consistent with field boundaries ({canny_density:.3f})",
              "Edge density ({canny_density:.3f}) is atypical for field boundaries"),
    ]


def urban_terms(n, r):
    return [
        _term("Urban", "edge_density", n["canny_density"],
              "High edge density ({canny_density:.3f})",
              "Low edge density ({canny_density:.3f})"),
        _term("Urban", "structural_geometry", n["rectilinearity"],
              "Strong geometric edge structure (rectilinearity {rectilinearity:.2f})",
              "Weak geometric structure (rectilinearity {rectilinearity:.2f})"),
        _term("Urban", "builtup_cover", n["builtup_region_frac"],
              "Built-up-like regions cover {builtup_region_frac:.0%} of the image",
              "Little built-up-like area ({builtup_region_frac:.0%})"),
        _term("Urban", "local_contrast", 0.5 * n["val_std"] + 0.5 * n["glcm_contrast"],
              "High local contrast (GLCM contrast {glcm_contrast:.1f})",
              "Low local contrast (GLCM contrast {glcm_contrast:.1f})"),
        _term("Urban", "fragmented_patches", n["patch_density"],
              "Many small separate patches ({n_patches:.0f} regions)",
              "Few separate patches ({n_patches:.0f} regions)"),
        _term("Urban", "low_vegetation", 1 - n["veg_frac"],
              "Not vegetation-dominated (vegetation-like colour {veg_frac:.0%})",
              "Vegetation-dominated appearance ({veg_frac:.0%})"),
        _term("Urban", "low_saturation", 1 - n["sat_mean"],
              "Low colour saturation (mean {sat_mean:.2f}), typical of concrete/asphalt",
              "Highly saturated colours (mean {sat_mean:.2f})"),
    ]


RULES = {"Agriculture": agriculture_terms, "Forest": forest_terms,
         "Urban": urban_terms, "Water": water_terms}


def score_class(cls: str, norm: dict, raw: dict) -> tuple:
    """Return (score 0-100, list of weighted terms) for one class."""
    terms = RULES[cls](norm, raw)
    return 100.0 * sum(t["weight"] * t["value"] for t in terms), terms
