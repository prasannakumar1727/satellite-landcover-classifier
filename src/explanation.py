"""
explanation.py
--------------
Builds the "Why this classification?" sentences.  Every sentence is a template
filled with the actual measured values of the analysed image; which templates
are shown depends on the computed term values (nothing is hard-coded per image).
"""
from .config import CLASSIFICATION_CONFIG as C


class _Safe(dict):
    def __missing__(self, key):
        return float("nan")


def _fmt(template: str, raw: dict) -> str:
    return template.format_map(_Safe(raw))


def _actual_label(name: str, raw: dict) -> str:
    percent = {"vegetation_cover": "veg_frac", "veg_continuity": "veg_continuity",
               "few_field_patches": "flat_fraction", "no_bare_soil": "soil_frac",
               "water_color": "water_frac", "water_continuity": "water_continuity",
               "low_vegetation": "veg_frac", "builtup_cover": "builtup_region_frac"}
    decimal = {"green_dominance": "green_dominance", "irregular_geometry": "rectilinearity",
               "local_contrast": "glcm_contrast", "edge_density": "canny_density",
               "low_edges": "canny_density", "low_saturation": "sat_mean"}
    if name in percent:
        return f"Actual: {raw.get(percent[name], 0.0):.1%}"
    if name in decimal:
        return f"Actual: {raw.get(decimal[name], 0.0):.2f}"
    if name == "canopy_darkness":
        return f"Luma: {raw.get('val_mean', 0.0) * 255:.1f} DN"
    if name == "natural_texture":
        return f"Entropy: {raw.get('local_entropy', 0.0):.2f}"
    return f"Actual term: {raw.get(name, 0.0):.2f}"


def build_reasons(top: str, second: str, terms: dict, raw: dict, norm: dict, low: bool) -> list:
    """Return list of (kind, text): kind in {'support', 'against', 'info'}."""
    out = []
    ranked = sorted(terms[top], key=lambda t: -t["weight"] * t["value"])
    for t in ranked:
        if t["value"] >= C["reason_min_term"] and len(out) < C["max_reasons"]:
            out.append(("support", _fmt(t["pos"], raw)))
    if not out:                                   # nothing strongly supports: be honest
        best = ranked[0]
        out.append(("info", "No single feature strongly supports the winning class; strongest "
                            "evidence: " + _fmt(best["pos"], raw)))
    weak = sorted([t for t in terms[top] if t["value"] < 0.25],
                  key=lambda t: -t["weight"])
    if weak:
        out.append(("against", _fmt(weak[0]["neg"], raw)))
    if low:
        out.append(("info", f"Low confidence — image characteristics overlap multiple land-cover "
                            f"classes (closest alternative: {second})."))
    return out


def build_reason_details(top: str, terms: dict, raw: dict, low: bool) -> list:
    """Return live rule evidence for the UI decision trail."""
    details = []
    ranked = sorted(terms[top], key=lambda t: -t["weight"] * t["value"])
    for term in ranked:
        if term["value"] < C["reason_min_term"] or len(details) >= C["max_reasons"]:
            continue
        details.append({
            "kind": "support",
            "name": term["name"],
            "description": _fmt(term["pos"], raw),
            "actual": term["value"],
            "actual_label": _actual_label(term["name"], raw),
            "weight": term["weight"],
            "rule": f"Evidence >= {C['reason_min_term']:.2f}",
        })
    return details
