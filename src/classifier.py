"""
classifier.py
-------------
Deterministic, dataset-free classifier:  features -> four class scores -> decision.

    result = classify_image(features)

Confidence is a "rule-based feature-agreement score", NOT a calibrated
probability:
      confidence = w_margin   * min(1, (top - runner_up) / margin_full)
                 + w_strength * min(1, top / strong_score)
i.e. it is high only when (a) the winning class is clearly ahead of the
runner-up AND (b) the winning class is itself well supported by evidence.
"""
from .config import CLASSES, CLASSIFICATION_CONFIG as C
from .feature_normalization import normalize_features
from .decision_rules import score_class
from .explanation import build_reasons, build_reason_details


def classify_image(features: dict, context: dict = None) -> dict:
    """
    features : raw feature dict from feature_extraction.extract_features
    context  : optional flags {"is_grayscale": bool, "small_image": bool}
    """
    context = context or {}
    cc = C["confidence"]
    norm = normalize_features(features)

    scores, terms = {}, {}
    for cls in CLASSES:
        scores[cls], terms[cls] = score_class(cls, norm, features)

    ranked = sorted(CLASSES, key=lambda c: (-scores[c], c))    # deterministic tie-break
    top, second = ranked[0], ranked[1]
    margin = scores[top] - scores[second]

    conf = (cc["w_margin"] * min(1.0, margin / cc["margin_full"]) +
            cc["w_strength"] * min(1.0, scores[top] / cc["strong_score"]))
    notes = []
    if context.get("is_grayscale"):
        conf *= cc["grayscale_penalty"]
        notes.append("Grayscale input: colour evidence unavailable, confidence reduced.")
    if context.get("small_image"):
        conf *= cc["small_image_penalty"]
        notes.append("Small input: texture/edge evidence less reliable, confidence reduced.")
    conf = max(cc["min_confidence"], min(cc["max_confidence"], conf))

    low = conf < cc["low_conf_threshold"] or margin < cc["low_margin_points"]
    return {
        "predicted_class": top,
        "runner_up": second,
        "confidence": conf,
        "low_confidence": bool(low),
        "margin": margin,
        "scores": scores,
        "terms": terms,
        "normalized": norm,
        "notes": notes,
        "reasons": build_reasons(top, second, terms, features, norm, low),
        "reason_details": build_reason_details(top, terms, features, low),
    }
