# Satellite Land-Cover Classifier
### Classical Digital Image Processing Based Land-Cover Classifier

A single-image satellite/aerial land-cover classifier that decides between
**Agriculture, Forest, Urban and Water** using only classical Digital Image
Processing (DIP) techniques — no machine learning, no dataset.

## 1. Problem Statement
Given one RGB satellite/aerial image, determine which of four land-cover
classes it most closely represents, using measurable image properties rather
than a trained model.

## 2. Objective
Demonstrate a complete DIP pipeline — acquisition, preprocessing, enhancement,
feature extraction, normalisation and rule-based decision-making — end to end
in an interactive Streamlit app.

## 3. Why No Dataset
The brief requires the classifier to work from the uploaded image alone.
There is **no EuroSAT, no training data, no kNN, no nearest-neighbour
comparison, no pretrained/neural network**. Every number shown in the UI is
computed live from the single image the user uploads.

## 4. System Architecture
```
app.py                      Streamlit UI
src/
  config.py                 all thresholds/weights (single source of truth)
  preprocessing.py           1. validation  2. resize  3. median denoise
  enhancement.py              4. CLAHE / hist-eq   5. unsharp mask
  segmentation.py            colour-rule masks (vegetation/water/soil/built-up)
  morphology.py              opening/closing, connected-component region stats
  feature_extraction.py      6. colour, histogram, texture, edge/spatial features
  feature_normalization.py   7. fixed-range min-max scaling
  decision_rules.py          per-class weighted scoring terms
  classifier.py              8. combines scores -> prediction + confidence
  explanation.py             9. builds "why this classification" sentences
  visualization.py           histogram / region-overlay / feature-bar plots
  pipeline.py                orchestrates all stages; also runnable as a CLI
tests/
  synthetic.py               procedurally generated demo scenes (NOT a dataset)
  smoke_test.py               end-to-end sanity check
```

## 5. DIP Pipeline
```
Upload → Validate → Resize → Median Denoise → CLAHE/Hist-Eq → Unsharp Mask
       → Feature Extraction → Normalisation → Rule-Based Scoring
       → Prediction + Confidence + Explanation
```

## 6. Preprocessing
- **Resize** (`cv2.INTER_AREA`/`INTER_CUBIC`) to a controlled resolution, aspect ratio kept.
- **Median filter** removes impulsive noise while preserving edges (unlike a Gaussian blur).

## 7. Histogram Enhancement
CLAHE (default) or global histogram equalisation, applied to the **luminance**
channel only (LAB-L or YCrCb-Y) so hue/saturation are not distorted.

## 8. Feature Extraction (Stage 6)
| Group | Examples |
|---|---|
| Colour / spectral-like | RGB & HSV means/std, chromaticity, Excess-Green index, green/blue dominance |
| Histogram | grayscale/hue/saturation entropy, dynamic range |
| Texture | GLCM contrast/energy/homogeneity/correlation, local std, local entropy |
| Edge / spatial | Canny density, Sobel gradient, edge-orientation rectilinearity, block uniformity |
| Morphological | cleaned vegetation/water/soil/built-up masks: area, continuity, patch count |

Radiometric measurements are taken on the **denoised** image (enhancement
changes absolute brightness); structural measurements use the
**enhanced+sharpened** image, where boundaries are clearest.

**RGB vs multispectral:** the app never computes NDVI or treats the green
channel as near-infrared — that requires an actual NIR band, which an RGB
photo does not have. Vegetation is measured with the RGB-only Excess-Green
index and HSV statistics instead.

## 9. Segmentation
Per-pixel colour rules (config-driven thresholds) build candidate masks:
vegetation (Excess-Green + HSV), water (blue-dominant or dark+flat), bare
soil, and built-up/grey. These are visual/explanatory aids and direct inputs
to the morphological + patch features — not a hidden classifier.

## 10. Morphological Processing
Opening removes speckle noise from a mask, closing fills small gaps.
Connected-component analysis then gives area fraction, largest-region share
("continuity") and region count ("fragmentation") for each mask.

## 11. Rule-Based Classification
```
score(class) = 100 * Σ weight_i * term_i        (term_i ∈ [0,1])
```
Each class combines several evidence terms (e.g. Agriculture = cultivated
colour + field patches + regular boundaries + patch-to-patch colour
variation + moderate texture). Weights and reference ranges live in
`src/config.py` — nothing is scattered as magic numbers in the code.

## 12. Confidence Calculation
```
confidence = w_margin   * min(1, (top_score − runner_up_score) / margin_full)
           + w_strength * min(1, top_score / strong_score)
```
capped below 97% (never a fake 100%), reduced for grayscale or very small
images, and reported as **"Rule-based confidence"** — a feature-agreement
score, not a calibrated ML probability. When the margin is small the UI shows
*"Low confidence — image characteristics overlap multiple land-cover
classes."*

## 13. UI
Sidebar: upload + processing settings (resolution / median kernel /
CLAHI vs hist-eq / sharpening) + class list. Main page: input image →
processing-stage gallery + histograms → segmentation/texture/edge views →
feature cards → classification (bar chart + result card) → feature-based
"Why this classification?" reasons → processing-summary checklist.

## 14. Installation
```bash
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

## 15. How to Run
```bash
streamlit run app.py
```
Or from VS Code: **Run and Debug → "Run: Streamlit app"**.

## 16. Example Workflow
Upload a JPG/PNG/TIFF aerial or satellite image → the app validates it →
runs all 9 stages → shows the predicted class, confidence and reasons.

## 17. Limitations
This is an educational classical DIP classifier. Its prediction is based on
handcrafted visual features and deterministic decision rules, **not**
supervised machine learning — it will not generalise as well as a trained
remote-sensing model, and mixed/ambiguous scenes may get a low-confidence or
borderline result. That is expected and reported honestly, not hidden.

## 18. Viva Q&A
**Q: Where is your dataset?** There is no training dataset. Classification
uses handcrafted image features and deterministic decision rules computed
directly from the uploaded image.
**Q: How do you classify without training?** By measuring colour, histogram,
texture, edge and spatial characteristics and combining them into
class-specific weighted scores.
**Q: Is this AI?** No trained deep-learning model — a classical image-analysis
and rule-based decision system.
**Q: What features do you use?** RGB/HSV colour statistics, histogram
entropy, GLCM texture, edge density/orientation, and morphological region
statistics.
**Q: Why not NDVI?** NDVI needs a near-infrared band; the input is RGB, so it
is never faked from the green channel.
**Q: What is your confidence?** A rule-based feature-agreement score, not a
calibrated ML probability.

## 19. Removed From the Previous Version
The earlier project loaded EuroSAT, extracted HSV+GLCM features per image,
and used a kNN classifier trained on 27,000 labelled tiles (`download_data.py`,
train/test split, `classifier.py` kNN, dataset-comparison/"most similar
images" UI). All of that has been removed per the "no dataset" requirement;
the HSV and GLCM texture ideas were kept and substantially extended into the
feature set above, computed only on the uploaded image.
