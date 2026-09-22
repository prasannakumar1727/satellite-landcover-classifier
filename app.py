"""
app.py - Streamlit UI for the Satellite Land-Cover Classifier.

Classical Digital Image Processing • Dataset-Free Classification
No dataset, no training, no neural network: every result comes from
processing the single uploaded image.
"""
import streamlit as st

from src.config import CLASSES, CLASS_META, RESOLUTION_OPTIONS, CONTRAST_OPTIONS, OUTPUT_DIR
from src.preprocessing import ImageValidationError
from src.pipeline import run_pipeline
from src.visualization import histogram_figure, feature_bar_figure
from src.feature_extraction import FEATURE_GROUPS

st.set_page_config(page_title="Satellite Land-Cover Classifier", page_icon="🛰️", layout="wide")

CSS = """
<style>
@keyframes fadeUp { from{opacity:0; transform:translateY(10px);} to{opacity:1; transform:translateY(0);} }
@keyframes glow { 0%,100%{box-shadow:0 0 0 rgba(101,193,140,0);} 50%{box-shadow:0 0 26px rgba(101,193,140,.28);} }
@keyframes drift { 0%{background-position:0% 0%;} 100%{background-position:200% 0%;} }
:root { --bg:#070d14; --line:rgba(255,255,255,.09); --muted:#8ea0b3; --text:#edf4f8; --accent:#65c18c; --blue:#6daee8; --glass:rgba(255,255,255,.045); }
.stApp { color:var(--text);
  background:
    radial-gradient(1100px 550px at 12% -8%, rgba(109,174,232,.16), transparent 60%),
    radial-gradient(900px 500px at 100% 0%, rgba(101,193,140,.14), transparent 55%),
    var(--bg); }
.block-container { max-width:1500px; padding-top:1.4rem; padding-bottom:3rem; }
section[data-testid="stSidebar"] { background:rgba(8,14,22,.55); backdrop-filter:blur(18px) saturate(140%); -webkit-backdrop-filter:blur(18px) saturate(140%); border-right:1px solid var(--line); }
section[data-testid="stSidebar"] > div { padding-top:1.5rem; }
.hero { display:flex; justify-content:space-between; align-items:flex-end; padding:10px 18px; margin-bottom:24px;
  background:var(--glass); border:1px solid var(--line); border-radius:16px; backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px);
  animation:fadeUp .5s ease both; }
.hero h1 { margin:0; font-size:1.8rem; letter-spacing:-.03em; color:#f3f7fa; }
.hero p { margin:6px 0 0; color:var(--muted); font-size:.9rem; }
.badge { border:1px solid rgba(101,193,140,.4); background:rgba(101,193,140,.12); color:#7ee2a6; padding:6px 10px; border-radius:999px;
  font-size:.72rem; font-weight:700; letter-spacing:.08em; animation:glow 2.6s ease-in-out infinite; }
.card { background:var(--glass); border:1px solid var(--line); border-radius:14px; padding:16px 18px; height:100%;
  backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px); box-shadow:0 8px 24px rgba(0,0,0,.22);
  transition:transform .25s ease, border-color .25s ease, box-shadow .25s ease; animation:fadeUp .5s ease both; }
.card:hover { transform:translateY(-3px); border-color:rgba(109,174,232,.35); box-shadow:0 14px 30px rgba(0,0,0,.3); }
.stage-num { color:var(--blue); font-weight:700; font-size:.72rem; letter-spacing:.08em; }
.result-card { position:relative; overflow:hidden; background:linear-gradient(135deg, rgba(101,193,140,.14), rgba(109,174,232,.08) 60%, rgba(255,255,255,.03));
  border:1px solid rgba(101,193,140,.35); border-radius:18px; padding:22px 24px;
  backdrop-filter:blur(16px) saturate(150%); -webkit-backdrop-filter:blur(16px) saturate(150%);
  box-shadow:0 18px 40px rgba(0,0,0,.3); animation:fadeUp .55s ease both; }
.result-card::before { content:""; position:absolute; inset:-2px; z-index:-1; border-radius:18px; opacity:.5;
  background:linear-gradient(120deg, rgba(101,193,140,.35), rgba(109,174,232,.35), rgba(101,193,140,.35)); background-size:200% 100%; animation:drift 6s linear infinite; }
.result-kicker { color:#a9c9bb; font-size:.72rem; font-weight:800; letter-spacing:.14em; margin-bottom:8px; }
.result-class { font-size:2.05rem; font-weight:850; letter-spacing:-.025em; display:flex; align-items:center; gap:10px; }
.result-icon { font-size:1.7rem; }
.confidence-row { display:flex; justify-content:space-between; align-items:center; margin-top:18px; color:#c3d3de; font-size:.84rem; }
.confidence-row strong { color:#effaf3; font-size:1.25rem; }
.confidence-track { height:7px; background:rgba(255,255,255,.08); border-radius:999px; overflow:hidden; margin-top:8px; }
.confidence-track span { display:block; height:100%; background:linear-gradient(90deg,#65c18c,#6daee8); border-radius:999px; transition:width .8s cubic-bezier(.22,1,.36,1); }
.result-note { margin-top:10px; color:#9fb3a8; font-size:.78rem; }
.info-box { background:var(--glass); border:1px solid var(--line); border-left:3px solid var(--blue); border-radius:10px; padding:11px 14px;
  font-size:.84rem; color:#c1cdd8; backdrop-filter:blur(12px); -webkit-backdrop-filter:blur(12px); }
.reason-yes { color:#87e0a6; margin:7px 0; transition:transform .15s; } .reason-yes:hover{transform:translateX(3px);}
.reason-no { color:#e79c9c; margin:7px 0; } .reason-info { color:#dfc57e; margin:7px 0; }
[data-testid="stMetric"] { background:var(--glass); border:1px solid var(--line); border-radius:12px; padding:10px; backdrop-filter:blur(12px); -webkit-backdrop-filter:blur(12px); }
h1,h2,h3 { letter-spacing:-.02em; }
[data-testid="stExpander"] { border:1px solid var(--line); border-radius:12px; background:var(--glass); backdrop-filter:blur(12px); -webkit-backdrop-filter:blur(12px); }
.stCaption { color:var(--muted); }
.stButton>button, .stDownloadButton>button { transition:transform .15s ease, box-shadow .15s ease; }
.stButton>button:hover, .stDownloadButton>button:hover { transform:translateY(-1px); box-shadow:0 6px 16px rgba(109,174,232,.25); }
img { transition:transform .3s ease; border-radius:10px; }
img:hover { transform:scale(1.015); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    '<div class="hero"><div><h1>🛰️ Satellite Land-Cover Classifier</h1>'
    '<p>Classical Digital Image Processing • Dataset-Free Classification</p></div>'
    '<div class="badge">● READY FOR ANALYSIS</div></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Project")
    upload = st.file_uploader("Upload Satellite Image", type=["jpg", "jpeg", "png", "tif", "tiff"])
    st.caption("Supported: JPG / JPEG / PNG / TIFF")

    st.markdown("### Processing Settings")
    res_label = st.selectbox("Image Resolution", list(RESOLUTION_OPTIONS.keys()), index=0)
    kernel = st.selectbox("Noise Filter (Median kernel)", [3, 5], index=0)
    contrast = st.selectbox("Contrast Enhancement", CONTRAST_OPTIONS, index=0)
    sharpen_on = st.toggle("Sharpening", value=True)

    st.markdown("### Classification Classes")
    for c in CLASSES:
        st.markdown(f"{CLASS_META[c]['icon']} {c}")

    st.markdown("---")
    st.markdown('<div class="info-box"><b>Dataset-Free Classification</b><br>'
                'This system does not train on or compare against an external image dataset. '
                'The land-cover decision is generated entirely from classical Digital Image '
                'Processing features extracted from the uploaded image.</div>',
                unsafe_allow_html=True)

if upload is None:
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown(
            '<div class="card" style="text-align:center; padding:48px 20px;">'
            '<h3>Upload a Satellite Image</h3>'
            '<p style="color:#8E9AAF;">Drop your image here or browse using the sidebar</p>'
            '<p style="color:#5C6779; font-size:0.85rem;">Supported formats: JPG, PNG, TIFF</p>'
            '</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div style="text-align:center; color:#8E9AAF; line-height:2.1;">'
            'Image → Preprocessing → Feature Extraction → Classification → Land Cover</div>',
            unsafe_allow_html=True)
    st.stop()

settings = {"resolution": RESOLUTION_OPTIONS[res_label], "median_kernel": kernel,
            "contrast": contrast, "sharpen": sharpen_on}

try:
    out = run_pipeline(upload.getvalue(), upload.name, settings)
except ImageValidationError as e:
    st.error(f"Could not process this image: {e}")
    st.stop()
except Exception as e:
    st.error("An unexpected error occurred while analysing this image. Please try a different file.")
    st.exception(e)
    st.stop()

loaded, feats, res = out["loaded"], out["features"], out["result"]

for w in loaded.warnings:
    st.warning(w)

# ---------- 1. Input Image
st.header("1. Input Image")
c1, c2 = st.columns([2, 1])
with c1:
    st.image(loaded.rgb, use_container_width=True)
with c2:
    st.markdown(f"""<div class="card">
    <p><b>Filename</b><br>{loaded.filename}</p>
    <p><b>Resolution</b><br>{loaded.orig_width} × {loaded.orig_height} px</p>
    <p><b>Format</b><br>{loaded.fmt}</p>
    <p><b>Channels</b><br>{loaded.orig_channels} ({loaded.bit_depth}{', grayscale' if loaded.is_grayscale else ', RGB'})</p>
    </div>""", unsafe_allow_html=True)

# ---------- 2. Digital Image Processing
st.header("2. Digital Image Processing")
cols = st.columns(len(out["stages"]))
for col, st_ in zip(cols, out["stages"]):
    with col:
        st.image(st_["image"], use_container_width=True)
        st.markdown(f'<span class="stage-num">{st_["id"]}</span> **{st_["name"]}**  \n'
                    f'<span style="font-size:0.82rem; color:#8E9AAF;">{st_["text"]}</span>',
                    unsafe_allow_html=True)

st.subheader("Histogram Comparison (grayscale + RGB)")
st.pyplot(histogram_figure(out["hist_before"], out["hist_after"],
                          f"After {contrast}"), use_container_width=True)

with st.expander("Segmentation & region analysis (Otsu-style colour masks + morphology)"):
    r1, r2, r3 = st.columns(3)
    r1.image(out["region_overlay"], caption="Vegetation / Water / Soil / Built-up regions",
              use_container_width=True)
    r2.image(out["texture_map"], caption="Local texture (bright = rough)", use_container_width=True)
    r3.image(out["edge_map"], caption="Canny edges", use_container_width=True, clamp=True)

# ---------- 3. Feature Extraction
st.header("3. Feature Extraction")
st.caption("RGB / HSV colour-based features — not true multispectral or NDVI data.")
visible_groups = [(g, items) for g, items in FEATURE_GROUPS.items() if g not in {"TEXTURE", "SPATIAL"}]
gcols = st.columns(len(visible_groups))
for col, (group, items) in zip(gcols, visible_groups):
    with col:
        st.markdown(f'<div class="card"><b>{group}</b><br><br>' + "<br>".join(
            f'{label}: <b>{fmt.format(feats[key])}</b>' for label, key, fmt in items
        ) + "</div>", unsafe_allow_html=True)

with st.expander("View detailed feature vector"):
    st.dataframe({"feature": list(feats.keys()), "value": [round(v, 4) for v in feats.values()]},
                use_container_width=True, hide_index=True)

# ---------- 4. Classification
st.header("4. Classification")
cc1, cc2 = st.columns([1, 1.4])
with cc1:
    meta = CLASS_META[res["predicted_class"]]
    conf_note = "Characteristics overlap multiple land-cover classes." if res["low_confidence"] else "Feature agreement is strong."
    st.markdown(f"""
    <div class="result-card">
      <div class="result-kicker">LAND-COVER DECISION</div>
      <div class="result-class"><span class="result-icon">{meta['icon']}</span>{res['predicted_class'].upper()}</div>
      <div class="confidence-row"><span>Rule-based confidence</span><strong>{res['confidence']:.0%}</strong></div>
      <div class="confidence-track"><span style="width:{res['confidence']*100:.0f}%"></span></div>
      <div class="result-note">{conf_note}</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="info-box" style="margin-top:10px;"><b>Interpretation</b><br>'
                'Confidence is a heuristic agreement score from the extracted image features; '
                'it is not a calibrated machine-learning probability.</div>', unsafe_allow_html=True)
with cc2:
    st.bar_chart({c: res["scores"][c] for c in CLASSES}, use_container_width=True)

# ---------- 5. Why this classification?
st.header("5. Why This Classification?")
icon = {"support": "✅", "against": "⚠️", "info": "ℹ️"}
css_cl = {"support": "reason-yes", "against": "reason-no", "info": "reason-info"}
for kind, text in res["reasons"]:
    st.markdown(f'<div class="{css_cl[kind]}">{icon[kind]} {text}</div>', unsafe_allow_html=True)

# ---------- Footer
st.markdown("---")
st.caption("Educational classical Digital Image Processing classifier. Predictions are based on "
          "handcrafted visual features and deterministic decision rules — not supervised machine "
          "learning, and not a trained neural network.")
