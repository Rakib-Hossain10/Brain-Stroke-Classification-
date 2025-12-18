# app.py
# ------------------------------------------------------------
# Brain Stroke Classification (ResNet50) - Streamlit Frontend
# Files required in same folder:
#   - BrainStroke_resnet50.keras
#   - BrainStroke_resnet50_metadata.json (optional; app will still run)
# ------------------------------------------------------------

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import altair as alt


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Brain Stroke Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Theme / CSS (Dark + Green)
# -----------------------------
CUSTOM_CSS = """
<style>
/* App background */
.stApp {
  background: radial-gradient(1200px 600px at 20% 10%, rgba(20, 60, 40, 0.25), transparent 55%),
              radial-gradient(900px 500px at 85% 20%, rgba(20, 120, 70, 0.18), transparent 60%),
              linear-gradient(180deg, #0b0f14 0%, #070a0f 100%);
  color: #e8eef7;
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0a0e13 0%, #070a0f 100%) !important;
  border-right: 1px solid rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] * {
  color: #e8eef7;
}

/* Global text */
h1, h2, h3, h4 { letter-spacing: -0.02em; color: #f2f6ff; }
p, div, span, label { color: #d7deea; }

/* Cards */
.card {
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  padding: 16px 16px;
  background: rgba(10, 14, 19, 0.72);
  box-shadow: 0 10px 24px rgba(0,0,0,0.35);
}

/* Hero */
.hero {
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  padding: 18px 18px;
  background: rgba(10, 14, 19, 0.72);
  box-shadow: 0 10px 24px rgba(0,0,0,0.35);
}
.hero-title { font-size: 2.05rem; font-weight: 800; margin: 0; }
.hero-sub { margin-top: .35rem; color: rgba(215, 222, 234, 0.9); line-height: 1.4; }

/* KPI */
.kpi {
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 16px;
  padding: 14px 14px;
  background: rgba(10, 14, 19, 0.72);
  box-shadow: 0 10px 24px rgba(0,0,0,0.35);
}
.kpi .label { color: rgba(215, 222, 234, 0.75); font-size: 0.85rem; margin-bottom: 4px; }
.kpi .value { font-size: 1.35rem; font-weight: 800; margin: 0; }

/* Buttons -> GREEN */
.stButton > button {
  width: 100%;
  border-radius: 14px !important;
  border: 1px solid rgba(64, 255, 170, 0.25) !important;
  background: linear-gradient(180deg, #0db36b 0%, #079a5c 100%) !important;
  color: #07110c !important;
  font-weight: 800 !important;
  padding: 0.70rem 1rem !important;
}
.stButton > button:hover {
  filter: brightness(1.05);
  transform: translateY(-1px);
}
.stButton > button:active {
  transform: translateY(0px);
}

/* Slider accent (best-effort; Streamlit internals vary by version) */
div[data-baseweb="slider"] * {
  accent-color: #0db36b !important;
}

/* File uploader */
div[data-testid="stFileUploader"] {
  border-radius: 16px;
}

/* Tables */
div[data-testid="stDataFrame"] {
  border-radius: 16px;
  overflow: hidden;
  border: 1px solid rgba(255,255,255,0.10);
}

/* Reduce top padding */
.block-container { padding-top: 1.4rem; padding-bottom: 2rem; }

/* Footer */
.footer {
  margin-top: 1.2rem;
  color: rgba(215, 222, 234, 0.7);
  font-size: 0.85rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------
# Files
# -----------------------------
MODEL_PATH = "BrainStroke_resnet50.keras"
METADATA_PATH = "BrainStroke_resnet50_metadata.json"

# Your fixed class order:
CLASS_NAMES = ["Bleeding", "Ischemia", "Normal"]


# -----------------------------
# Helpers
# -----------------------------
def safe_read_json(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def get_model_input_size(model: tf.keras.Model) -> Tuple[int, int]:
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    h = int(shape[1]) if len(shape) > 2 and shape[1] is not None else 224
    w = int(shape[2]) if len(shape) > 2 and shape[2] is not None else 224
    return (h, w)


def preprocess_image(img: Image.Image, target_hw: Tuple[int, int]) -> np.ndarray:
    img = img.convert("RGB")
    img = ImageOps.fit(img, target_hw, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    x = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(x, axis=0)


def softmax_if_needed(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec).reshape(-1)
    if vec.size == 0:
        return vec
    if (vec.min() < 0.0) or (vec.max() > 1.0) or (abs(vec.sum() - 1.0) > 1e-2):
        ex = np.exp(vec - np.max(vec))
        vec = ex / (ex.sum() + 1e-12)
    return vec


@st.cache_resource(show_spinner=True)
def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Place '{os.path.basename(model_path)}' in the same folder as app.py."
        )
    return tf.keras.models.load_model(model_path)


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Controls")

    conf_threshold = st.slider(
        "Decision threshold (for confidence label)",
        min_value=0.50,
        max_value=0.99,
        value=0.70,
        step=0.01,
    )
    show_probs = st.checkbox("Show probability breakdown", value=True)
    show_meta = st.checkbox("Show model & metadata", value=False)

    st.divider()
    st.caption("Tip: Use clear, well-centered scans for best results.")


# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
<div class="hero">
  <div class="hero-title">Brain Stroke Classification</div>
  <div class="hero-sub">
    Upload a brain scan image to classify into <b>Bleeding</b>, <b>Ischemia</b>, or <b>Normal</b>.
    This interface is intended for demonstration and academic/project use only.
  </div>
</div>
""",
    unsafe_allow_html=True,
)
st.write("")


# -----------------------------
# Load model + metadata
# -----------------------------
try:
    model = load_model(MODEL_PATH)
    meta = safe_read_json(METADATA_PATH)
    input_hw = get_model_input_size(model)
except Exception as e:
    st.error("Failed to load the model or read metadata.")
    st.exception(e)
    st.stop()


# -----------------------------
# Upload / Preview
# -----------------------------
col_left, col_right = st.columns([1.05, 1.0], gap="large")

with col_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("1) Upload Image")

    uploaded = st.file_uploader(
        "Supported formats: JPG, JPEG, PNG",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )

    st.write("")
    run_btn = st.button("Classify", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("2) Preview")

    if uploaded is None:
        st.info("Upload an image to preview it here.")
    else:
        try:
            pil_img = Image.open(uploaded)
            st.image(pil_img, caption=f"{uploaded.name}", use_container_width=True)
            st.caption(f"Model input size: {input_hw[0]}×{input_hw[1]}")
        except Exception as e:
            st.error("Unable to read the uploaded image.")
            st.exception(e)

    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# Inference
# -----------------------------
st.write("")

if run_btn:
    if uploaded is None:
        st.warning("Please upload an image first.")
    else:
        try:
            pil_img = Image.open(uploaded)
            x = preprocess_image(pil_img, input_hw)

            with st.spinner("Running inference..."):
                raw_pred = model.predict(x, verbose=0)

            pred_vec = np.asarray(raw_pred).squeeze()

            if pred_vec.ndim == 0:
                pred_vec = np.array([float(pred_vec)])

            # Expect multiclass (3). If not, handle generically.
            probs = softmax_if_needed(pred_vec)

            # Ensure label alignment
            if probs.size != len(CLASS_NAMES):
                class_names = [f"Class_{i}" for i in range(probs.size)]
            else:
                class_names = CLASS_NAMES

            top_idx = int(np.argmax(probs))
            top_label = class_names[top_idx]
            top_prob = float(probs[top_idx])

            # KPI row
            k1, k2, k3 = st.columns([1, 1, 1], gap="large")
            with k1:
                st.markdown(
                    f"""
<div class="kpi">
  <div class="label">Predicted class</div>
  <div class="value">{top_label}</div>
</div>
""",
                    unsafe_allow_html=True,
                )
            with k2:
                st.markdown(
                    f"""
<div class="kpi">
  <div class="label">Confidence</div>
  <div class="value">{top_prob*100:.2f}%</div>
</div>
""",
                    unsafe_allow_html=True,
                )
            with k3:
                conf_tag = "High confidence" if top_prob >= conf_threshold else "Low confidence"
                st.markdown(
                    f"""
<div class="kpi">
  <div class="label">Assessment</div>
  <div class="value">{conf_tag}</div>
</div>
""",
                    unsafe_allow_html=True,
                )

            st.write("")

            if show_probs:
                df = (
                    pd.DataFrame({"Class": class_names, "Probability": probs.astype(float)})
                    .sort_values("Probability", ascending=False)
                    .reset_index(drop=True)
                )

                cA, cB = st.columns([1, 1], gap="large")

                with cA:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.subheader("Probability breakdown")
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                with cB:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.subheader("Chart")

                    chart_df = df.set_index("Class")
                    st.bar_chart(chart_df, y="Probability") 
                    st.markdown("</div>", unsafe_allow_html=True)


            if show_meta:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Model & metadata")
                st.write("**Model file:**", MODEL_PATH)
                st.write("**Metadata file:**", METADATA_PATH)
                st.write("**Class order used:**", class_names)
                st.write("**Input size:**", f"{input_hw[0]}×{input_hw[1]}")
                with st.expander("Show metadata JSON"):
                    st.json(meta if meta else {"note": "Metadata file not found or empty."})
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(
                """
<div class="footer">
  Disclaimer: This tool is for educational/prototype use only and is not a medical device. Do not use it for clinical decision-making.
</div>
""",
                unsafe_allow_html=True,
            )

        except Exception as e:
            st.error("Classification failed.")
            st.exception(e)
