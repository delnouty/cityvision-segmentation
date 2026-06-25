"""
frontend/streamlit_app.py
=========================

Streamlit UI for presenting CityVision segmentation results by consuming the
prediction API (backend/app.py).

Features
--------
- Lists the available image IDs for a chosen dataset split.
- Sends the selected image to the API to get the predicted mask.
- Shows, side by side: the real image, the real (ground-truth) mask, and the
  predicted mask.

Run
---
    # 1) start the API (separate terminal, from project root):
    uvicorn backend.app:app --port 8000
    # 2) start this app (from project root):
    streamlit run frontend/streamlit_app.py
"""

import os
import sys
import io

import numpy as np
import requests
import streamlit as st
from PIL import Image

# --- make project modules importable ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from dataloader import get_cityscapes_pairs          # same dataloader as training
from backend.inference import PALETTE, CLASS_NAMES    # same colours as the model

# Raw Cityscapes labelId -> 9-class training scheme (mirrors the training scripts).
TARGET_CLASSES = {7: 1, 11: 2, 21: 3, 23: 4, 24: 5, 26: 6, 20: 7, 33: 8}

IMG_ROOT = os.path.join(PROJECT_ROOT, "data/cityscapes/P8_Cityscapes_leftImg8bit_trainvaltest/leftImg8bit")
MASK_ROOT = os.path.join(PROJECT_ROOT, "data/cityscapes/P8_Cityscapes_gtFine_trainvaltest/gtFine")

DEFAULT_API = "http://localhost:8000"


# ============================================================
# Helpers
# ============================================================

def image_id(img_path: str) -> str:
    """e.g. .../berlin/berlin_000000_000019_leftImg8bit.png -> berlin_000000_000019"""
    return os.path.basename(img_path).replace("_leftImg8bit.png", "")


@st.cache_data(show_spinner=False)
def load_pairs(split: str):
    """{image_id: (img_path, mask_path)} for the split (same dataloader as training)."""
    pairs = get_cityscapes_pairs(IMG_ROOT, MASK_ROOT, split=split)
    return {image_id(ip): (ip, mp) for ip, mp in pairs}


def colorize_gt(mask_path: str) -> tuple[Image.Image, bool]:
    """Read a gtFine labelIds mask, remap to 9 classes, colourise.
    Returns (image, has_real_labels)."""
    raw = np.array(Image.open(mask_path))
    remapped = np.zeros_like(raw, dtype=np.uint8)
    for src, dst in TARGET_CLASSES.items():
        remapped[raw == src] = dst
    has_real = bool(remapped.any())  # test split remaps to all-background
    return Image.fromarray(PALETTE[remapped], mode="RGB"), has_real


@st.cache_data(show_spinner=False, ttl=30)
def api_health(api_url: str):
    r = requests.get(f"{api_url}/health", timeout=5)
    r.raise_for_status()
    return r.json()


def api_predict(api_url: str, img_path: str, fmt: str = "color", alpha: float = 0.5) -> bytes:
    with open(img_path, "rb") as f:
        files = {"file": (os.path.basename(img_path), f, "image/png")}
        r = requests.post(f"{api_url}/predict",
                          params={"format": fmt, "alpha": alpha},
                          files=files, timeout=120)
    r.raise_for_status()
    return r.content


def api_classes(api_url: str, img_path: str) -> dict:
    with open(img_path, "rb") as f:
        files = {"file": (os.path.basename(img_path), f, "image/png")}
        r = requests.post(f"{api_url}/predict/classes", files=files, timeout=120)
    r.raise_for_status()
    return r.json()


# ============================================================
# UI
# ============================================================

st.set_page_config(page_title="CityVision — Segmentation", layout="wide")
st.title("CityVision — Urban Scene Segmentation")
st.caption("Presentation UI that consumes the prediction API.")

with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API URL", value=DEFAULT_API).rstrip("/")

    # API status
    try:
        health = api_health(api_url)
        st.success(f"API online — {health['architecture']} on {health['device']}"
                   + (f" · val mIoU {health['val_mIoU']:.4f}" if health.get("val_mIoU") else ""))
    except Exception as e:
        st.error(f"API not reachable at {api_url}\n\n{e}")
        st.info("Start it with:\n\n`uvicorn backend.app:app --port 8000`")
        st.stop()

    split = st.selectbox("Dataset split", ["val", "test", "train"], index=0,
                         help="val/train have real ground-truth masks; "
                              "the test split does not (Cityscapes withholds it).")
    overlay = st.checkbox("Show prediction as overlay", value=False)
    alpha = st.slider("Overlay opacity", 0.0, 1.0, 0.5, 0.05, disabled=not overlay)

# --- image id selection ---
pairs = load_pairs(split)
if not pairs:
    st.warning(f"No images found for split '{split}'.")
    st.stop()

ids = sorted(pairs.keys())
col_sel, col_btn = st.columns([4, 1])
with col_sel:
    selected = st.selectbox(f"Image ID ({len(ids)} available in '{split}')", ids)
with col_btn:
    st.write("")
    st.write("")
    run = st.button("Predict mask", type="primary", use_container_width=True)

img_path, mask_path = pairs[selected]

# --- display: real image | real mask | predicted mask ---
c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("Real image")
    st.image(img_path, use_container_width=True)

with c2:
    st.subheader("Real mask")
    gt_img, has_real = colorize_gt(mask_path)
    st.image(gt_img, use_container_width=True)
    if not has_real:
        st.info("This split has no real ground truth — mask is all background.")

with c3:
    st.subheader("Predicted mask")
    if run:
        try:
            with st.spinner("Calling API…"):
                fmt = "overlay" if overlay else "color"
                png = api_predict(api_url, img_path, fmt=fmt, alpha=alpha)
            st.image(Image.open(io.BytesIO(png)), use_container_width=True)

            classes = api_classes(api_url, img_path)["detected_classes"]
            with st.expander("Detected classes", expanded=True):
                for c in classes:
                    r, g, b = c["color"]
                    st.markdown(
                        f"<span style='display:inline-block;width:12px;height:12px;"
                        f"background:rgb({r},{g},{b});border-radius:2px;margin-right:6px'></span>"
                        f"{c['class_name']} — {c['percentage']}%",
                        unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.info("Select an image ID and click **Predict mask**.")

# --- class legend ---
st.divider()
st.caption("Classes")
legend_cols = st.columns(len(CLASS_NAMES))
for col, name, color in zip(legend_cols, CLASS_NAMES, PALETTE):
    r, g, b = (int(color[0]), int(color[1]), int(color[2]))
    col.markdown(
        f"<div style='display:flex;align-items:center;gap:6px'>"
        f"<span style='width:14px;height:14px;background:rgb({r},{g},{b});"
        f"border-radius:3px;border:1px solid #0003'></span>"
        f"<span style='font-size:.8rem'>{name}</span></div>",
        unsafe_allow_html=True)
