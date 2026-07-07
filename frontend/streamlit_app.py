"""
frontend/streamlit_app.py
=========================

Streamlit UI for presenting CityVision segmentation results by consuming the
prediction API (backend/app.py).

Image source
------------
Uses a small curated sample set committed to the repo at `data/samples/`
(two image+mask pairs per city, built by `scripts/make_samples.py`), so it runs
without the full Cityscapes dataset and shows the real image, the real mask, and
the predicted mask.

Run
---
    # 1) start the API (separate terminal, from project root):
    uvicorn backend.app:app --port 8000
    # 2) start this app (from project root):
    streamlit run frontend/streamlit_app.py

Pure helpers live in frontend/utils.py (unit-tested); this file is the UI layer.
"""

import os
import sys
import io

import streamlit as st
from PIL import Image

# Allow `import utils` whether launched via `streamlit run frontend/streamlit_app.py`
# (script dir on path) or as part of the `frontend` package.
sys.path.insert(0, os.path.dirname(__file__))
import utils

# Streamlit-cached wrappers around the pure helpers.
load_samples = st.cache_data(show_spinner=False)(utils.list_sample_images)
cached_health = st.cache_data(show_spinner=False, ttl=30)(utils.api_health)


st.set_page_config(page_title="CityVision — Segmentation", layout="wide")
st.title("CityVision — Urban Scene Segmentation")
st.caption("Presentation UI that consumes the prediction API.")

with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API URL", value=utils.DEFAULT_API).rstrip("/")

    try:
        health = cached_health(api_url)
        st.success(
            f"API online — {health['architecture']} on {health['device']}"
            + (
                f" · val mIoU {health['val_mIoU']:.4f}"
                if health.get("val_mIoU")
                else ""
            )
        )
    except Exception as e:
        st.error(f"API not reachable at {api_url}\n\n{e}")
        st.info("Start it with:\n\n`uvicorn backend.app:app --port 8000`")
        st.stop()

    overlay = st.checkbox("Show prediction as overlay", value=False)
    alpha = st.slider("Overlay opacity", 0.0, 1.0, 0.5, 0.05, disabled=not overlay)

# --- available sample images (bundled in data/samples/) ---
pairs = load_samples()
if not pairs:
    st.warning(
        "No sample images found in `data/samples/`.\n\n"
        "Build them with: `python scripts/make_samples.py`"
    )
    st.stop()
label = f"Image ({len(pairs)} sample images, 2 per city)"
ids = sorted(pairs.keys())

col_sel, col_btn = st.columns([4, 1])
with col_sel:
    selected = st.selectbox(label, ids)
with col_btn:
    st.write("")
    st.write("")
    run = st.button("Predict mask", type="primary", use_container_width=True)

img_path, mask_path = pairs[selected]

# --- panels: real image | real mask | predicted mask ---
has_gt = mask_path is not None
columns = st.columns(3 if has_gt else 2)

with columns[0]:
    st.subheader("Real image")
    st.image(img_path, use_container_width=True)

pred_col = columns[1]
if has_gt:
    with columns[1]:
        st.subheader("Real mask")
        gt_img, has_real = utils.colorize_gt(mask_path)
        st.image(gt_img, use_container_width=True)
        if not has_real:
            st.info("This split has no real ground truth — mask is all background.")
    pred_col = columns[2]

with pred_col:
    st.subheader("Predicted mask")
    if run:
        try:
            with st.spinner("Calling API…"):
                fmt = "overlay" if overlay else "color"
                png = utils.api_predict(api_url, img_path, fmt=fmt, alpha=alpha)
            st.image(Image.open(io.BytesIO(png)), use_container_width=True)

            classes = utils.api_classes(api_url, img_path)["detected_classes"]
            with st.expander("Detected classes", expanded=True):
                for c in classes:
                    r, g, b = c["color"]
                    st.markdown(
                        f"<span style='display:inline-block;width:12px;height:12px;"
                        f"background:rgb({r},{g},{b});border-radius:2px;margin-right:6px'></span>"
                        f"{c['class_name']} — {c['percentage']}%",
                        unsafe_allow_html=True,
                    )
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.info("Select an image and click **Predict mask**.")

st.divider()
st.caption("Classes")
legend_cols = st.columns(len(utils.CLASS_NAMES))
for col, name, color in zip(legend_cols, utils.CLASS_NAMES, utils.PALETTE):
    r, g, b = (int(color[0]), int(color[1]), int(color[2]))
    col.markdown(
        f"<div style='display:flex;align-items:center;gap:6px'>"
        f"<span style='width:14px;height:14px;background:rgb({r},{g},{b});"
        f"border-radius:3px;border:1px solid #0003'></span>"
        f"<span style='font-size:.8rem'>{name}</span></div>",
        unsafe_allow_html=True,
    )
