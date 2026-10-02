"""Overview page: what the PoC is and where to go."""

import common
import streamlit as st

stats = common.load_stats()

st.title("CityVision proof of concept")
st.markdown(
    "Semantic segmentation of street scenes into **9 classes**. This app compares the "
    "project's **ResNet50-UNet** baseline with two recent transformers, **Mask2Former** "
    "(CVPR 2022) and **EoMT** (CVPR 2025), trained on the same data with the same budget."
)

if stats:
    s = stats["splits"]
    cols = st.columns(4)
    cols[0].metric("Training images", f"{s['train']['n_images']:,}")
    cols[1].metric("Validation images", f"{s['val']['n_images']:,}")
    cols[2].metric("Test images (no public labels)", f"{s['test']['n_images']:,}")
    cols[3].metric("Cities", len({c for sp in s.values() for c in sp["cities"]}))

st.subheader("Pages")
st.page_link(
    "views/dataset.py",
    label="Dataset: counts, example images, transformations",
    icon="🖼️",
)
st.page_link(
    "views/prediction.py", label="Prediction: pick an image and a model", icon="🎯"
)
st.page_link(
    "views/comparison.py",
    label="Model comparison: accuracy, speed, statistics",
    icon="📊",
)

st.subheader("Classes")
st.caption(
    "Colours used in every mask. Each class is also named in the tables next to the masks."
)
common.class_legend()

st.subheader("Accessibility")
st.markdown(
    "- Every chart has a title, a text summary and a data table.\n"
    "- Masks always come with a table of class names and percentages, so colour is never the only cue.\n"
    "- Model colours are chosen to stay distinguishable with colour-vision deficiencies.\n"
    "- All controls are standard widgets usable with the keyboard (Tab, arrows, Enter)."
)
