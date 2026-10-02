"""
poc_app/app.py
==============

CityVision proof-of-concept dashboard (local): dataset exploration,
prediction with the three models, and the model comparison.

Run from the project root:
    python scripts/build_eda_stats.py      # once: dataset statistics + samples
    python scripts/evaluate_poc.py         # once: comparison results (page 3)
    streamlit run poc_app/app.py

Needs the PoC dependencies (requirements-poc.txt) and the checkpoints in
backend/model/ and model_backups/.
"""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

st.set_page_config(
    page_title="CityVision PoC",
    page_icon="🛣️",
    layout="wide",
    menu_items={
        "About": "CityVision proof of concept: ResNet50-UNet vs Mask2Former vs EoMT."
    },
)

# Page paths are relative to this file (Streamlit's convention)
pages = [
    st.Page("views/home.py", title="Overview", icon="🏠", default=True),
    st.Page("views/dataset.py", title="Dataset", icon="🖼️"),
    st.Page("views/prediction.py", title="Prediction", icon="🎯"),
    st.Page("views/comparison.py", title="Model comparison", icon="📊"),
]
st.navigation(pages).run()
