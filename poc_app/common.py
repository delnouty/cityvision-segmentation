"""
Shared helpers for the PoC app: paths, cached data, mask colouring and the
accessibility helpers every page uses.

Accessibility conventions (WCAG essentials)
-------------------------------------------
- 1.1.1  every image gets real alt text plus a visible caption (``figure``;
         ``st.image`` sets alt="0"), every chart a title, an ARIA description
         shown as a caption, and a data table (``accessible_chart``).
- 1.4.1  colour is never the only cue: masks come with a per-class table, chart
         series are named on an axis, in the legend and in the tooltip.
- 1.4.3 / 1.4.11  model series use a colour-blind-safe palette with >= 3:1
         contrast; the class legend uses readable text and outlined swatches.
- 2.1.1  only native Streamlit widgets (keyboard operable).
"""

import base64
import html
import io
import json
import os
import sys

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (ROOT, os.path.join(ROOT, "src"), os.path.join(ROOT, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from cityvision.constants import CLASS_NAMES, PALETTE  # noqa: E402

APP_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(APP_DIR, "data")
STATS_PATH = os.path.join(DATA_DIR, "eda_stats.json")

# One fixed colour per model, everywhere (categorical slots 1-3, validated
# colour-blind safe for three series; >= 3:1 on light and dark backgrounds).
MODEL_ORDER = ["ResNet50-UNet", "Mask2Former-SwinT", "EoMT-B"]
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
# Splits reuse the same three slots (never shown together with models).
SPLIT_ORDER = ["train", "val", "test"]

CLASS_LABELS = [n.replace("_", " ") for n in CLASS_NAMES]


@st.cache_data(show_spinner=False)
def load_stats():
    if not os.path.exists(STATS_PATH):
        return None
    with open(STATS_PATH, encoding="utf-8") as f:
        return json.load(f)


def require_stats():
    stats = load_stats()
    if stats is None:
        st.error(
            "Dataset statistics not found. Build them once from the local "
            "Cityscapes copy:\n\n`python scripts/build_eda_stats.py`"
        )
        st.stop()
    return stats


@st.cache_data(show_spinner=False)
def load_image(rel_path):
    return Image.open(os.path.join(DATA_DIR, rel_path)).convert("RGB")


@st.cache_data(show_spinner=False)
def load_mask(rel_path):
    """9-class label map (H, W) uint8, values 0-8."""
    return np.array(Image.open(os.path.join(DATA_DIR, rel_path)))


def colorize(labels):
    return Image.fromarray(PALETTE[labels])


def overlay(image, labels, alpha):
    img = np.asarray(image, dtype=np.float32)
    col = PALETTE[labels].astype(np.float32)
    return Image.fromarray((img * (1 - alpha) + col * alpha).astype(np.uint8))


def figure(image, alt, caption=None):
    """
    Image with real alt text and a visible caption (WCAG 1.1.1).
    ``st.image`` renders alt="0", which a screen reader would read out.
    """
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=88)
    src = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    cap = (
        f"<figcaption style='font-size:.875rem;opacity:.8;margin-top:.3rem'>"
        f"{html.escape(caption)}</figcaption>"
        if caption
        else ""
    )
    st.markdown(
        f"<figure style='margin:0 0 1rem 0'><img src='{src}' alt='{html.escape(alt, quote=True)}' "
        f"style='width:100%;height:auto;border-radius:.4rem;display:block'>{cap}</figure>",
        unsafe_allow_html=True,
    )


def sample_label(s):
    return f"{s['id']}  ({s['city']})"


def class_legend():
    """Readable legend: swatch with outline + class name + id (not colour alone)."""
    cells = "".join(
        f"<span style='display:inline-flex;align-items:center;gap:.45rem;margin:.2rem .9rem .2rem 0;"
        f"font-size:.95rem'><span aria-hidden='true' style='width:1rem;height:1rem;border-radius:3px;"
        f"background:rgb({r},{g},{b});outline:1px solid rgba(128,128,128,.9)'></span>"
        f"{i} · {name}</span>"
        for i, (name, (r, g, b)) in enumerate(zip(CLASS_LABELS, PALETTE.tolist()))
    )
    st.markdown(
        f"<div role='list' aria-label='Mask colour legend'>{cells}</div>",
        unsafe_allow_html=True,
    )


def class_share_table(labels_by_name):
    """Per-class % of pixels for one or more label maps -> DataFrame."""
    rows = []
    for c, name in enumerate(CLASS_LABELS):
        row = {"class": f"{c} · {name}"}
        for key, labels in labels_by_name.items():
            row[key] = round(100 * float((labels == c).mean()), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def accessible_chart(chart, title, description, table, height=None):
    """
    Altair chart + its text alternative: the description is set as the
    chart's ARIA description and shown as a caption; the data table sits in
    an expander right under the chart.
    """
    chart = chart.properties(title=title, description=description)
    if height:
        chart = chart.properties(height=height)
    st.altair_chart(chart, width="stretch")
    st.caption(description)
    with st.expander(f"Data table: {title}"):
        st.dataframe(table, hide_index=True, width="stretch")


def model_color_scale():
    return alt.Scale(domain=MODEL_ORDER, range=MODEL_COLORS)


def split_color_scale(splits):
    return alt.Scale(domain=splits, range=MODEL_COLORS[: len(splits)])
