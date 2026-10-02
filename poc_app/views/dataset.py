"""Dataset page: counts, example images by category, image transformations."""

import altair as alt
import common
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import ImageOps

stats = common.require_stats()
splits = stats["splits"]
samples = stats["samples"]
W, H = stats["original_size"]

st.title("Dataset")
st.markdown(
    f"**Cityscapes**: street scenes from German cities, {W}×{H} pixels, with pixel-level "
    "labels. The project keeps **9 classes**: every Cityscapes label not listed below is "
    "merged into *background*."
)

# ============================================================
# 1. Counts
# ============================================================
st.header("1. Counts")
cols = st.columns(len(splits))
for col, (name, s) in zip(cols, splits.items()):
    col.metric(
        f"{name.capitalize()} images · {len(s['cities'])} cities", f"{s['n_images']:,}"
    )

# --- images per city ---
city_df = pd.DataFrame(
    [
        {"split": sp, "city": c, "images": n}
        for sp, s in splits.items()
        for c, n in s["cities"].items()
    ]
)
labelled = [sp for sp in common.SPLIT_ORDER if sp in splits]
# Explicit order (by split, then by size): a layered chart ignores per-layer sorts
city_df["split_rank"] = city_df["split"].map(
    {s: i for i, s in enumerate(common.SPLIT_ORDER)}
)
city_order = city_df.sort_values(["split_rank", "images"], ascending=[True, False])[
    "city"
].tolist()
city_df = city_df.drop(columns="split_rank")
bars = (
    alt.Chart(city_df)
    .mark_bar(cornerRadiusEnd=3, height={"band": 0.75})
    .encode(
        y=alt.Y(
            "city:N",
            sort=city_order,
            title=None,
            axis=alt.Axis(labelLimit=200, labelOverlap=False),
        ),
        x=alt.X("images:Q", title="number of images"),
        color=alt.Color(
            "split:N", scale=common.split_color_scale(labelled), title="Split"
        ),
        tooltip=[
            alt.Tooltip("city:N"),
            alt.Tooltip("split:N"),
            alt.Tooltip("images:Q"),
        ],
    )
)
common.accessible_chart(
    bars,
    "Images per city",
    f"Each city belongs to one split only. Training: {splits['train']['n_images']:,} images over "
    f"{len(splits['train']['cities'])} cities; validation: {splits['val']['n_images']} images from "
    f"{', '.join(splits['val']['cities'])}; test: {splits['test']['n_images']:,} images (labels not public).",
    city_df.sort_values(["split", "images"], ascending=[True, False]),
    height=24 * len(city_df),
)

# --- class distribution ---
st.subheader("Class distribution (train and val, 9 classes)")
cls_rows = []
for sp in ("train", "val"):
    s = splits[sp]
    for name, label in zip(stats["classes"], common.CLASS_LABELS):
        cls_rows.append(
            {
                "split": sp,
                "class": label,
                "% of pixels": round(100 * s["pixel_share"][name], 2),
                "% of images containing it": round(
                    100 * s["images_with"][name] / s["n_images"], 1
                ),
                "images containing it": s["images_with"][name],
            }
        )
cls_df = pd.DataFrame(cls_rows)
class_order = common.CLASS_LABELS
c1, c2 = st.columns(2)
with c1:
    ch = (
        alt.Chart(cls_df)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X(
                "class:N",
                sort=class_order,
                title=None,
                axis=alt.Axis(labelAngle=-40, labelOverlap=False),
            ),
            xOffset=alt.XOffset("split:N", sort=["train", "val"]),
            y=alt.Y("% of pixels:Q", title="% of all labelled pixels"),
            color=alt.Color(
                "split:N",
                scale=common.split_color_scale(["train", "val"]),
                title="Split",
            ),
            tooltip=["class:N", "split:N", alt.Tooltip("% of pixels:Q", format=".2f")],
        )
    )
    tr = splits["train"]["pixel_share"]
    common.accessible_chart(
        ch,
        "Share of pixels per class",
        f"Strong imbalance: road ({100 * tr['road']:.1f}%), background ({100 * tr['background']:.1f}%) "
        f"and building ({100 * tr['building']:.1f}%) dominate the training pixels, while person "
        f"({100 * tr['person']:.2f}%), traffic sign ({100 * tr['traffic_sign']:.2f}%) and bicycle "
        f"({100 * tr['bicycle']:.2f}%) are rare. Train and val have similar shares.",
        cls_df[["split", "class", "% of pixels"]],
    )
with c2:
    ch2 = (
        alt.Chart(cls_df)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X(
                "class:N",
                sort=class_order,
                title=None,
                axis=alt.Axis(labelAngle=-40, labelOverlap=False),
            ),
            xOffset=alt.XOffset("split:N", sort=["train", "val"]),
            y=alt.Y(
                "% of images containing it:Q",
                title="% of images containing the class",
                scale=alt.Scale(domain=[0, 100]),
            ),
            color=alt.Color(
                "split:N",
                scale=common.split_color_scale(["train", "val"]),
                title="Split",
            ),
            tooltip=[
                "class:N",
                "split:N",
                "images containing it:Q",
                alt.Tooltip("% of images containing it:Q", format=".1f"),
            ],
        )
    )
    iw, n = splits["train"]["images_with"], splits["train"]["n_images"]
    common.accessible_chart(
        ch2,
        "Images containing each class",
        f"Rare in pixels does not mean rare in images: traffic signs appear in "
        f"{100 * iw['traffic_sign'] / n:.0f}% of training images and bicycles in {100 * iw['bicycle'] / n:.0f}%, "
        "but they are small objects. This is why the baseline loss up-weights them.",
        cls_df[["split", "class", "images containing it", "% of images containing it"]],
    )

# ============================================================
# 2. Example images by category
# ============================================================
st.header("2. Example images by category")
st.caption(
    f"{len(samples)} validation images bundled with the app (10 per city), stored at the models' "
    "input size, 1024×512."
)
g1, g2, g3 = st.columns([1, 1, 1])
with g1:
    browse = st.radio("Browse by", ["City", "Class"], horizontal=True, key="ds_browse")
with g2:
    if browse == "City":
        cities = sorted({s["city"] for s in samples})
        pick = st.selectbox("City", cities, key="ds_city")
        shown = [s for s in samples if s["city"] == pick]
    else:
        cls_label = st.selectbox("Class", common.CLASS_LABELS[1:], key="ds_class")
        cls_name = stats["classes"][common.CLASS_LABELS.index(cls_label)]
        shown = sorted(
            (s for s in samples if s["class_share"][cls_name] > 0),
            key=lambda s: -s["class_share"][cls_name],
        )[:6]
with g3:
    show_mask = st.toggle(
        "Overlay the ground-truth mask", value=False, key="ds_overlay"
    )

if browse == "Class":
    st.markdown(
        f"The {len(shown)} bundled images where **{cls_label}** covers the largest share of pixels."
    )
grid = st.columns(3)
for k, s in enumerate(shown):
    img = common.load_image(s["image"])
    if show_mask:
        img = common.overlay(img, common.load_mask(s["mask"]), 0.5)
    top = sorted(s["class_share"].items(), key=lambda kv: -kv[1])[:3]
    caption = f"{s['city']}, {s['id']}. Main classes: " + ", ".join(
        f"{n.replace('_', ' ')} {100 * v:.0f}%" for n, v in top
    )
    if browse == "Class":
        caption += f". {cls_label}: {100 * s['class_share'][cls_name]:.1f}% of pixels"
    alt_text = f"Street scene in {s['city']}" + (
        " with its ground-truth mask overlaid" if show_mask else ""
    )
    with grid[k % 3]:
        common.figure(img, alt_text, caption)
if show_mask:
    common.class_legend()

# ============================================================
# 3. Transformations
# ============================================================
st.header("3. Image transformations")
st.markdown(
    "Classic pre-processing operations, and the normalisation actually applied before the models. "
    "The histogram compares the brightness distribution before and after."
)
t1, t2 = st.columns([1, 1])
with t1:
    tsample = st.selectbox(
        "Image", samples, format_func=common.sample_label, key="tf_img"
    )
with t2:
    method = st.selectbox(
        "Transformation",
        [
            "Histogram equalisation",
            "CLAHE (adaptive equalisation)",
            "Gaussian blur",
            "Auto-contrast",
            "Model input normalisation",
        ],
        key="tf_method",
    )

rgb = np.asarray(common.load_image(tsample["image"]))
params = st.columns(3)
if method == "Histogram equalisation":
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    ycc[..., 0] = cv2.equalizeHist(ycc[..., 0])
    out = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)
    explain = "Spreads the brightness channel (Y of YCrCb) over the full 0-255 range; colours are kept."
elif method.startswith("CLAHE"):
    clip = params[0].slider("Clip limit", 1.0, 8.0, 2.0, 0.5, key="tf_clip")
    tiles = params[1].slider("Tile grid (n × n)", 2, 16, 8, 1, key="tf_tiles")
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(
        lab[..., 0]
    )
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    explain = (
        f"Equalises the lightness channel (L of LAB) tile by tile ({tiles}×{tiles} tiles), with the "
        f"contrast gain capped at {clip}: shadows get detail without over-brightening the sky."
    )
elif method == "Gaussian blur":
    sigma = params[0].slider("Sigma (pixels)", 0.5, 10.0, 3.0, 0.5, key="tf_sigma")
    out = cv2.GaussianBlur(rgb, (0, 0), sigma)
    explain = f"Low-pass filter with a Gaussian kernel of standard deviation {sigma} pixels: removes noise and fine detail."
elif method == "Auto-contrast":
    cutoff = params[0].slider("Ignored extreme pixels (%)", 0, 10, 1, 1, key="tf_cut")
    out = np.asarray(
        ImageOps.autocontrast(common.load_image(tsample["image"]), cutoff=cutoff)
    )
    explain = f"Stretches each channel linearly so the darkest and brightest {cutoff}% map to 0 and 255."
else:
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    norm = (rgb / 255.0 - mean) / std
    lo, hi = norm.min(axis=(0, 1)), norm.max(axis=(0, 1))
    out = ((norm - lo) / (hi - lo) * 255).astype(np.uint8)
    explain = (
        "What the three models receive: each channel minus the ImageNet mean, divided by the ImageNet "
        "standard deviation (shown rescaled to 0-255 so it can be displayed)."
    )
    st.dataframe(
        pd.DataFrame(
            {
                "channel": ["R", "G", "B"],
                "mean before": (rgb / 255.0).mean(axis=(0, 1)).round(3),
                "std before": (rgb / 255.0).std(axis=(0, 1)).round(3),
                "mean after": norm.mean(axis=(0, 1)).round(3),
                "std after": norm.std(axis=(0, 1)).round(3),
            }
        ),
        hide_index=True,
    )

st.caption(explain)
i1, i2 = st.columns(2)
with i1:
    common.figure(
        rgb,
        f"Original street scene in {tsample['city']}",
        f"Original: {tsample['id']} ({tsample['city']})",
    )
with i2:
    common.figure(
        out,
        f"The same scene after {method.lower()}",
        f"After: {method.lower()}",
    )


def luminance(a):
    return (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).ravel()


bins = np.linspace(0, 256, 33)
hist_rows = []
for version, arr in (("original", rgb), ("transformed", out)):
    counts, _ = np.histogram(luminance(arr.astype(np.float32)), bins=bins)
    for k, c in enumerate(counts):
        hist_rows.append(
            {
                "version": version,
                "brightness from": int(bins[k]),
                "brightness to": int(bins[k + 1]) - 1,
                "% of pixels": round(100 * c / counts.sum(), 2),
            }
        )
hist_df = pd.DataFrame(hist_rows)
hist = (
    alt.Chart(hist_df)
    .mark_line(
        point=alt.OverlayMarkDef(size=30), strokeWidth=2, interpolate="step-after"
    )
    .encode(
        x=alt.X(
            "brightness from:Q",
            title="brightness (0 = black, 255 = white)",
            scale=alt.Scale(domain=[0, 256]),
        ),
        y=alt.Y("% of pixels:Q"),
        color=alt.Color(
            "version:N",
            scale=alt.Scale(
                domain=["original", "transformed"], range=common.MODEL_COLORS[:2]
            ),
            title=None,
        ),
        strokeDash=alt.StrokeDash(
            "version:N",
            scale=alt.Scale(domain=["original", "transformed"], range=[[1, 0], [5, 3]]),
            legend=None,
        ),
        tooltip=[
            "version:N",
            "brightness from:Q",
            "brightness to:Q",
            alt.Tooltip("% of pixels:Q", format=".2f"),
        ],
    )
)
y_o, y_t = luminance(rgb.astype(np.float32)), luminance(out.astype(np.float32))
common.accessible_chart(
    hist,
    "Brightness histogram, before and after",
    f"Mean brightness {y_o.mean():.0f} → {y_t.mean():.0f}; spread (standard deviation) "
    f"{y_o.std():.0f} → {y_t.std():.0f}. Solid line: original; dashed line: transformed.",
    hist_df,
    height=260,
)
