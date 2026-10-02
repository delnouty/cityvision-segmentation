"""Prediction page: choose an image and a model, see the predicted mask and its scores."""

import altair as alt
import common
import numpy as np
import pandas as pd
import predictors
import streamlit as st

stats = common.require_stats()
samples = stats["samples"]

st.title("Prediction")
st.markdown(
    "Choose a validation image and a model. The model runs on this machine "
    f"(**{predictors.DEVICE.upper()}**) on the 512×1024 input used in training and evaluation."
)

models = predictors.available_models()
if not models:
    st.error(
        "No checkpoint found. Expected files: `backend/model/resnet50_best.pth`, "
        "`model_backups/mask2former_swint_best.pth`, `model_backups/eomt_base_best.pth`."
    )
    st.stop()

# ---------- Inputs ----------
c1, c2, c3 = st.columns([1, 2, 2])
with c1:
    city = st.selectbox(
        "City", ["All"] + sorted({s["city"] for s in samples}), key="pr_city"
    )
pool = [s for s in samples if city == "All" or s["city"] == city]
with c2:
    sample = st.selectbox("Image", pool, format_func=common.sample_label, key="pr_img")
with c3:
    model_name = st.selectbox("Model", models, index=len(models) - 1, key="pr_model")

o1, o2, o3 = st.columns([1, 1, 2])
with o1:
    compare_all = st.toggle("Compare all models", value=False, key="pr_all")
with o2:
    use_overlay = st.toggle("Overlay on image", value=False, key="pr_overlay")
with o3:
    alpha = st.slider(
        "Overlay opacity", 0.1, 0.9, 0.5, 0.1, disabled=not use_overlay, key="pr_alpha"
    )

# Secondary style: the default primary (white on red) is ~3:1, below WCAG 4.5:1
run = st.button("Predict", type="secondary")
key = (sample["id"], tuple(models) if compare_all else (model_name,))
if run:
    image = common.load_image(sample["image"])
    results = {}
    for name in key[1]:
        with st.spinner(
            f"Running {name}… (the first run of a model loads it, ~10-30 s)"
        ):
            results[name] = predictors.predict(name, image)
    st.session_state["pr_result"] = (key, results)

result = st.session_state.get("pr_result")
if not result or result[0] != key:
    st.info("Choose an image and a model, then press **Predict**.")
    st.stop()

# ---------- Results ----------
_, results = result
image = common.load_image(sample["image"])
gt = common.load_mask(sample["mask"])


def show_mask(labels):
    return (
        common.overlay(image, labels, alpha) if use_overlay else common.colorize(labels)
    )


def mask_alt(kind, labels):
    """Text alternative of a mask: its main classes and their share of pixels."""
    share = (
        np.bincount(labels.ravel(), minlength=len(common.CLASS_LABELS)) / labels.size
    )
    top = np.argsort(share)[::-1][:4]
    return f"{kind} segmentation mask. Main classes: " + ", ".join(
        f"{common.CLASS_LABELS[c]} {100 * share[c]:.0f}%" for c in top if share[c] > 0
    )


st.header("Result")
cols = st.columns(2 + len(results))
with cols[0]:
    common.figure(
        image,
        f"Street scene in {sample['city']}",
        f"Input image: {sample['id']} ({sample['city']})",
    )
with cols[1]:
    common.figure(
        show_mask(gt), mask_alt("Ground-truth", gt), "Ground truth (real mask)"
    )
scores = {}
for col, (name, (pred, ms)) in zip(cols[2:], results.items()):
    pa, iou, miou = predictors.per_image_scores(pred, gt)
    scores[name] = (pa, iou, miou, ms)
    with col:
        common.figure(
            show_mask(pred),
            mask_alt(f"{name} predicted", pred),
            f"Prediction: {name}, mIoU {miou:.3f} on this image",
        )
common.class_legend()

st.subheader("Scores on this image")
mcols = st.columns(len(results))
for col, (name, (pa, iou, miou, ms)) in zip(mcols, scores.items()):
    with col:
        st.markdown(f"**{name}**")
        st.metric("mIoU (classes present)", f"{miou:.3f}")
        st.metric("Pixel accuracy", f"{100 * pa:.1f} %")
        st.metric("Inference time", f"{ms:.0f} ms")

# ---------- Per-class table (text equivalent of the masks) ----------
table = common.class_share_table(
    {"real %": gt, **{f"{n} %": p for n, (p, _) in results.items()}}
)
for name, (_, iou, _, _) in scores.items():
    table[f"{name} IoU"] = [None if np.isnan(v) else round(float(v), 3) for v in iou]
st.subheader("Per class")
st.caption(
    "Share of the image's pixels given to each class, real and predicted, and the IoU of the "
    "prediction (empty when the class is in neither mask). This table is the text equivalent of the masks."
)
st.dataframe(table, hide_index=True, width="stretch")

# ---------- IoU per class chart ----------
iou_rows = [
    {"model": n, "class": common.CLASS_LABELS[c], "IoU": float(v)}
    for n, (_, iou, _, _) in scores.items()
    for c, v in enumerate(iou)
    if not np.isnan(v)
]
if iou_rows:
    iou_df = pd.DataFrame(iou_rows)
    ch = (
        alt.Chart(iou_df)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            y=alt.Y("class:N", sort=common.CLASS_LABELS, title=None),
            yOffset=alt.YOffset("model:N", sort=common.MODEL_ORDER),
            x=alt.X(
                "IoU:Q",
                scale=alt.Scale(domain=[0, 1]),
                title="IoU on this image (1 = perfect)",
            ),
            color=alt.Color("model:N", scale=common.model_color_scale(), title="Model"),
            tooltip=["model:N", "class:N", alt.Tooltip("IoU:Q", format=".3f")],
        )
    )
    worst = min(iou_rows, key=lambda r: r["IoU"])
    common.accessible_chart(
        ch,
        "IoU per class on this image",
        f"Lowest IoU: {worst['class']} with {worst['model']} ({worst['IoU']:.2f}). "
        + "; ".join(f"{n}: mIoU {s[2]:.3f}" for n, s in scores.items())
        + ". Single-image scores vary a lot; the model comparison page gives the scores over 500 images.",
        iou_df.round(3),
        height=max(220, 24 * len(iou_df)),
    )
