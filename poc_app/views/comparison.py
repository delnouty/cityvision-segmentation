"""Model comparison page: same data as dashboard/poc.html (evaluate_poc.py + MLflow)."""

import json
import os
import sqlite3

import altair as alt
import common
import pandas as pd
import streamlit as st
from build_poc_dashboard import build

COMP_PATH = os.path.join(common.ROOT, "results", "poc_comparison.json")
DB_PATH = os.path.join(common.ROOT, "mlflow.db")


@st.cache_data(show_spinner=False)
def load(mtime):
    with open(COMP_PATH, encoding="utf-8") as f:
        comparison = json.load(f)
    conn = sqlite3.connect(DB_PATH) if os.path.exists(DB_PATH) else None
    try:
        return build(comparison, conn)
    finally:
        if conn:
            conn.close()


st.title("Model comparison")
if not os.path.exists(COMP_PATH):
    st.error(
        "No results yet. Run `python scripts/evaluate_poc.py` once (about 5 minutes on GPU)."
    )
    st.stop()
data = load(os.path.getmtime(COMP_PATH))
M = data["models"]
names = [m["name"] for m in M]
by = {m["name"]: m for m in M}
base = M[0]

st.markdown(
    f"All models re-evaluated by one script on the **{data['n_images']} Cityscapes validation images** "
    f"(9 classes, 512×1024), evaluated {data['generated'][:10]} on {data['device']}."
)

# ---------- KPIs ----------
best = max(M, key=lambda m: m["mIoU"])
k = st.columns(len(M))
for col, m in zip(k, M):
    delta = None if m is base else f"{m['mIoU'] - base['mIoU']:+.4f} vs baseline"
    col.metric(f"{m['name']} · val mIoU", f"{m['mIoU']:.4f}", delta)

comps = data["comparisons"]
lines = []
for c in comps:
    sig = c["ci"][0] > 0 or c["ci"][1] < 0
    verdict = "significant" if sig else "not significant (the interval contains 0)"
    lines.append(
        f"- **{c['B']} − {c['A']}**: {c['delta']:+.4f} mIoU, 95% CI [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}], {verdict}."
    )
st.markdown("\n".join(lines))

# ============================================================
# Accuracy vs speed
# ============================================================
st.header("Accuracy and speed")
prec = st.radio(
    "Precision for timing", ["bf16", "fp32"], horizontal=True, key="cmp_prec"
)
sc_df = pd.DataFrame(
    [
        {
            "model": m["name"],
            "val mIoU": m["mIoU"],
            "ms per image": m["latency_ms"][prec],
            "params (M)": m["params_M"],
        }
        for m in M
        if m["latency_ms"]
    ]
)
if not sc_df.empty:
    pts = (
        alt.Chart(sc_df)
        .mark_point(filled=True, size=220, opacity=1)
        .encode(
            x=alt.X(
                "ms per image:Q",
                title=f"inference time per image, ms ({prec})",
                # Headroom on the right so the last point's label is not clipped
                scale=alt.Scale(domain=[0, float(sc_df["ms per image"].max()) * 1.45]),
            ),
            y=alt.Y("val mIoU:Q", scale=alt.Scale(zero=False, padding=20)),
            color=alt.Color("model:N", scale=common.model_color_scale(), title="Model"),
            # Colour + shape share one legend: model identity is not colour alone
            shape=alt.Shape("model:N", sort=common.MODEL_ORDER, title="Model"),
            tooltip=[
                "model:N",
                alt.Tooltip("val mIoU:Q", format=".4f"),
                alt.Tooltip("ms per image:Q", format=".0f"),
                "params (M):Q",
            ],
        )
    )
    fastest = sc_df.loc[sc_df["ms per image"].idxmin()]
    common.accessible_chart(
        pts,
        "Validation mIoU against inference time",
        f"Up and to the left is better. Fastest: {fastest['model']} ({fastest['ms per image']:.0f} ms); "
        f"most accurate: {best['name']} ({best['mIoU']:.4f}). Timed on one 512×1024 image, batch 1, on the "
        "project's laptop GPU; only ratios between models carry over to other hardware.",
        sc_df.round(4),
        height=300,
    )

# ============================================================
# Per-class IoU
# ============================================================
st.header("Per-class IoU")
pc_df = pd.DataFrame(
    [
        {"model": m["name"], "class": common.CLASS_LABELS[i], "IoU": m["iou"][c]}
        for m in M
        for i, c in enumerate(data["classes"])
    ]
)
rng = (
    alt.Chart(pc_df)
    .mark_rule(strokeWidth=2, color="gray", opacity=0.5)
    .encode(
        y=alt.Y("class:N", sort=common.CLASS_LABELS, title=None),
        x="min(IoU):Q",
        x2="max(IoU):Q",
    )
)
dots = (
    alt.Chart(pc_df)
    .mark_point(filled=True, size=110, opacity=1)
    .encode(
        y=alt.Y("class:N", sort=common.CLASS_LABELS, title=None),
        x=alt.X("IoU:Q", scale=alt.Scale(zero=False), title="validation IoU"),
        color=alt.Color("model:N", scale=common.model_color_scale(), title="Model"),
        shape=alt.Shape("model:N", sort=common.MODEL_ORDER, title="Model"),
        tooltip=["class:N", "model:N", alt.Tooltip("IoU:Q", format=".4f")],
    )
)
wide = (
    pc_df.pivot(index="class", columns="model", values="IoU")
    .reindex(common.CLASS_LABELS)[names]
    .reset_index()
)
others = [n for n in names if n != base["name"]]
gain = wide.set_index("class")[others].sub(
    wide.set_index("class")[base["name"]], axis=0
)
top = gain.mean(axis=1).sort_values(ascending=False).index[:3].tolist()
behind = gain[(gain < 0).any(axis=1)].index.tolist()
common.accessible_chart(
    rng + dots,
    "IoU per class and model",
    "Each row is a class; the grey line spans the models' scores. Largest average gains over the baseline: "
    + ", ".join(top)
    + ". "
    + (
        f"Classes where a new model is below the baseline: {', '.join(behind)}."
        if behind
        else "No class below the baseline."
    ),
    wide.round(4),
    height=330,
)

# ============================================================
# Bootstrap
# ============================================================
st.header("Is the difference real?")
if comps:
    labels = [f"{c['B']} − {c['A']}" for c in comps]
    pick = st.selectbox("Comparison", labels, key="cmp_pair")
    c = comps[labels.index(pick)]
    rows = [
        {"metric": "mIoU", "Δ": c["delta"], "low": c["ci"][0], "high": c["ci"][1]}
    ] + [
        {
            "metric": common.CLASS_LABELS[i],
            "Δ": c["classes"][k]["delta"],
            "low": c["classes"][k]["ci"][0],
            "high": c["classes"][k]["ci"][1],
        }
        for i, k in enumerate(data["classes"])
    ]
    fo_df = pd.DataFrame(rows)
    fo_df["result"] = [
        "significant" if (r.low > 0 or r.high < 0) else "not significant"
        for r in fo_df.itertuples()
    ]
    order = ["mIoU"] + common.CLASS_LABELS
    color = (
        common.MODEL_COLORS[common.MODEL_ORDER.index(c["B"])]
        if c["B"] in common.MODEL_ORDER
        else "gray"
    )
    zero = (
        alt.Chart(pd.DataFrame({"x": [0]}))
        .mark_rule(color="gray", strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    bars = (
        alt.Chart(fo_df)
        .mark_rule(strokeWidth=3, color=color)
        .encode(
            y=alt.Y("metric:N", sort=order, title=None),
            x=alt.X("low:Q", title=f"difference {pick} (positive = {c['B']} better)"),
            x2="high:Q",
        )
    )
    pts = (
        alt.Chart(fo_df)
        .mark_point(size=90, strokeWidth=2, color=color, opacity=1)
        .encode(
            y=alt.Y("metric:N", sort=order),
            x="Δ:Q",
            fill=alt.condition(
                alt.datum.result == "significant", alt.value(color), alt.value("white")
            ),
            shape=alt.Shape(
                "result:N",
                scale=alt.Scale(
                    domain=["significant", "not significant"],
                    range=["circle", "square"],
                ),
                title="Result",
            ),
            tooltip=[
                "metric:N",
                alt.Tooltip("Δ:Q", format="+.4f"),
                alt.Tooltip("low:Q", format="+.4f"),
                alt.Tooltip("high:Q", format="+.4f"),
                "result:N",
            ],
        )
    )
    n_sig = int((fo_df["result"] == "significant").sum())
    common.accessible_chart(
        zero + bars + pts,
        "Paired bootstrap, 95% confidence intervals",
        f"{pick}: mIoU {c['delta']:+.4f}, interval [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]. {n_sig} of 10 rows are "
        f"significant (filled circles: the interval excludes 0; hollow squares: not significant). "
        f"{data['bootstrap']['resamples']} resamples of the validation images; training variance is not covered.",
        fo_df.round(4),
        height=330,
    )

# ============================================================
# Training curves
# ============================================================
st.header("Training")
mode = st.radio(
    "Curve",
    ["Validation mIoU", "Train − val gap (overfitting)"],
    horizontal=True,
    key="cmp_curve",
)
cv_rows = []
for m in M:
    cv = m["curves"]
    if not cv:
        continue
    for e, v, t in zip(cv["epoch"], cv["val"], cv["train"]):
        cv_rows.append(
            {
                "model": m["name"],
                "epoch": e,
                "val mIoU": v,
                "train mIoU": t,
                "gap": round(t - v, 4),
            }
        )
if cv_rows:
    cv_df = pd.DataFrame(cv_rows)
    ycol = "val mIoU" if mode.startswith("Validation") else "gap"
    plot_df = cv_df if ycol == "gap" else cv_df[cv_df["val mIoU"] >= 0.78]
    layers = []
    eomt = by.get("EoMT-B")
    if eomt and eomt["curves"] and eomt["curves"].get("mask_probs"):
        mp, ep = eomt["curves"]["mask_probs"], eomt["curves"]["epoch"]
        start = next((ep[i] for i, p in enumerate(mp) if any(v < 1 for v in p)), None)
        end = next((ep[i] for i, p in enumerate(mp) if all(v == 0 for v in p)), ep[-1])
        if start:
            band = pd.DataFrame(
                {"from": [start - 1], "to": [end], "label": ["EoMT mask annealing"]}
            )
            layers.append(
                alt.Chart(band)
                .mark_rect(opacity=0.12, color=common.MODEL_COLORS[2])
                .encode(x="from:Q", x2="to:Q")
            )
            layers.append(
                alt.Chart(band)
                .mark_text(baseline="top", dy=4, fontSize=11)
                .encode(x=alt.X("mid:Q"), y=alt.value(0), text="label:N")
                .transform_calculate(mid="(datum.from + datum.to) / 2")
            )
    lines_ch = (
        alt.Chart(plot_df)
        .mark_line(point=alt.OverlayMarkDef(filled=True, size=40), strokeWidth=2)
        .encode(
            x=alt.X(
                "epoch:Q",
                scale=alt.Scale(domain=[1, max(cv_df["epoch"])]),
                axis=alt.Axis(tickMinStep=1),
            ),
            y=alt.Y(
                f"{ycol}:Q",
                scale=alt.Scale(zero=False),
                title="validation mIoU" if ycol != "gap" else "train mIoU − val mIoU",
            ),
            color=alt.Color("model:N", scale=common.model_color_scale(), title="Model"),
            strokeDash=alt.StrokeDash("model:N", sort=common.MODEL_ORDER, legend=None),
            tooltip=[
                "model:N",
                "epoch:Q",
                alt.Tooltip("val mIoU:Q", format=".4f"),
                alt.Tooltip("train mIoU:Q", format=".4f"),
                alt.Tooltip("gap:Q", format="+.4f"),
            ],
        )
    )
    best_ep = ", ".join(
        f"{m['name']} epoch {m['curves']['epoch'][m['curves']['val'].index(max(m['curves']['val']))]}"
        for m in M
        if m["curves"]
    )
    desc = (
        f"Best epochs (the evaluated checkpoints): {best_ep}. "
        + (
            "Shown from 0.78 up; the baseline's first epochs are lower (see the table). "
            if ycol != "gap"
            else "A gap that grows while validation stalls is overfitting. Final gaps: "
            + ", ".join(
                f"{n} {g:+.3f}"
                for n, g in cv_df.sort_values("epoch")
                .groupby("model")["gap"]
                .last()
                .items()
            )
            + ". "
        )
        + "EoMT is validated without masked attention, as deployed. Line styles differ per model."
    )
    common.accessible_chart(
        alt.layer(*layers, lines_ch),
        "Validation mIoU per epoch" if ycol != "gap" else "Train − val gap per epoch",
        desc,
        cv_df,
        height=320,
    )

# ============================================================
# Model cards
# ============================================================
st.header("The three models")
cards = st.columns(len(M))
for col, m in zip(cards, M):
    with col:
        st.subheader(m["name"])
        st.caption(f"{m['role']} · {m['paper']}")
        st.markdown(
            f"- val mIoU **{m['mIoU']:.4f}**\n- {m['params_M']} M parameters\n- pretraining: {m['pretraining']}\n"
            f"- precision: {m['precision']}\n- training: {m['train_hours'] or '–'} h\n"
            f"- checkpoint `{m['checkpoint'].split('/')[-1]}` ({m['sha256']})"
        )
