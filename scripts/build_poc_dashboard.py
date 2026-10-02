"""
scripts/build_poc_dashboard.py
==============================

Refreshes the data of the proof-of-concept dashboard (``dashboard/poc.html``).

The page is a self-contained HTML file; its numbers live in one JSON block
between the markers ``/*POC_DATA_BEGIN*/`` and ``/*POC_DATA_END*/``. This
script rebuilds that block from:

1. ``results/poc_comparison.json`` — written by ``scripts/evaluate_poc.py``
   (mIoU, per-class IoU, latency, bootstrap CIs, checkpoint hashes);
2. ``mlflow.db`` — per-epoch training curves, training time and best epoch of
   the three runs.

Usage
-----
    python scripts/evaluate_poc.py          # once, or after retraining
    python scripts/build_poc_dashboard.py   # then open dashboard/poc.html
"""

import argparse
import json
import os
import re
import sqlite3
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from cityvision.constants import CLASS_NAMES  # noqa: E402

BEGIN, END = "/*POC_DATA_BEGIN*/", "/*POC_DATA_END*/"

# Dashboard model name -> how to find its MLflow run + static facts.
# Baseline run: the ResNet50-UNet run whose best checkpoint is served
# (the run name predates the PoC naming).
RUNS = {
    "ResNet50-UNet": {
        "run_name": "model_train_resnet50",
        "pretraining": "ImageNet-1k (encoder only)",
        "paper": "U-Net (2015) + ResNet-50 (2016)",
        "role": "Baseline",
    },
    "Mask2Former-SwinT": {
        "run_name": "Mask2Former-SwinT",
        "pretraining": "ADE20K semantic (whole model)",
        "paper": "Cheng et al., CVPR 2022 · arXiv 2112.01527",
        "role": "Solution 1",
    },
    "EoMT-B": {
        "run_name": "EoMT-B",
        "pretraining": "COCO panoptic, DINOv2 ViT-B (whole model)",
        "paper": "Kerssies et al., CVPR 2025 · arXiv 2503.19108",
        "role": "Solution 2",
    },
}


def mlflow_run(conn, run_name, best_miou):
    """
    Latest FINISHED run with that name; if several, the one whose
    best_val_mIoU matches the evaluated checkpoint.
    """
    rows = conn.execute(
        "select run_uuid, start_time, end_time from runs "
        "where name=? and status='FINISHED' order by start_time desc",
        (run_name,),
    ).fetchall()
    best = None
    for run_uuid, t0, t1 in rows:
        r = conn.execute(
            "select value from metrics where run_uuid=? and key='best_val_mIoU'",
            (run_uuid,),
        ).fetchone()
        if r is None:
            continue
        if best is None or abs(r[0] - best_miou) < abs(best[3] - best_miou):
            best = (run_uuid, t0, t1, r[0])
    return best


def curves(conn, run_uuid):
    keys = (
        "train_mIoU",
        "val_mIoU",
        "attn_mask_prob_0",
        "attn_mask_prob_1",
        "attn_mask_prob_2",
    )
    by_epoch = {}
    q = (
        "select key, value, step from metrics where run_uuid=? and key in "
        f"({','.join('?' * len(keys))}) order by step"
    )
    for key, value, step in conn.execute(q, (run_uuid, *keys)):
        by_epoch.setdefault(step, {})[key] = value
    epochs = sorted(e for e in by_epoch if "val_mIoU" in by_epoch[e])
    out = {
        "epoch": epochs,
        "val": [round(by_epoch[e]["val_mIoU"], 4) for e in epochs],
        "train": [
            round(by_epoch[e].get("train_mIoU", float("nan")), 4) for e in epochs
        ],
    }
    if any("attn_mask_prob_0" in by_epoch[e] for e in epochs):
        out["mask_probs"] = [
            [round(by_epoch[e].get(f"attn_mask_prob_{b}", 0.0), 3) for b in range(3)]
            for e in epochs
        ]
    return out


def build(comparison, conn):
    models = []
    for name, m in comparison["models"].items():
        meta = RUNS.get(name, {})
        run = mlflow_run(conn, meta.get("run_name", name), m["mIoU"]) if conn else None
        entry = {
            "name": name,
            "role": meta.get("role", ""),
            "paper": meta.get("paper", ""),
            "pretraining": meta.get("pretraining", ""),
            "checkpoint": m["checkpoint"],
            "sha256": m["sha256"][:12],
            "precision": m["precision"],
            "params_M": round(m["params_M"], 1),
            "mIoU": round(m["mIoU"], 4),
            "pixel_accuracy": round(m["pixel_accuracy"], 4),
            "iou": {c: round(m["iou"][c], 4) for c in CLASS_NAMES},
            "latency_ms": (
                {k: round(v, 1) for k, v in m["latency_ms"].items()}
                if m["latency_ms"]
                else None
            ),
            "train_hours": None,
            "curves": None,
        }
        if run:
            run_uuid, t0, t1, _ = run
            entry["train_hours"] = round((t1 - t0) / 3_600_000, 2)
            entry["curves"] = curves(conn, run_uuid)
        models.append(entry)

    comps = [
        {
            "A": c["A"],
            "B": c["B"],
            "delta": round(c["delta_mIoU"], 4),
            "ci": [round(x, 4) for x in c["ci_mIoU"]],
            "p_le_0": round(c["p_delta_le_0"], 3),
            "classes": {
                k: {
                    "delta": round(c["delta_iou"][k], 4),
                    "ci": [round(x, 4) for x in c["ci_iou"][k]],
                }
                for k in CLASS_NAMES
            },
        }
        for c in comparison["comparisons"]
    ]
    return {
        "generated": comparison["generated"],
        "device": comparison["device"],
        "n_images": comparison["n_images"],
        "versions": comparison["versions"],
        "bootstrap": comparison["bootstrap"],
        "classes": CLASS_NAMES,
        "models": models,
        "comparisons": comps,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--comparison", default="results/poc_comparison.json")
    p.add_argument("--mlflow-db", default="mlflow.db")
    p.add_argument("--page", default="dashboard/poc.html")
    args = p.parse_args()

    comp_path = os.path.join(ROOT, args.comparison)
    if not os.path.exists(comp_path):
        sys.exit(f"{args.comparison} not found — run scripts/evaluate_poc.py first.")
    with open(comp_path, encoding="utf-8") as f:
        comparison = json.load(f)

    db_path = os.path.join(ROOT, args.mlflow_db)
    conn = sqlite3.connect(db_path) if os.path.exists(db_path) else None
    if conn is None:
        print(f"[warn] {args.mlflow_db} not found — training curves left empty.")
    data = build(comparison, conn)

    page_path = os.path.join(ROOT, args.page)
    with open(page_path, encoding="utf-8") as f:
        page = f.read()
    pattern = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.S)
    if not pattern.search(page):
        sys.exit(f"Data markers not found in {args.page}.")
    block = BEGIN + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + END
    page = pattern.sub(lambda _: block, page, count=1)
    with open(page_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(page)

    print(
        f"Updated {args.page}: {len(data['models'])} models, {len(data['comparisons'])} comparisons"
    )
    for m in data["models"]:
        c = m["curves"]
        print(
            f"  {m['name']:18s} mIoU {m['mIoU']:.4f}  "
            f"curves: {len(c['epoch']) if c else 0} epochs  train {m['train_hours']} h"
        )


if __name__ == "__main__":
    main()
