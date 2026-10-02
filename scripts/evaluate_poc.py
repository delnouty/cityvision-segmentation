"""
scripts/evaluate_poc.py
=======================

Final comparison table of the proof of concept: ResNet50-UNet (baseline) vs
Mask2Former Swin-T vs EoMT-B, all re-evaluated by one script on the same data.

What it does
------------
1. Loads each checkpoint and rebuilds its architecture exactly as trained
   (``cityvision.models``, ``src/training_mask2former.py``,
   ``src/training_eomt.py``).
2. Predicts the 500 Cityscapes val images (512x1024, 9 classes) with the same
   dataloader and remap as training, in each model's training precision
   (baseline fp32, transformers bf16; EoMT mask-free, as deployed), and keeps
   one confusion matrix per image.
3. Reports mIoU, pixel accuracy and per-class IoU (same definition as
   ``SegmentationMetrics``), parameters, and inference latency (1 image,
   fp32 and bf16, GPU only).
4. Paired bootstrap over images for every pair of models: 95% CI of the mIoU
   (and per-class IoU) difference.

Outputs (``results/`` is gitignored)
------------------------------------
- ``results/poc_comparison.md``          tables, also printed to the console
- ``results/poc_comparison.json``        every number + checkpoint hashes,
                                         library versions, GPU
- ``results/poc_confusions.npz``         per-image confusion matrices, so the
                                         bootstrap can be redone without a GPU

Usage
-----
    python scripts/evaluate_poc.py
    python scripts/evaluate_poc.py --limit 20 --bootstrap 200 --no-latency   # quick check
    python scripts/evaluate_poc.py --eomt model_backups/other_run.pth
"""

import argparse
import datetime
import hashlib
import json
import os
import sys
import time
from itertools import combinations

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

import training_eomt  # noqa: E402
import training_mask2former  # noqa: E402
from dataloader import CityscapesDataset, get_cityscapes_pairs  # noqa: E402

from cityvision.constants import CLASS_NAMES, NUM_CLASSES  # noqa: E402
from cityvision.models import ResNet50UNet, remap_mask  # noqa: E402

IMG_ROOT = os.path.join(
    ROOT, "data/cityscapes/P8_Cityscapes_leftImg8bit_trainvaltest/leftImg8bit"
)
MASK_ROOT = os.path.join(
    ROOT, "data/cityscapes/P8_Cityscapes_gtFine_trainvaltest/gtFine"
)
IMG_SIZE = (512, 1024)


# ============================================================
# 1. MODELS
# ============================================================


def _load_resnet50(path):
    model = ResNet50UNet(num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model


def _load_mask2former(path):
    model = training_mask2former.build_model()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model


def _load_eomt(path):
    model = training_eomt.build_model()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.attn_mask_probs.zero_()  # mask-free inference, as deployed
    return model


def _predict_resnet50(model, imgs):
    return model(imgs).argmax(dim=1)


def _predict_mask_classification(model, imgs):
    outputs = model(pixel_values=imgs)
    return training_mask2former.semantic_logits(outputs, imgs.shape[-2:]).argmax(dim=1)


# name -> (CLI option, default checkpoint, loader, predict, bf16 for accuracy)
# Precision for accuracy = the one each model was trained and validated in.
MODELS = {
    "ResNet50-UNet": (
        "baseline",
        "backend/model/resnet50_best.pth",
        _load_resnet50,
        _predict_resnet50,
        False,
    ),
    "Mask2Former-SwinT": (
        "mask2former",
        "model_backups/mask2former_swint_best.pth",
        _load_mask2former,
        _predict_mask_classification,
        True,
    ),
    "EoMT-B": (
        "eomt",
        "model_backups/eomt_base_best.pth",
        _load_eomt,
        _predict_mask_classification,
        True,
    ),
}


# ============================================================
# 2. METRICS
# ============================================================


def confusion(preds, targets):
    """(K, K) confusion matrix, rows = ground truth, cols = prediction."""
    idx = targets.flatten() * NUM_CLASSES + preds.flatten()
    return (
        torch.bincount(idx, minlength=NUM_CLASSES**2)
        .reshape(NUM_CLASSES, NUM_CLASSES)
        .cpu()
        .numpy()
    )


def scores(conf):
    """mIoU, per-class IoU and pixel accuracy of a summed confusion matrix."""
    tp = np.diag(conf).astype(np.float64)
    denom = conf.sum(axis=0) + conf.sum(axis=1) - tp
    iou = np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)
    return float(np.nanmean(iou)), iou, float(tp.sum() / conf.sum())


def paired_bootstrap(conf_a, conf_b, n_resamples, seed):
    """
    Resample images with replacement (same indices for both models) and
    recompute mIoU / per-class IoU differences B - A.
    conf_a, conf_b: (n_images, K, K) per-image confusion matrices.
    """
    rng = np.random.default_rng(seed)
    n = len(conf_a)
    d_miou = np.empty(n_resamples)
    d_iou = np.empty((n_resamples, NUM_CLASSES))
    for r in range(n_resamples):
        idx = rng.integers(0, n, n)
        ma, ia, _ = scores(conf_a[idx].sum(axis=0))
        mb, ib, _ = scores(conf_b[idx].sum(axis=0))
        d_miou[r], d_iou[r] = mb - ma, ib - ia
    return d_miou, d_iou


def _ci(samples):
    return [
        float(np.nanpercentile(samples, 2.5)),
        float(np.nanpercentile(samples, 97.5)),
    ]


# ============================================================
# 3. EVALUATION
# ============================================================


def _autocast(device, enabled):
    return torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=enabled and device == "cuda"
    )


def evaluate(model, predict, bf16, loader, device):
    confs = []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), remap_mask(masks).to(device)
            with _autocast(device, bf16):
                preds = predict(model, imgs)
            for p, m in zip(preds, masks):
                confs.append(confusion(p, m))
    return np.stack(confs)


def latency_ms(model, predict, bf16, device, runs, warmup=5):
    """Mean time of one 512x1024 image, prediction included (GPU only)."""
    x = torch.randn(1, 3, *IMG_SIZE, device=device)
    with torch.no_grad(), _autocast(device, bf16):
        for _ in range(warmup):
            predict(model, x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            predict(model, x)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / runs * 1000


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


# ============================================================
# 4. REPORT
# ============================================================


def _fmt_ci(d, ci):
    return f"{d:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]"


def markdown_report(res):
    names = list(res["models"])
    lines = [
        "# PoC comparison — Cityscapes val",
        "",
        f"{res['n_images']} images, {NUM_CLASSES} classes, "
        f"{IMG_SIZE[0]}x{IMG_SIZE[1]}; generated {res['generated']} on {res['device']}.",
        "",
        "## Overall",
        "",
        "| | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
    ]
    rows = [
        ("Val mIoU", lambda m: f"{m['mIoU']:.4f}"),
        ("Pixel accuracy", lambda m: f"{m['pixel_accuracy']:.4f}"),
        ("Parameters", lambda m: f"{m['params_M']:.1f} M"),
        ("Accuracy precision", lambda m: m["precision"]),
        (
            "Latency fp32 (1 image)",
            lambda m: f"{m['latency_ms']['fp32']:.0f} ms" if m["latency_ms"] else "n/a",
        ),
        (
            "Latency bf16 (1 image)",
            lambda m: f"{m['latency_ms']['bf16']:.0f} ms" if m["latency_ms"] else "n/a",
        ),
    ]
    for label, f in rows:
        lines.append(
            f"| {label} | " + " | ".join(f(res["models"][n]) for n in names) + " |"
        )

    lines += [
        "",
        "## Per-class IoU",
        "",
        "| Class | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
    ]
    for c, cls in enumerate(CLASS_NAMES):
        lines.append(
            f"| {cls} | "
            + " | ".join(f"{res['models'][n]['iou'][cls]:.4f}" for n in names)
            + " |"
        )

    if res["comparisons"]:
        lines += [
            "",
            f"## Paired bootstrap ({res['bootstrap']['resamples']} resamples, "
            f"seed {res['bootstrap']['seed']}) — difference B − A, 95% CI",
            "",
            "| B − A | Δ mIoU [95% CI] | P(Δ ≤ 0) |",
            "|---|---|---|",
        ]
        for comp in res["comparisons"]:
            lines.append(
                f"| {comp['B']} − {comp['A']} | "
                f"{_fmt_ci(comp['delta_mIoU'], comp['ci_mIoU'])} | "
                f"{comp['p_delta_le_0']:.3f} |"
            )
        lines += [
            "",
            "Per-class Δ IoU [95% CI]:",
            "",
            "| Class | "
            + " | ".join(f"{c['B']} − {c['A']}" for c in res["comparisons"])
            + " |",
            "|---|" + "---|" * len(res["comparisons"]),
        ]
        for cls in CLASS_NAMES:
            lines.append(
                f"| {cls} | "
                + " | ".join(
                    _fmt_ci(c["delta_iou"][cls], c["ci_iou"][cls])
                    for c in res["comparisons"]
                )
                + " |"
            )
    lines += [
        "",
        "Latency: synthetic input, batch 1, prediction included; only ratios between "
        "models are meaningful. Bootstrap covers evaluation variance (which images), "
        "not training variance (one run per model).",
    ]
    return "\n".join(lines) + "\n"


# ============================================================
# 5. MAIN
# ============================================================


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for name, (opt, default, *_rest) in MODELS.items():
        p.add_argument(f"--{opt}", default=default, help=f"{name} checkpoint")
    p.add_argument("--bootstrap", type=int, default=2000, help="bootstrap resamples")
    p.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    p.add_argument("--latency-runs", type=int, default=30)
    p.add_argument("--no-latency", action="store_true", help="skip latency")
    p.add_argument("--limit", type=int, default=0, help="first N val images; 0 = all")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--out-dir", default="results")
    return p.parse_args()


def main():
    args = parse_args()
    # Windows consoles (cp1252) cannot print "−", "Δ"...; the files stay UTF-8
    sys.stdout.reconfigure(errors="replace")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    print("Device:", gpu)

    pairs = get_cityscapes_pairs(IMG_ROOT, MASK_ROOT, split="val")
    if args.limit:
        pairs = pairs[: args.limit]
    loader = DataLoader(
        CityscapesDataset(pairs, img_size=IMG_SIZE),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    results = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "device": gpu,
        "n_images": len(pairs),
        "versions": {"torch": torch.__version__},
        "models": {},
        "comparisons": [],
        "bootstrap": {"resamples": args.bootstrap, "seed": args.seed},
    }
    try:
        import transformers

        results["versions"]["transformers"] = transformers.__version__
    except ImportError:
        pass

    confs = {}
    for name, (opt, _default, load, predict, bf16) in MODELS.items():
        path = os.path.join(ROOT, getattr(args, opt))
        if not os.path.exists(path):
            print(f"\n[skip] {name}: checkpoint not found at {path}")
            continue
        print(f"\n=== {name} ({os.path.relpath(path, ROOT)}) ===")
        model = load(path).to(device).eval()

        t0 = time.perf_counter()
        confs[name] = evaluate(model, predict, bf16, loader, device)
        miou, iou, pa = scores(confs[name].sum(axis=0))
        print(f"mIoU {miou:.4f}  PixAcc {pa:.4f}  ({time.perf_counter() - t0:.0f} s)")

        lat = None
        if device == "cuda" and not args.no_latency:
            lat = {
                "fp32": latency_ms(model, predict, False, device, args.latency_runs),
                "bf16": latency_ms(model, predict, True, device, args.latency_runs),
            }
            print(f"latency fp32 {lat['fp32']:.0f} ms  bf16 {lat['bf16']:.0f} ms")

        results["models"][name] = {
            "checkpoint": os.path.relpath(path, ROOT).replace("\\", "/"),
            "sha256": sha256(path),
            "precision": "bf16" if bf16 and device == "cuda" else "fp32",
            "params_M": sum(p.numel() for p in model.parameters()) / 1e6,
            "mIoU": miou,
            "pixel_accuracy": pa,
            "iou": {c: float(v) for c, v in zip(CLASS_NAMES, iou)},
            "latency_ms": lat,
        }
        model.to("cpu")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    if not confs:
        sys.exit("No checkpoint found — nothing to evaluate.")

    # Pairs in table order: baseline -> Mask2Former, baseline -> EoMT, Mask2Former -> EoMT
    if args.bootstrap > 0:
        for a, b in combinations(list(confs), 2):
            print(f"\nBootstrap {b} - {a} ...")
            d_miou, d_iou = paired_bootstrap(
                confs[a], confs[b], args.bootstrap, args.seed
            )
            ma, ia, _ = scores(confs[a].sum(axis=0))
            mb, ib, _ = scores(confs[b].sum(axis=0))
            results["comparisons"].append(
                {
                    "A": a,
                    "B": b,
                    "delta_mIoU": mb - ma,
                    "ci_mIoU": _ci(d_miou),
                    "p_delta_le_0": float(np.mean(d_miou <= 0)),
                    "delta_iou": {
                        c: float(ib[k] - ia[k]) for k, c in enumerate(CLASS_NAMES)
                    },
                    "ci_iou": {c: _ci(d_iou[:, k]) for k, c in enumerate(CLASS_NAMES)},
                }
            )

    out_dir = os.path.join(ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    report = markdown_report(results)
    with open(os.path.join(out_dir, "poc_comparison.md"), "w", encoding="utf-8") as f:
        f.write(report)
    with open(os.path.join(out_dir, "poc_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    np.savez_compressed(
        os.path.join(out_dir, "poc_confusions.npz"),
        **{name.replace("-", "_"): c for name, c in confs.items()},
        image_paths=np.array([os.path.relpath(p[0], ROOT) for p in pairs]),
    )

    print("\n" + report)
    print(
        f"Saved to {os.path.relpath(out_dir, ROOT)}/poc_comparison.(md|json), poc_confusions.npz"
    )


if __name__ == "__main__":
    main()
