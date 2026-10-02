"""
The three segmentation models, loaded in-process (no API needed locally).

Loaders, prediction functions and precisions come from
``scripts/evaluate_poc.py``, and preprocessing from the training dataloader,
so a prediction here is the one the evaluation scored.
"""

import os
import time

import numpy as np
import streamlit as st
import torch

import common  # noqa: F401  (sets sys.path)
from dataloader import CityscapesDataset
from evaluate_poc import MODELS

INPUT_SIZE = (512, 1024)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Same transform object as training/evaluation: resize, ToTensor, ImageNet norm
_TRANSFORM = CityscapesDataset([], img_size=INPUT_SIZE).img_transform


def checkpoint_path(name):
    _opt, default, *_ = MODELS[name]
    return os.path.join(common.ROOT, default)


def available_models():
    return [n for n in common.MODEL_ORDER if os.path.exists(checkpoint_path(n))]


@st.cache_resource(show_spinner=False)
def load_model(name):
    _opt, _default, load, predict_fn, bf16 = MODELS[name]
    model = load(checkpoint_path(name)).to(DEVICE).eval()
    # Warm-up, so the first displayed inference time is not the kernel set-up
    x = torch.zeros(1, 3, *INPUT_SIZE, device=DEVICE)
    with (
        torch.no_grad(),
        torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=bf16 and DEVICE == "cuda"
        ),
    ):
        for _ in range(2):
            predict_fn(model, x)
    return model


def predict(name, image):
    """PIL image -> (labels (512, 1024) uint8, inference time in ms)."""
    _opt, _default, _load, predict_fn, bf16 = MODELS[name]
    model = load_model(name)
    x = _TRANSFORM(image).unsqueeze(0).to(DEVICE)
    with (
        torch.no_grad(),
        torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=bf16 and DEVICE == "cuda"
        ),
    ):
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        labels = predict_fn(model, x)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
    return labels[0].cpu().numpy().astype(np.uint8), ms


def per_image_scores(pred, gt, num_classes=9):
    """Pixel accuracy, IoU per class (NaN if absent from both), mean IoU."""
    iou = np.full(num_classes, np.nan)
    for c in range(num_classes):
        p, g = pred == c, gt == c
        union = np.logical_or(p, g).sum()
        if union:
            iou[c] = np.logical_and(p, g).sum() / union
    return float((pred == gt).mean()), iou, float(np.nanmean(iou))
