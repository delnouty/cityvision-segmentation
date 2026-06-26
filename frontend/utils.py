"""
frontend/utils.py
=================

Pure, Streamlit-free helpers for the CityVision frontend: dataset listing,
ground-truth colourisation, and API client calls. Keeping these out of the
Streamlit script makes them unit-testable.
"""

import os
import sys
import glob

import numpy as np
import requests
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Class names + palette (mirrors backend/inference.py). Defined locally so the
# frontend has NO dependency on torch — it stays a lean image.
CLASS_NAMES = [
    "background",
    "road",
    "building",
    "vegetation",
    "sky",
    "person",
    "car",
    "traffic_sign",
    "bicycle",
]
NUM_CLASSES = 9
PALETTE = np.array(
    [
        (0, 0, 0),  # background
        (128, 64, 128),  # road
        (70, 70, 70),  # building
        (107, 142, 35),  # vegetation
        (70, 130, 180),  # sky
        (220, 20, 60),  # person
        (0, 0, 142),  # car
        (220, 220, 0),  # traffic_sign
        (119, 11, 32),  # bicycle
    ],
    dtype=np.uint8,
)

# Raw Cityscapes labelId -> 9-class training scheme (mirrors the training scripts).
TARGET_CLASSES = {7: 1, 11: 2, 21: 3, 23: 4, 24: 5, 26: 6, 20: 7, 33: 8}

IMG_ROOT = os.path.join(
    PROJECT_ROOT, "data/cityscapes/P8_Cityscapes_leftImg8bit_trainvaltest/leftImg8bit"
)
MASK_ROOT = os.path.join(
    PROJECT_ROOT, "data/cityscapes/P8_Cityscapes_gtFine_trainvaltest/gtFine"
)

# Backend API base URL — overridable via env var (e.g. in Docker / Azure).
DEFAULT_API = os.environ.get("CITYVISION_API", "http://localhost:8000")

# Curated sample set committed to the repo (built by scripts/make_samples.py):
# two image+mask pairs per city (from val, which has real masks), so the frontend
# works without the full dataset and can show real image / real mask / prediction.
SAMPLE_DIR = os.path.join(PROJECT_ROOT, "data", "samples")


def image_id(img_path: str) -> str:
    """e.g. .../berlin/berlin_000000_000019_leftImg8bit.png -> berlin_000000_000019"""
    return os.path.basename(img_path).replace("_leftImg8bit.png", "")


def build_pairs(split: str) -> dict:
    """{image_id: (img_path, mask_path)} for a full-dataset split.

    Imports the training dataloader lazily (it pulls in torch) so that the
    common path — bundled sample images — has no heavy dependency.
    """
    src_dir = os.path.join(PROJECT_ROOT, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from dataloader import get_cityscapes_pairs  # lazy: needs torch

    pairs = get_cityscapes_pairs(IMG_ROOT, MASK_ROOT, split=split)
    return {image_id(ip): (ip, mp) for ip, mp in pairs}


def list_sample_images() -> dict:
    """{image_id: (image_path, mask_path)} for the bundled sample set in data/samples/.

    A small curated subset (two image+mask pairs per city) committed to the repo
    so the frontend works without the full Cityscapes dataset and can show the
    real image, the real mask, and the predicted mask.
    """
    out = {}
    if not os.path.isdir(SAMPLE_DIR):
        return out
    pattern = os.path.join(SAMPLE_DIR, "*", "*_leftImg8bit.png")
    for img_path in sorted(glob.glob(pattern)):
        mask_path = img_path.replace("_leftImg8bit.png", "_gtFine_color.png")
        out[image_id(img_path)] = (
            img_path,
            mask_path if os.path.exists(mask_path) else None,
        )
    return out


def remap_labels(raw: np.ndarray) -> np.ndarray:
    """Map raw Cityscapes labelIds to the 9-class scheme (uint8)."""
    remapped = np.zeros_like(raw, dtype=np.uint8)
    for src, dst in TARGET_CLASSES.items():
        remapped[raw == src] = dst
    return remapped


def colorize_gt(mask_path: str):
    """Return (colour RGB mask, has_real_labels) for display.

    Accepts either an already-colourised RGB mask (the bundled samples,
    `*_gtFine_color.png`) or a raw labelIds mask (the dataset splits), which is
    remapped to the 9-class palette. A test-split labelIds mask remaps to
    all-background, so has_real_labels is False there."""
    raw = np.array(Image.open(mask_path))
    if raw.ndim == 3:  # already a colour mask
        rgb = raw[..., :3].astype(np.uint8)
        return Image.fromarray(rgb, mode="RGB"), bool(rgb.any())
    remapped = remap_labels(raw)
    return Image.fromarray(PALETTE[remapped], mode="RGB"), bool(remapped.any())


# --- API client ---


def api_health(api_url: str) -> dict:
    r = requests.get(f"{api_url}/health", timeout=5)
    r.raise_for_status()
    return r.json()


def api_predict(
    api_url: str, img_path: str, fmt: str = "color", alpha: float = 0.5
) -> bytes:
    """Predicted mask rendered as PNG (visualisation). The raw mask-as-JSON
    is available at POST /predict."""
    with open(img_path, "rb") as f:
        files = {"file": (os.path.basename(img_path), f, "image/png")}
        r = requests.post(
            f"{api_url}/predict/image",
            params={"format": fmt, "alpha": alpha},
            files=files,
            timeout=120,
        )
    r.raise_for_status()
    return r.content


def api_classes(api_url: str, img_path: str) -> dict:
    with open(img_path, "rb") as f:
        files = {"file": (os.path.basename(img_path), f, "image/png")}
        r = requests.post(f"{api_url}/predict/classes", files=files, timeout=120)
    r.raise_for_status()
    return r.json()
