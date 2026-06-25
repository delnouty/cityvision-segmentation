"""
frontend/utils.py
=================

Pure, Streamlit-free helpers for the CityVision frontend: dataset listing,
ground-truth colourisation, and API client calls. Keeping these out of the
Streamlit script makes them unit-testable.
"""

import os
import sys

import numpy as np
import requests
from PIL import Image

# --- make project modules importable ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dataloader import get_cityscapes_pairs  # same dataloader as training

# PALETTE/CLASS_NAMES are re-exported here for the Streamlit UI (utils.CLASS_NAMES).
from backend.inference import PALETTE, CLASS_NAMES  # noqa: F401  same colours as model

# Raw Cityscapes labelId -> 9-class training scheme (mirrors the training scripts).
TARGET_CLASSES = {7: 1, 11: 2, 21: 3, 23: 4, 24: 5, 26: 6, 20: 7, 33: 8}

IMG_ROOT = os.path.join(
    PROJECT_ROOT, "data/cityscapes/P8_Cityscapes_leftImg8bit_trainvaltest/leftImg8bit"
)
MASK_ROOT = os.path.join(
    PROJECT_ROOT, "data/cityscapes/P8_Cityscapes_gtFine_trainvaltest/gtFine"
)

DEFAULT_API = "http://localhost:8000"


def image_id(img_path: str) -> str:
    """e.g. .../berlin/berlin_000000_000019_leftImg8bit.png -> berlin_000000_000019"""
    return os.path.basename(img_path).replace("_leftImg8bit.png", "")


def build_pairs(split: str) -> dict:
    """{image_id: (img_path, mask_path)} for the split (same dataloader as training)."""
    pairs = get_cityscapes_pairs(IMG_ROOT, MASK_ROOT, split=split)
    return {image_id(ip): (ip, mp) for ip, mp in pairs}


def remap_labels(raw: np.ndarray) -> np.ndarray:
    """Map raw Cityscapes labelIds to the 9-class scheme (uint8)."""
    remapped = np.zeros_like(raw, dtype=np.uint8)
    for src, dst in TARGET_CLASSES.items():
        remapped[raw == src] = dst
    return remapped


def colorize_gt(mask_path: str):
    """Read a gtFine labelIds mask, remap to 9 classes, colourise.
    Returns (PIL.Image, has_real_labels). The test split remaps to all-background."""
    raw = np.array(Image.open(mask_path))
    remapped = remap_labels(raw)
    has_real = bool(remapped.any())
    return Image.fromarray(PALETTE[remapped], mode="RGB"), has_real


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
