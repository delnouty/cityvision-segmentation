"""
scripts/make_samples.py
=======================

Build a small, committable sample set for the Streamlit frontend:
**two image+mask pairs per city**, copied into ``data/samples/<city>/`` and
downscaled by half (≈1024x512) to keep the repo light.

Pairs are taken from the Cityscapes **val** split, because it has real
ground-truth masks (the official *test* split withholds them — its labelIds are
all ignore-labels). The mask is saved as a **colourised** RGB PNG
(``*_gtFine_color.png``) using the 9-class palette, so it is directly viewable
in colour and matches the predicted-mask colours.

The full dataset is gitignored; this curated subset is committed so the frontend
(and a cloud deployment) works without it and can show real image / real mask /
predicted mask.

Run from the project root:
    python scripts/make_samples.py
"""

import os
import shutil
from glob import glob

import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPLIT = "val"  # has real ground-truth masks
IMG_SRC = os.path.join(
    PROJECT_ROOT,
    "data/cityscapes/P8_Cityscapes_leftImg8bit_trainvaltest/leftImg8bit",
    SPLIT,
)
MASK_SRC = os.path.join(
    PROJECT_ROOT,
    "data/cityscapes/P8_Cityscapes_gtFine_trainvaltest/gtFine",
    SPLIT,
)
DEST = os.path.join(PROJECT_ROOT, "data", "samples")

PER_CITY = 2
SCALE = 0.5  # downscale factor (2048x1024 -> 1024x512)

# 9-class remap + palette (mirrors the training scripts / backend.inference).
TARGET_CLASSES = {7: 1, 11: 2, 21: 3, 23: 4, 24: 5, 26: 6, 20: 7, 33: 8}
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


def _mask_for(img_path: str) -> str:
    city = os.path.basename(os.path.dirname(img_path))
    base = os.path.basename(img_path).replace("_leftImg8bit.png", "")
    return os.path.join(MASK_SRC, city, base + "_gtFine_labelIds.png")


def _colorize(label_ids: np.ndarray) -> Image.Image:
    remapped = np.zeros_like(label_ids, dtype=np.uint8)
    for src, dst in TARGET_CLASSES.items():
        remapped[label_ids == src] = dst
    return Image.fromarray(PALETTE[remapped], mode="RGB")


def main() -> None:
    if not os.path.isdir(IMG_SRC):
        raise SystemExit(f"Split not found: {IMG_SRC}")

    # Start clean so re-runs don't leave stale cities behind.
    if os.path.isdir(DEST):
        shutil.rmtree(DEST)

    cities = sorted(
        d for d in os.listdir(IMG_SRC) if os.path.isdir(os.path.join(IMG_SRC, d))
    )
    total = 0
    for city in cities:
        imgs = sorted(glob(os.path.join(IMG_SRC, city, "*_leftImg8bit.png")))[:PER_CITY]
        out_dir = os.path.join(DEST, city)
        os.makedirs(out_dir, exist_ok=True)
        for src in imgs:
            mask_src = _mask_for(src)
            if not os.path.exists(mask_src):
                print(f"  ! skip (no mask): {os.path.basename(src)}")
                continue

            img = Image.open(src).convert("RGB")
            w, h = img.size
            size = (int(w * SCALE), int(h * SCALE))
            img = img.resize(size, Image.LANCZOS)

            # labelIds -> resize (NEAREST, never interpolate) -> colourise
            mask = Image.open(mask_src).resize(size, Image.NEAREST)
            color = _colorize(np.array(mask))

            base = os.path.basename(src)
            img.save(os.path.join(out_dir, base), format="PNG")
            color.save(
                os.path.join(
                    out_dir, base.replace("_leftImg8bit.png", "_gtFine_color.png")
                ),
                format="PNG",
            )
            total += 1
            print(f"  {city}: {base}  (+ colour mask)  {size[0]}x{size[1]}")
    print(f"\nWrote {total} image+mask pairs to {DEST}")


if __name__ == "__main__":
    main()
