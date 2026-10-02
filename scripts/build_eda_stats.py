"""
scripts/build_eda_stats.py
==========================

Precomputes everything the PoC app's "Dataset" page shows, from the full local
Cityscapes copy, so the app itself needs only a small committed sample set.

What it does
------------
1. Pairs images and labelId masks exactly as training does
   (``get_cityscapes_pairs``: <split>/<city>/*_leftImg8bit.png, which also
   ignores the ``~BROMIUM`` sandbox copies found in some folders).
2. Counts images per split and per city (train / val / test).
3. For train and val, counts the pixels of each of the 9 classes (after the
   project's labelId -> 9-class remap) and the number of images containing
   each class.
4. Selects ~30 val images (10 per city) that cover every class, and saves them
   at the models' input size, 1024x512 (bilinear for images, nearest for
   masks), with 9-class masks (values 0-8), to ``poc_app/data/samples/``.
   The training dataloader resizes to this same size, so predictions on these
   files equal those of the evaluation.

Output
------
- ``poc_app/data/eda_stats.json``
- ``poc_app/data/samples/<city>/<id>_image.png`` and ``<id>_mask.png``

Usage
-----
    python scripts/build_eda_stats.py
    python scripts/build_eda_stats.py --per-city 10
"""

import argparse
import datetime
import json
import os
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from dataloader import get_cityscapes_pairs  # noqa: E402

from cityvision.constants import CLASS_NAMES, NUM_CLASSES, remap_labels  # noqa: E402

IMG_ROOT = os.path.join(
    ROOT, "data/cityscapes/P8_Cityscapes_leftImg8bit_trainvaltest/leftImg8bit"
)
MASK_ROOT = os.path.join(
    ROOT, "data/cityscapes/P8_Cityscapes_gtFine_trainvaltest/gtFine"
)
OUT_DIR = os.path.join(ROOT, "poc_app", "data")
SAMPLE_SIZE = (1024, 512)  # (W, H) = model input 512x1024

# Order in which classes get a sample image: rarest first, so they are covered
_PICK_ORDER = [8, 7, 5, 6, 4, 3, 2, 1]


def image_id(img_path):
    return os.path.basename(img_path).replace("_leftImg8bit.png", "")


def city_of(img_path):
    return os.path.basename(os.path.dirname(img_path))


def count_split(split):
    """Images per city, and per-image 9-class pixel counts (if labelled)."""
    pairs = get_cityscapes_pairs(IMG_ROOT, MASK_ROOT, split=split)
    cities = {}
    for img_path, _ in pairs:
        cities[city_of(img_path)] = cities.get(city_of(img_path), 0) + 1
    result = {"n_images": len(pairs), "cities": dict(sorted(cities.items()))}
    if split == "test":  # Cityscapes keeps test labels private
        return result, []

    per_image = []
    for img_path, mask_path in tqdm(pairs, desc=f"{split} masks"):
        labels = remap_labels(np.array(Image.open(mask_path)))
        counts = np.bincount(labels.ravel(), minlength=NUM_CLASSES)
        per_image.append((img_path, mask_path, counts))
    pixels = np.sum([c for *_, c in per_image], axis=0)
    with_class = np.sum([c > 0 for *_, c in per_image], axis=0)
    result["pixels"] = {n: int(v) for n, v in zip(CLASS_NAMES, pixels)}
    result["images_with"] = {n: int(v) for n, v in zip(CLASS_NAMES, with_class)}
    result["pixel_share"] = {
        n: round(float(v / pixels.sum()), 6) for n, v in zip(CLASS_NAMES, pixels)
    }
    return result, per_image


def select_samples(per_image, per_city):
    """
    Per city: one image per class (rarest classes first) where that class
    covers the largest share of pixels, then fill up with the images richest
    in the rare classes (person, traffic sign, bicycle).
    """
    by_city = {}
    for img_path, mask_path, counts in per_image:
        by_city.setdefault(city_of(img_path), []).append(
            (img_path, mask_path, counts / counts.sum())
        )
    chosen = []
    for city, items in sorted(by_city.items()):
        picked = []  # indices into items
        for c in _PICK_ORDER:
            if len(picked) >= per_city:
                break
            cands = [
                k for k, it in enumerate(items) if k not in picked and it[2][c] > 0
            ]
            if cands:
                picked.append(max(cands, key=lambda k: items[k][2][c]))
        rest = sorted(
            (k for k in range(len(items)) if k not in picked),
            key=lambda k: -(items[k][2][5] + items[k][2][7] + items[k][2][8]),
        )
        picked += rest[: per_city - len(picked)]
        chosen += [(city, *items[k]) for k in picked]
    return chosen


def save_samples(chosen):
    samples = []
    for city, img_path, mask_path, share in tqdm(chosen, desc="samples"):
        sid = image_id(img_path)
        rel_dir = os.path.join("samples", city)
        os.makedirs(os.path.join(OUT_DIR, rel_dir), exist_ok=True)
        img = Image.open(img_path).convert("RGB").resize(SAMPLE_SIZE, Image.BILINEAR)
        labels = remap_labels(np.array(Image.open(mask_path)))
        mask = Image.fromarray(labels).resize(SAMPLE_SIZE, Image.NEAREST)
        img_rel = os.path.join(rel_dir, f"{sid}_image.png").replace("\\", "/")
        mask_rel = os.path.join(rel_dir, f"{sid}_mask.png").replace("\\", "/")
        img.save(os.path.join(OUT_DIR, img_rel), optimize=True)
        mask.save(os.path.join(OUT_DIR, mask_rel), optimize=True)
        samples.append(
            {
                "id": sid,
                "city": city,
                "image": img_rel,
                "mask": mask_rel,
                "class_share": {
                    n: round(float(v), 5) for n, v in zip(CLASS_NAMES, share)
                },
            }
        )
    return samples


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--per-city", type=int, default=10, help="val samples per city")
    args = p.parse_args()

    if not os.path.isdir(IMG_ROOT):
        sys.exit(f"Cityscapes not found at {IMG_ROOT}")

    splits, val_images = {}, []
    for split in ("train", "val", "test"):
        splits[split], per_image = count_split(split)
        if split == "val":
            val_images = per_image

    chosen = select_samples(val_images, args.per_city)
    samples = save_samples(chosen)

    stats = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "source": "Cityscapes leftImg8bit + gtFine (labelIds), remapped to 9 classes",
        "original_size": [2048, 1024],
        "sample_size": list(SAMPLE_SIZE),
        "classes": CLASS_NAMES,
        "splits": splits,
        "samples": samples,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "eda_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=1)

    size = sum(
        os.path.getsize(os.path.join(OUT_DIR, s[k]))
        for s in samples
        for k in ("image", "mask")
    )
    print(
        f"\nSaved poc_app/data/eda_stats.json and {len(samples)} samples "
        f"({size / 2**20:.1f} MB)"
    )
    for split, s in splits.items():
        print(f"  {split:5s} {s['n_images']:5d} images, {len(s['cities'])} cities")


if __name__ == "__main__":
    main()
