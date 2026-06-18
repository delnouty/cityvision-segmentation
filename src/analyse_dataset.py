import os
import sys
import csv
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm

# Full Cityscapes label set (labelId 0-33)
CITYSCAPES_CLASSES = {
    0:  "unlabeled",
    1:  "ego vehicle",
    2:  "rectification border",
    3:  "out of roi",
    4:  "static",
    5:  "dynamic",
    6:  "ground",
    7:  "road",
    8:  "sidewalk",
    9:  "parking",
    10: "rail track",
    11: "building",
    12: "wall",
    13: "fence",
    14: "guard rail",
    15: "bridge",
    16: "tunnel",
    17: "pole",
    18: "pole group",
    19: "traffic light",
    20: "traffic sign",
    21: "vegetation",
    22: "terrain",
    23: "sky",
    24: "person",
    25: "rider",
    26: "car",
    27: "truck",
    28: "bus",
    29: "caravan",
    30: "trailer",
    31: "train",
    32: "motorcycle",
    33: "bicycle",
}


def analyse_split(mask_dir, split, csv_writer):
    mask_paths = sorted(glob(os.path.join(mask_dir, split, "*", "*_gtFine_labelIds.png")))
    if not mask_paths:
        print(f"  No masks found in {split}")
        return

    pixel_counts = np.zeros(34, dtype=np.int64)
    image_counts = np.zeros(34, dtype=np.int64)

    for mask_path in tqdm(mask_paths, desc=split):
        raw = np.array(Image.open(mask_path))
        for label_id in range(34):
            count = int(np.sum(raw == label_id))
            pixel_counts[label_id] += count
            if count > 0:
                image_counts[label_id] += 1

    total_pixels = pixel_counts.sum()
    total_images = len(mask_paths)

    print(f"\n{'='*72}")
    print(f"Split: {split}  ({total_images} images, {total_pixels:,} total pixels)")
    print(f"{'='*72}")
    print(f"  {'ID':>3}  {'Class':<22}  {'Pixels':>12}  {'% pixels':>9}  {'Images':>7}")
    print(f"  {'-'*3}  {'-'*22}  {'-'*12}  {'-'*9}  {'-'*7}")
    for label_id in range(34):
        if pixel_counts[label_id] == 0:
            continue
        pct_px = 100.0 * pixel_counts[label_id] / total_pixels
        name = CITYSCAPES_CLASSES[label_id]
        print(f"  {label_id:>3}  {name:<22}  {pixel_counts[label_id]:>12,}  {pct_px:>8.2f}%  {image_counts[label_id]:>7}")
        csv_writer.writerow([split, label_id, name, int(pixel_counts[label_id]),
                             round(pct_px, 4), int(image_counts[label_id]), total_images])


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mask_root = os.path.join(project_root, "data/cityscapes/P8_Cityscapes_gtFine_trainvaltest/gtFine")
    csv_path = os.path.join(project_root, "dataset_analysis.csv")

    if not os.path.exists(mask_root):
        print("ERROR: mask_root not found:", mask_root)
        return

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "label_id", "class_name", "pixels", "pct_pixels", "images", "total_images"])
        for split in ("train", "val"):
            analyse_split(mask_root, split, writer)

    print(f"\nSaved to {csv_path}")


if __name__ == "__main__":
    main()
