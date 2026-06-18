import os
from glob import glob

import numpy as np
from PIL import Image

MASK_ROOT = "data/cityscapes/P8_Cityscapes_gtFine_trainvaltest/gtFine/train"

# твои 8 целевых классов: id → имя
TARGET_CLASSES = {
    19: "traffic light",
    20: "traffic sign",
    24: "person",
    26: "car",
    27: "truck",
    28: "bus",
    32: "motorcycle",
    33: "bicycle",
}


def iter_masks(mask_root):
    cities = os.listdir(mask_root)
    for city in cities:
        city_dir = os.path.join(mask_root, city)
        for mask_path in glob(os.path.join(city_dir, "*_gtFine_labelIds.png")):
            yield mask_path


def main():
    counts = {cid: 0 for cid in TARGET_CLASSES.keys()}
    total_masks = 0

    for mask_path in iter_masks(MASK_ROOT):
        total_masks += 1
        mask = np.array(Image.open(mask_path))
        unique = set(np.unique(mask).tolist())

        for cid in TARGET_CLASSES.keys():
            if cid in unique:
                counts[cid] += 1

    print(f"Всего масок просмотрено: {total_masks}\n")
    print("Статистика по твоим 8 классам:\n")
    for cid, name in TARGET_CLASSES.items():
        print(f"{cid:2d}: {name:13s} — в {counts[cid]} масках")

    missing = [name for cid, name in TARGET_CLASSES.items() if counts[cid] == 0]
    if missing:
        print("\nВНИМАНИЕ: эти классы ни разу не встретились:")
        for name in missing:
            print("  -", name)
    else:
        print("\nВсе 8 классов присутствуют в датасете (но не на каждом изображении).")


if __name__ == "__main__":
    main()
