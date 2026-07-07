"""
cityvision.constants
=====================

Single source of truth for the 9-class taxonomy and colour palette.
Torch-free (only numpy) so any consumer — including the lean frontend — can
import it without pulling in PyTorch.
"""

import numpy as np

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
NUM_CLASSES = 9  # 0 = background + 8 objects

# Raw Cityscapes labelId -> 9-class training scheme.
TARGET_CLASSES = {
    7: 1,  # road
    11: 2,  # building
    21: 3,  # vegetation
    23: 4,  # sky
    24: 5,  # person
    26: 6,  # car
    20: 7,  # traffic_sign
    33: 8,  # bicycle
}

# Cityscapes-style RGB palette, one colour per class index (0-8).
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


def remap_labels(raw: np.ndarray) -> np.ndarray:
    """Map raw Cityscapes labelIds to the 9-class scheme (numpy, uint8)."""
    remapped = np.zeros_like(raw, dtype=np.uint8)
    for src, dst in TARGET_CLASSES.items():
        remapped[raw == src] = dst
    return remapped
