"""Dataset constants and a thin bridge to the existing data pipeline.

The Cityscapes loading / weighted-sampling logic already lives in
``src/dataloader.py``; we reuse it rather than duplicate it.
"""

import os
import sys

import torch

# Reuse the project's existing dataloader without copying it.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dataloader import create_dataloaders  # noqa: E402  (path set above)

# ------------------------------------------------------------------
# Cityscapes labelIds → 8 foreground objects + background
# ------------------------------------------------------------------
TARGET_CLASSES = {
    7:  1,  # road
    11: 2,  # building
    21: 3,  # vegetation
    23: 4,  # sky
    24: 5,  # person
    26: 6,  # car
    20: 7,  # traffic_sign
    33: 8,  # bicycle
}

CLASS_NAMES = ["background", "road", "building", "vegetation",
               "sky", "person", "car", "traffic_sign", "bicycle"]

NUM_CLASSES = 9  # 0 = background + 8 objects


def remap_mask(mask: torch.Tensor) -> torch.Tensor:
    """mask: tensor HxW (Cityscapes labelIds) → remapped to 0..8."""
    new_mask = torch.zeros_like(mask)
    for src, dst in TARGET_CLASSES.items():
        new_mask[mask == src] = dst
    return new_mask


__all__ = [
    "create_dataloaders",
    "TARGET_CLASSES",
    "CLASS_NAMES",
    "NUM_CLASSES",
    "remap_mask",
]
