import os
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import numpy as np

# Remapping + weights mirrored from training scripts (used for sampler only)
_TARGET_CLASSES = {7: 1, 11: 2, 21: 3, 23: 4, 24: 5, 26: 6, 20: 7, 33: 8}
_CLASS_WEIGHTS = [0.5, 1.0, 1.0, 1.0, 1.0, 2.5, 1.0, 3.0, 3.0]


def get_cityscapes_pairs(img_root, mask_root, split="train"):
    img_dir = os.path.join(img_root, split)
    mask_dir = os.path.join(mask_root, split)

    img_paths = sorted(glob(os.path.join(img_dir, "*", "*_leftImg8bit.png")))
    pairs = []

    for img_path in img_paths:
        city = os.path.basename(os.path.dirname(img_path))
        filename = os.path.basename(img_path)
        base = filename.replace("_leftImg8bit.png", "")
        mask_path = os.path.join(mask_dir, city, base + "_gtFine_labelIds.png")

        if os.path.exists(mask_path):
            pairs.append((img_path, mask_path))
        else:
            print("WARNING: mask not found for:", img_path)
            print("  Expected mask:", mask_path)

    return pairs


class CityscapesDataset(Dataset):
    def __init__(self, pairs, img_size=(512, 1024)):
        self.pairs = pairs
        self.img_size = img_size

        self.img_transform = T.Compose(
            [
                T.Resize(img_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.mask_transform = T.Compose(
            [
                T.Resize(img_size, interpolation=T.InterpolationMode.NEAREST),
            ]
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        img = self.img_transform(img)
        mask = self.mask_transform(mask)
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask


def _compute_sample_weights(dataset: CityscapesDataset) -> list:
    """
    Per-image sampling weight = sum of class_weights for every unique
    remapped class present in that mask.

    Images containing traffic_sign or bicycle (w=3.0) or person (w=2.5) are
    sampled more often than background-only frames (w=0.5), addressing class
    imbalance at the batch level.
    """
    print(f"Computing class-balanced sample weights for {len(dataset)} images...")
    weights = []
    for _, mask_path in dataset.pairs:
        raw = np.array(Image.open(mask_path))
        remapped = np.zeros_like(raw, dtype=np.int32)
        for src, dst in _TARGET_CLASSES.items():
            remapped[raw == src] = dst
        unique_classes = np.unique(remapped)
        non_bg_classes = [c for c in unique_classes if c != 0]
        if len(non_bg_classes) > 0:
            w = max(_CLASS_WEIGHTS[c] for c in non_bg_classes)
        else:
            w = _CLASS_WEIGHTS[0]
        weights.append(float(w))
    print(f"  Done. weight range [{min(weights):.1f}, {max(weights):.1f}]")
    return weights


def create_dataloaders(
    img_root,
    mask_root,
    batch_size=4,
    balanced=False,
    img_size=(512, 1024),
    num_workers=0,
):
    """
    Args:
        balanced: if True, use WeightedRandomSampler so that batches are
                  enriched with images that contain rare classes.
        img_size: (H, W) the images/masks are resized to. Higher resolution
                  helps small classes (traffic signs, bicycles) but uses more
                  GPU memory — reduce batch_size if you hit OOM.
        num_workers: decoding a 2048x1024 Cityscapes PNG costs ~130 ms, so with
                  the default 0 the main process spends ~6 min per epoch just
                  loading data while the GPU idles. Set 4 to overlap the two.
                  Workers are safe on Windows because CityscapesDataset lives in
                  this module (spawn re-imports it), but they cannot be used
                  from a plain script without an `if __name__ == "__main__"`
                  guard.
    """
    train_pairs = get_cityscapes_pairs(img_root, mask_root, split="train")
    val_pairs = get_cityscapes_pairs(img_root, mask_root, split="val")

    train_ds = CityscapesDataset(train_pairs, img_size=img_size)
    val_ds = CityscapesDataset(val_pairs, img_size=img_size)

    # Keeping workers alive between epochs avoids re-paying the spawn cost
    # (~2 s each on Windows) 20 times over.
    worker_kwargs = (
        {"persistent_workers": True, "prefetch_factor": 2} if num_workers > 0 else {}
    )

    if balanced:
        sample_weights = _compute_sample_weights(train_ds)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        # shuffle=True is mutually exclusive with sampler
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            **worker_kwargs,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            **worker_kwargs,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        **worker_kwargs,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    img_root = "data/cityscapes/P8_Cityscapes_leftImg8bit_trainvaltest/leftImg8bit"
    mask_root = "data/cityscapes/P8_Cityscapes_gtFine_trainvaltest/gtFine"

    train_loader, val_loader = create_dataloaders(img_root, mask_root, balanced=True)

    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))

    for img, mask in train_loader:
        print("Image shape:", img.shape)
        print("Mask shape:", mask.shape)
        break
