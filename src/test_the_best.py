"""
test_the_best.py
================

Qualitative test of the *best* trained model.

What it does
------------
1. Queries the MLflow backend (``sqlite:///mlflow.db``, experiment
   ``urban-segmentation``) for the run with the highest ``best_val_mIoU``
   whose checkpoint is available in ``backend/model/``.
2. Rebuilds that architecture and loads its weights from the backend.
3. Samples 4 random examples from the ``test`` split using the *same*
   dataloader as training (``src/dataloader.py``) so the preprocessing
   (resize + ImageNet normalization) is identical.
4. Renders, for each example, three panels side by side:
   original image | ground-truth mask | model prediction.
   The figure is saved to ``test_the_best_predictions.png``.

Note on the test split
----------------------
Official Cityscapes withholds the test ground truth, so the test
``*_labelIds.png`` files only contain {0, 1, 3}. After the training
remapping these become all-"background", i.e. the middle "mask" panel is
not a real annotation. Pass ``--split val`` to compare against real
ground-truth masks instead.

Usage
-----
    python src/test_the_best.py                 # 4 random test examples
    python src/test_the_best.py --split val      # real ground-truth masks
    python src/test_the_best.py --num 6 --seed 0
"""

import os
import sys
import argparse

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")  # save to file without needing a display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import mlflow
from mlflow.tracking import MlflowClient

sys.path.insert(0, os.path.dirname(__file__))
from dataloader import get_cityscapes_pairs, CityscapesDataset

# ============================================================
# Constants — mirror the training scripts
# ============================================================

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

# Cityscapes-style RGB palette, one colour per remapped class (0-8).
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

# ImageNet normalization used by the dataloader — needed to de-normalize
# images back to a displayable RGB range.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

EXPERIMENT = "urban-segmentation"

# architecture (as logged in the "architecture" MLflow param) -> checkpoint file
CKPT_BY_ARCH = {
    "UNet": "unet_best.pth",
    "ResNet34-UNet": "resnet_best.pth",
    "ResNet50-UNet": "resnet50_best.pth",
    "VGG16-UNet": "vgg_best.pth",
    "SegNet": "segnet_best.pth",
}


# ============================================================
# Helpers
# ============================================================


def remap_mask(mask: torch.Tensor) -> torch.Tensor:
    """Map raw Cityscapes labelIds to the 9-class training scheme."""
    new_mask = torch.zeros_like(mask)
    for src, dst in TARGET_CLASSES.items():
        new_mask[mask == src] = dst
    return new_mask


def colorize(label: np.ndarray) -> np.ndarray:
    """(H, W) class indices -> (H, W, 3) RGB image using PALETTE."""
    return PALETTE[label]


def denormalize(img_tensor: torch.Tensor) -> np.ndarray:
    """(3, H, W) normalized tensor -> (H, W, 3) uint8 RGB for display."""
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def build_model(arch: str, num_classes: int) -> torch.nn.Module:
    """Instantiate the model class matching the logged architecture name."""
    if arch == "UNet":
        from training import UNet

        return UNet(num_classes=num_classes)
    if arch == "ResNet34-UNet":
        from training_resnet import ResNetUNet

        return ResNetUNet(num_classes=num_classes, pretrained=False)
    if arch == "ResNet50-UNet":
        from training_resnet50 import ResNet50UNet

        return ResNet50UNet(num_classes=num_classes, pretrained=False)
    if arch == "VGG16-UNet":
        from training_vgg import VGGUNet

        return VGGUNet(num_classes=num_classes, pretrained=False)
    if arch == "SegNet":
        from training_segnet import SegNet

        return SegNet(num_classes=num_classes)
    raise ValueError(f"Unknown architecture: {arch!r}")


def select_best_model(project_root: str):
    """
    Return (architecture, checkpoint_path, best_val_mIoU) for the run with the
    highest best_val_mIoU whose checkpoint exists in backend/model/.
    """
    mlflow.set_tracking_uri(f"sqlite:///{os.path.join(project_root, 'mlflow.db')}")
    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        raise RuntimeError(f"MLflow experiment {EXPERIMENT!r} not found.")

    runs = client.search_runs(
        [exp.experiment_id],
        order_by=["metrics.best_val_mIoU DESC"],
    )

    model_dir = os.path.join(project_root, "backend", "model")
    for run in runs:
        arch = run.data.params.get("architecture")
        miou = run.data.metrics.get("best_val_mIoU")
        if arch is None or miou is None or arch not in CKPT_BY_ARCH:
            continue
        ckpt = os.path.join(model_dir, CKPT_BY_ARCH[arch])
        if os.path.exists(ckpt):
            return arch, ckpt, miou

    raise RuntimeError(
        "No run with a logged architecture has a matching checkpoint in "
        f"{model_dir}. Train a model first."
    )


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="Visualize the best model on random examples."
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["test", "val", "train"],
        help="Dataset split to sample from (default: test).",
    )
    parser.add_argument(
        "--num", type=int, default=2, help="Number of random examples (default: 2)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for example selection (default: 42).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output image path (default: <project_root>/test_the_best_predictions.png).",
    )
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    img_root = os.path.join(
        project_root,
        "data/cityscapes/P8_Cityscapes_leftImg8bit_trainvaltest/leftImg8bit",
    )
    mask_root = os.path.join(
        project_root, "data/cityscapes/P8_Cityscapes_gtFine_trainvaltest/gtFine"
    )
    output_path = args.output or os.path.join(
        project_root, "test_the_best_predictions.png"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- 1. pick the best model from the MLflow backend ---
    arch, ckpt_path, miou = select_best_model(project_root)
    print(f"Best model: {arch}  (best_val_mIoU={miou:.4f})")
    print(f"Checkpoint: {ckpt_path}")

    model = build_model(arch, NUM_CLASSES).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # --- 2. build the test dataset with the SAME dataloader as training ---
    pairs = get_cityscapes_pairs(img_root, mask_root, split=args.split)
    if not pairs:
        raise RuntimeError(f"No image/mask pairs found for split={args.split!r}.")
    dataset = CityscapesDataset(pairs, img_size=(512, 1024))

    if args.split == "test":
        print(
            "NOTE: the Cityscapes 'test' split has no real ground truth — "
            "the middle 'Ground truth' panel will be all-background. "
            "Use --split val for real masks."
        )

    # --- 3. pick N random examples ---
    n = min(args.num, len(dataset))
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(dataset), size=n, replace=False).tolist()
    print(f"Sampling {n} random examples from '{args.split}' (indices: {indices})")

    # --- 4. render image | ground truth | prediction ---
    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Image", "Ground truth", f"Prediction ({arch})"]

    with torch.no_grad():
        for row, idx in enumerate(indices):
            img_tensor, raw_mask = dataset[idx]
            gt = remap_mask(raw_mask).cpu().numpy()

            logits = model(img_tensor.unsqueeze(0).to(device))
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()

            panels = [denormalize(img_tensor), colorize(gt), colorize(pred)]
            city = os.path.basename(os.path.dirname(pairs[idx][0]))
            for col, (panel, title) in enumerate(zip(panels, col_titles)):
                ax = axes[row, col]
                ax.imshow(panel)
                ax.axis("off")
                if row == 0:
                    ax.set_title(title, fontsize=14)
                if col == 0:
                    ax.set_ylabel(city, fontsize=11)

    # Shared legend for the class palette (two rows so labels stay readable).
    handles = [
        mpatches.Patch(
            facecolor=np.array(PALETTE[i]) / 255.0,
            edgecolor="black",
            linewidth=0.5,
            label=CLASS_NAMES[i],
        )
        for i in range(NUM_CLASSES)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        fontsize=12,
        frameon=True,
        title="Classes",
        title_fontsize=13,
        handlelength=1.6,
        handleheight=1.6,
        columnspacing=1.8,
        bbox_to_anchor=(0.5, 0.0),
    )

    fig.suptitle(
        f"Best model: {arch}  (val mIoU={miou:.4f})  —  split: {args.split}",
        fontsize=15,
    )
    fig.tight_layout(rect=[0, 0.10, 1, 0.97])
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
