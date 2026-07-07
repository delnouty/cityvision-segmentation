"""
scripts/visualize_best.py
=========================

Qualitative evaluation of the *best* trained model (NOT a pytest test — it was
previously misnamed ``src/test_the_best.py``).

What it does
------------
1. Queries the MLflow backend (``sqlite:///mlflow.db``, experiment
   ``urban-segmentation``) for the run with the highest ``best_val_mIoU``
   whose checkpoint is available in ``backend/model/``.
2. Rebuilds that architecture (via ``cityvision.models``) and loads its weights.
3. Samples N random examples from a split using the *same* dataloader as
   training (``src/dataloader.py``) so preprocessing is identical.
4. Renders, per example: original image | ground-truth mask | prediction,
   saved to ``results/best_prediction.png``.

Note on the test split
----------------------
Cityscapes withholds the test ground truth (labelIds are only ignore-labels),
so with ``--split test`` the middle panel is all-background. Use ``--split val``
for real ground-truth masks.

Usage
-----
    python scripts/visualize_best.py --split val
    python scripts/visualize_best.py --split val --num 6 --seed 0
"""

import os
import sys
import argparse

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")  # save to file without needing a display
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402

import mlflow  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

# Make the shared package and the training dataloader importable.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dataloader import get_cityscapes_pairs, CityscapesDataset  # noqa: E402
from cityvision.constants import CLASS_NAMES, NUM_CLASSES, PALETTE  # noqa: E402
from cityvision.models import build_model, remap_mask  # noqa: E402

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


def colorize(label: np.ndarray) -> np.ndarray:
    """(H, W) class indices -> (H, W, 3) RGB image using PALETTE."""
    return PALETTE[label]


def denormalize(img_tensor: torch.Tensor) -> np.ndarray:
    """(3, H, W) normalized tensor -> (H, W, 3) uint8 RGB for display."""
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


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


def main():
    parser = argparse.ArgumentParser(
        description="Visualize the best model on random examples."
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["test", "val", "train"],
        help="Dataset split to sample from (default: val — has real masks).",
    )
    parser.add_argument(
        "--num", type=int, default=4, help="Number of random examples (default: 4)."
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
        help="Output image path (default: <project_root>/results/best_prediction.png).",
    )
    args = parser.parse_args()

    img_root = os.path.join(
        PROJECT_ROOT,
        "data/cityscapes/P8_Cityscapes_leftImg8bit_trainvaltest/leftImg8bit",
    )
    mask_root = os.path.join(
        PROJECT_ROOT, "data/cityscapes/P8_Cityscapes_gtFine_trainvaltest/gtFine"
    )
    output_path = args.output or os.path.join(
        PROJECT_ROOT, "results", "best_prediction.png"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- 1. pick the best model from the MLflow backend ---
    arch, ckpt_path, miou = select_best_model(PROJECT_ROOT)
    print(f"Best model: {arch}  (best_val_mIoU={miou:.4f})")
    print(f"Checkpoint: {ckpt_path}")

    model = build_model(arch, NUM_CLASSES).to(device)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # --- 2. build the dataset with the SAME dataloader as training ---
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
