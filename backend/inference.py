"""
backend/inference.py
====================

Model loading + preprocessing + prediction for the CityVision segmentation API.

The trained checkpoints in ``backend/model/`` are plain ``state_dict``s produced
by the ``src/training_*.py`` scripts, so we rebuild the **exact same** model
classes from ``src/`` to load them, and we apply the **same** ImageNet
preprocessing as ``src/dataloader.py`` so inference matches training.
"""

import os
import sys
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Make the shared cityvision package importable.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Single source of truth for classes/palette and model architectures.
from cityvision.constants import CLASS_NAMES, NUM_CLASSES, PALETTE  # noqa: E402
from cityvision.models import build_model  # noqa: E402,F401 (re-exported)

# ============================================================
# Constants
# ============================================================

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Inference resolution (H, W) — same as training.
IMG_SIZE = (512, 1024)

MODEL_DIR = os.path.join(_PROJECT_ROOT, "backend", "model")

# architecture (as logged in MLflow) -> checkpoint filename
CKPT_BY_ARCH = {
    "UNet": "unet_best.pth",
    "ResNet34-UNet": "resnet_best.pth",
    "ResNet50-UNet": "resnet50_best.pth",
    "VGG16-UNet": "vgg_best.pth",
    "SegNet": "segnet_best.pth",
}
# Fallback preference when we cannot ask MLflow which run scored best.
_ARCH_PRIORITY = ["ResNet50-UNet", "ResNet34-UNet", "VGG16-UNet", "UNet", "SegNet"]


# ============================================================
# Checkpoint selection
# ============================================================


def _arch_from_filename(fname: str) -> Optional[str]:
    for arch, f in CKPT_BY_ARCH.items():
        if f == os.path.basename(fname):
            return arch
    return None


def _best_from_mlflow() -> Optional[Tuple[str, str, float]]:
    """Ask the MLflow backend for the best run whose checkpoint exists. Best-effort."""
    db = os.path.join(_PROJECT_ROOT, "mlflow.db")
    if not os.path.exists(db):
        return None
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(f"sqlite:///{db}")
        client = MlflowClient()
        exp = client.get_experiment_by_name("urban-segmentation")
        if exp is None:
            return None
        for run in client.search_runs(
            [exp.experiment_id], order_by=["metrics.best_val_mIoU DESC"]
        ):
            arch = run.data.params.get("architecture")
            miou = run.data.metrics.get("best_val_mIoU")
            if arch in CKPT_BY_ARCH and miou is not None:
                ckpt = os.path.join(MODEL_DIR, CKPT_BY_ARCH[arch])
                if os.path.exists(ckpt):
                    return arch, ckpt, float(miou)
    except Exception:
        return None
    return None


def select_checkpoint() -> Tuple[str, str, Optional[float]]:
    """
    Decide which model to serve, in priority order:
      1. CITYVISION_CHECKPOINT env var (explicit .pth path)
      2. CITYVISION_ARCH env var (architecture name)
      3. best run recorded in mlflow.db (if present)
      4. first available checkpoint by quality priority
    Returns (architecture, checkpoint_path, best_val_mIoU or None).
    """
    env_ckpt = os.environ.get("CITYVISION_CHECKPOINT")
    if env_ckpt:
        if not os.path.exists(env_ckpt):
            raise FileNotFoundError(f"CITYVISION_CHECKPOINT not found: {env_ckpt}")
        arch = os.environ.get("CITYVISION_ARCH") or _arch_from_filename(env_ckpt)
        if arch is None:
            raise ValueError(
                "Could not infer architecture from checkpoint name; "
                "set CITYVISION_ARCH."
            )
        return arch, env_ckpt, None

    env_arch = os.environ.get("CITYVISION_ARCH")
    if env_arch:
        if env_arch not in CKPT_BY_ARCH:
            raise ValueError(f"Unknown CITYVISION_ARCH={env_arch!r}.")
        ckpt = os.path.join(MODEL_DIR, CKPT_BY_ARCH[env_arch])
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"No checkpoint for {env_arch}: {ckpt}")
        return env_arch, ckpt, None

    best = _best_from_mlflow()
    if best is not None:
        return best

    for arch in _ARCH_PRIORITY:
        ckpt = os.path.join(MODEL_DIR, CKPT_BY_ARCH[arch])
        if os.path.exists(ckpt):
            return arch, ckpt, None

    raise RuntimeError(f"No model checkpoints found in {MODEL_DIR}.")


# ============================================================
# Segmenter — load once, predict many
# ============================================================


class Segmenter:
    """Loads the selected model once and runs segmentation on PIL images."""

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.arch, self.checkpoint_path, self.val_miou = select_checkpoint()

        self.model = build_model(self.arch).to(self.device)
        state_dict = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=True
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self._mean = IMAGENET_MEAN.to(self.device)
        self._std = IMAGENET_STD.to(self.device)

    # --- preprocessing ---
    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        img = image.convert("RGB").resize((IMG_SIZE[1], IMG_SIZE[0]), Image.BILINEAR)
        arr = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
        arr = arr.permute(2, 0, 1).unsqueeze(0).to(self.device)  # (1,3,H,W)
        return (arr - self._mean) / self._std

    @torch.no_grad()
    def predict(self, image: Image.Image) -> np.ndarray:
        """
        Returns the predicted class-index mask (H, W) at the ORIGINAL image size.
        """
        orig_w, orig_h = image.size
        x = self._preprocess(image)
        logits = self.model(x)  # (1, C, h, w)
        # Upsample logits to the original resolution, then argmax.
        logits = F.interpolate(
            logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False
        )
        return logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # --- output renderings ---
    @staticmethod
    def colorize(mask: np.ndarray) -> Image.Image:
        """Class-index mask (H, W) -> colour RGB PIL image."""
        return Image.fromarray(PALETTE[mask], mode="RGB")

    @staticmethod
    def overlay(
        image: Image.Image, mask: np.ndarray, alpha: float = 0.5
    ) -> Image.Image:
        """Blend the colour mask over the original image."""
        base = image.convert("RGB")
        color = Image.fromarray(PALETTE[mask], mode="RGB").resize(
            base.size, Image.NEAREST
        )
        return Image.blend(base, color, alpha)

    @staticmethod
    def rle_encode(mask: np.ndarray) -> list:
        """Row-major run-length encoding of the mask.

        Returns a list of [class_id, run_length] pairs. Reconstruct with:
            flat = np.concatenate([np.full(n, c) for c, n in runs])
            mask = flat.reshape(height, width)
        Far smaller than the dense grid for segmentation masks (large
        contiguous regions of the same class).
        """
        flat = mask.reshape(-1)
        if flat.size == 0:
            return []
        # run boundaries = positions where the value changes
        change = np.nonzero(np.diff(flat))[0] + 1
        starts = np.concatenate([[0], change])
        ends = np.concatenate([change, [flat.size]])
        return [[int(flat[s]), int(e - s)] for s, e in zip(starts, ends)]

    @staticmethod
    def class_summary(mask: np.ndarray) -> list:
        """Per-class pixel coverage of the prediction, sorted by area desc."""
        total = mask.size
        ids, counts = np.unique(mask, return_counts=True)
        out = []
        for cls, cnt in zip(ids.tolist(), counts.tolist()):
            out.append(
                {
                    "class_id": int(cls),
                    "class_name": CLASS_NAMES[cls],
                    "color": [int(c) for c in PALETTE[cls]],
                    "pixel_count": int(cnt),
                    "percentage": round(100.0 * cnt / total, 2),
                }
            )
        out.sort(key=lambda d: d["pixel_count"], reverse=True)
        return out

    def info(self) -> dict:
        return {
            "architecture": self.arch,
            "checkpoint": os.path.basename(self.checkpoint_path),
            "val_mIoU": self.val_miou,
            "device": self.device,
            "num_classes": NUM_CLASSES,
            "classes": CLASS_NAMES,
            "palette": [[int(c) for c in PALETTE[i]] for i in range(NUM_CLASSES)],
            "input_size": {"height": IMG_SIZE[0], "width": IMG_SIZE[1]},
        }
