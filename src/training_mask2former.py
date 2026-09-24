"""
Mask2Former (Swin-Tiny) fine-tuned on the 9-class Cityscapes scheme — PoC.

Starts from facebook/mask2former-swin-tiny-ade-semantic (ADE20K, never saw
Cityscapes) and swaps its 150-class head for our 9 classes. Data, remapping
and metric are the ones used by the baseline (ResNet50-UNet), so val mIoU is
directly comparable.

Model + loss come from Hugging Face Transformers
(Mask2FormerForUniversalSegmentation, Cheng et al., CVPR 2022, arXiv:2112.01527).

Usage:
    python src/training_mask2former.py                     # full training
    python src/training_mask2former.py --epochs 1 --limit 20   # quick run
"""

import argparse
import math
import os
import sys
from itertools import islice

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import Mask2FormerForUniversalSegmentation

sys.path.insert(0, os.path.dirname(__file__))
from dataloader import create_dataloaders  # noqa: E402
from training_resnet50 import SegmentationMetrics  # noqa: E402  same metric as baseline

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cityvision.constants import CLASS_NAMES, NUM_CLASSES  # noqa: E402
from cityvision.models import remap_mask  # noqa: E402

PRETRAINED = "facebook/mask2former-swin-tiny-ade-semantic"


# ============================================================
# 1. MODEL
# ============================================================


def build_model(pretrained=PRETRAINED):
    """Pretrained Mask2Former with a fresh 9-class classification head."""
    return Mask2FormerForUniversalSegmentation.from_pretrained(
        pretrained,
        num_labels=NUM_CLASSES,
        id2label=dict(enumerate(CLASS_NAMES)),
        label2id={n: i for i, n in enumerate(CLASS_NAMES)},
        ignore_mismatched_sizes=True,  # class_predictor 151 -> 10 outputs
    )


def backbone_params(model):
    return list(model.model.pixel_level_module.encoder.parameters())


# ============================================================
# 2. TARGETS  —  semantic mask -> (binary masks, class ids)
# ============================================================


def to_mask2former_targets(masks: torch.Tensor):
    """
    Mask2Former is trained on a *set* of masks, not a per-pixel label map:
    for each image, one binary mask per class present + its class id.

    masks: (B, H, W) long, already remapped to 0..NUM_CLASSES-1.
    """
    mask_labels, class_labels = [], []
    for m in masks:
        classes = torch.unique(m)
        mask_labels.append((m[None] == classes[:, None, None]).float())
        class_labels.append(classes)
    return mask_labels, class_labels


# ============================================================
# 3. PREDICTION  —  (queries) -> per-pixel label map
# ============================================================


def semantic_logits(outputs, size):
    """
    Same computation as Mask2FormerImageProcessor.post_process_semantic_segmentation,
    kept batched and on GPU: sum over queries of class prob x mask prob.
    Returns (B, NUM_CLASSES, H, W) scores; argmax gives the label map.
    """
    # Last class is Mask2Former's "no object"; drop it
    class_probs = outputs.class_queries_logits.float().softmax(-1)[..., :-1]
    mask_probs = outputs.masks_queries_logits.float().sigmoid()
    mask_probs = F.interpolate(
        mask_probs, size=size, mode="bilinear", align_corners=False
    )
    return torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)


# ============================================================
# 4. TRAIN / VALIDATE
# ============================================================


# bf16 autocast: halves activation memory (8 GB GPU) without the loss-scaling
# that fp16 needs. Falls back to fp32 on CPU.
def _autocast(device):
    return torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"
    )


def poly_with_warmup(total_steps, warmup_steps, power=0.9):
    """
    LR multiplier per optimizer step: linear warm-up from 0.1% to 100%, then
    poly decay to 0 (the Mask2Former paper's schedule, power 0.9).
    """

    def factor(step):
        if step < warmup_steps:
            return 0.001 + (1 - 0.001) * step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1 - progress) ** power

    return factor


def train_one_epoch(
    model, loader, optimizer, scheduler, device, accum_steps=1, clip=0.01, on_step=None
):
    """
    accum_steps: gradients of that many batches are summed before each
                 optimizer step (effective batch = batch_size x accum_steps).
    on_step:     optional callable run after each optimizer step (EoMT uses it
                 for mask annealing).
    """
    model.train()
    total_loss = 0.0
    metrics = SegmentationMetrics(NUM_CLASSES)

    optimizer.zero_grad()
    for i, (imgs, masks) in enumerate(tqdm(loader, desc="Train"), start=1):
        imgs = imgs.to(device)
        masks = remap_mask(masks).to(device)
        mask_labels, class_labels = to_mask2former_targets(masks)

        with _autocast(device):
            outputs = model(
                pixel_values=imgs, mask_labels=mask_labels, class_labels=class_labels
            )
        (outputs.loss / accum_steps).backward()

        if i % accum_steps == 0 or i == len(loader):
            # Mask2Former paper: full-model gradient clipping at 0.01
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if on_step is not None:
                on_step()

        total_loss += outputs.loss.item()
        with torch.no_grad():
            preds = semantic_logits(outputs, masks.shape[-2:]).argmax(dim=1)
        metrics.update(preds, masks)

    return total_loss / len(loader), metrics


def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    metrics = SegmentationMetrics(NUM_CLASSES)

    if len(loader) == 0:
        return float("nan"), metrics

    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Val"):
            imgs = imgs.to(device)
            masks = remap_mask(masks).to(device)
            mask_labels, class_labels = to_mask2former_targets(masks)

            with _autocast(device):
                outputs = model(
                    pixel_values=imgs,
                    mask_labels=mask_labels,
                    class_labels=class_labels,
                )
            total_loss += outputs.loss.item()
            preds = semantic_logits(outputs, masks.shape[-2:]).argmax(dim=1)
            metrics.update(preds, masks)

    return total_loss / len(loader), metrics


# ============================================================
# 5. MAIN
# ============================================================


class _Limited:
    """First n batches of a loader (for --limit quick runs)."""

    def __init__(self, loader, n):
        self.loader, self.n = loader, min(n, len(loader))

    def __iter__(self):
        return islice(iter(self.loader), self.n)

    def __len__(self):
        return self.n


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument(
        "--accum-steps",
        type=int,
        default=2,
        help="gradient accumulation; default 2 x batch 2 = effective batch 4 (as baseline)",
    )
    p.add_argument(
        "--warmup-steps",
        type=int,
        default=250,
        help="linear LR warm-up, in optimizer steps (~1/3 epoch at the defaults)",
    )
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="use only the first N train/val batches (quick runs); 0 = all",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if torch.cuda.is_available():
        device = "cuda"
        print("Device:", torch.cuda.get_device_name(0))
    else:
        device = "cpu"
        print("Device: CPU (CUDA not available, training will be slow)")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    img_root = os.path.join(
        project_root,
        "data/cityscapes/P8_Cityscapes_leftImg8bit_trainvaltest/leftImg8bit",
    )
    mask_root = os.path.join(
        project_root, "data/cityscapes/P8_Cityscapes_gtFine_trainvaltest/gtFine"
    )
    # Kept out of backend/model/ so the served model is never touched.
    checkpoint_path = os.path.join(
        project_root, "model_backups", "mask2former_swint_best.pth"
    )

    train_loader, val_loader = create_dataloaders(
        img_root,
        mask_root,
        batch_size=args.batch_size,
        balanced=True,
        num_workers=args.num_workers,
    )
    if args.limit:
        train_loader = _Limited(train_loader, args.limit)
        val_loader = _Limited(val_loader, args.limit)

    model = build_model().to(device)

    # Lower LR for the pretrained Swin backbone, full LR for decoders + head
    enc = backbone_params(model)
    enc_ids = {id(p) for p in enc}
    rest = [p for p in model.parameters() if id(p) not in enc_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": enc, "lr": args.lr * 0.1},
            {"params": rest, "lr": args.lr},
        ],
        weight_decay=0.05,
    )
    # Poly decay is planned over the full --epochs; early stopping just cuts it short.
    steps_per_epoch = math.ceil(len(train_loader) / args.accum_steps)
    total_steps = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, poly_with_warmup(total_steps, args.warmup_steps)
    )

    os.chdir(project_root)
    mlflow.set_tracking_uri(f"sqlite:///{project_root}/mlflow.db")
    mlflow.set_experiment("urban-segmentation")

    with mlflow.start_run(run_name="Mask2Former-SwinT"):
        mlflow.log_params(
            {
                "epochs": args.epochs,
                "patience": args.patience,
                "batch_size": args.batch_size,
                "accum_steps": args.accum_steps,
                "effective_batch_size": args.batch_size * args.accum_steps,
                "lr_encoder": args.lr * 0.1,
                "lr_decoder": args.lr,
                "lr_schedule": f"linear warm-up {args.warmup_steps} steps + poly(0.9)",
                "total_optimizer_steps": total_steps,
                "img_size": "512x1024",
                "num_classes": NUM_CLASSES,
                "optimizer": "AdamW(wd=0.05)",
                "architecture": "Mask2Former-SwinT",
                "pretrained": PRETRAINED,
                "loss": "Mask2Former (CE + mask BCE + Dice, Hungarian matching)",
                "sampler": "WeightedRandom",
                "precision": "bf16" if device == "cuda" else "fp32",
                "limit_batches": args.limit,
            }
        )

        best_val_miou = 0.0
        epochs_no_improve = 0

        for epoch in range(1, args.epochs + 1):
            print(f"\n=== EPOCH {epoch}/{args.epochs} ===")

            train_loss, train_m = train_one_epoch(
                model, train_loader, optimizer, scheduler, device, args.accum_steps
            )
            val_loss, val_m = validate(model, val_loader, device)

            train_miou = train_m.mean_iou()
            val_miou = val_m.mean_iou()
            val_pix_acc = val_m.pixel_accuracy()
            val_iou = val_m.iou_per_class()
            train_iou = train_m.iou_per_class()

            print(f"Train loss: {train_loss:.4f}  mIoU: {train_miou:.4f}")
            print(
                f"Val   loss: {val_loss:.4f}  mIoU: {val_miou:.4f}  PixAcc: {val_pix_acc:.4f}"
            )
            print(f"\n  {'Class':<15s}  {'Train IoU':>9}  {'Val IoU':>9}")
            print(f"  {'-'*15}  {'-'*9}  {'-'*9}")
            for cls_idx, name in enumerate(CLASS_NAMES):
                t = train_iou[cls_idx]
                v = val_iou[cls_idx]
                t_str = f"{t:.4f}" if not np.isnan(t) else "   n/a "
                v_str = f"{v:.4f}" if not np.isnan(v) else "   n/a "
                bar = "#" * int(v * 20) if not np.isnan(v) else ""
                print(f"  {name:<15s}  {t_str:>9}  {v_str:>9}  {bar}")
            print()

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_mIoU": train_miou,
                    "val_loss": val_loss,
                    "val_mIoU": val_miou,
                    "val_pixel_accuracy": val_pix_acc,
                    "lr_decoder": optimizer.param_groups[1]["lr"],
                },
                step=epoch,
            )
            for cls_idx, name in enumerate(CLASS_NAMES):
                if not np.isnan(val_iou[cls_idx]):
                    mlflow.log_metric(
                        f"val_iou_{name}", float(val_iou[cls_idx]), step=epoch
                    )
                if not np.isnan(train_iou[cls_idx]):
                    mlflow.log_metric(
                        f"train_iou_{name}", float(train_iou[cls_idx]), step=epoch
                    )

            if val_miou > best_val_miou:
                best_val_miou = val_miou
                epochs_no_improve = 0
                torch.save(model.state_dict(), checkpoint_path)
                mlflow.log_artifact(checkpoint_path)
                print(f"  → New best model saved (val mIoU={val_miou:.4f})")
            else:
                epochs_no_improve += 1
                print(
                    f"  → No improvement for {epochs_no_improve}/{args.patience} epoch(s) "
                    f"(best val mIoU={best_val_miou:.4f})"
                )
                if epochs_no_improve >= args.patience:
                    print(f"\nEarly stopping triggered at epoch {epoch}.")
                    break

        print(f"\nTraining complete. Best val mIoU: {best_val_miou:.4f}")
        mlflow.log_metric("best_val_mIoU", best_val_miou)


if __name__ == "__main__":
    main()
