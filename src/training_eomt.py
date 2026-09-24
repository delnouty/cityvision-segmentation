"""
EoMT (Encoder-only Mask Transformer, ViT-B / DINOv2) fine-tuned on the 9-class
Cityscapes scheme — PoC, step 2 (after Mask2Former).

Starts from tue-mps/coco_panoptic_eomt_base_640_2x (COCO panoptic, never saw
Cityscapes) and swaps its 133-class head for our 9 classes. Data, targets,
post-processing and metric are shared with training_mask2former.py, so val
mIoU is directly comparable with ResNet50-UNet and Mask2Former.

Model + loss: Hugging Face Transformers (EomtForUniversalSegmentation,
Kerssies et al., CVPR 2025, arXiv:2503.19108). Training recipe (layer-wise LR
decay, two-stage warm-up, mask annealing) follows the official code,
github.com/tue-mps/eomt (MIT licence), rescaled to our number of steps.

Usage:
    python src/training_eomt.py                          # full training
    python src/training_eomt.py --epochs 1 --limit 20    # quick run
"""

import argparse
import math
import os
import sys

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import EomtForUniversalSegmentation

sys.path.insert(0, os.path.dirname(__file__))
from dataloader import create_dataloaders  # noqa: E402
from training_mask2former import _Limited, train_one_epoch, validate  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cityvision.constants import CLASS_NAMES, NUM_CLASSES  # noqa: E402

PRETRAINED = "tue-mps/coco_panoptic_eomt_base_640_2x"
IMG_SIZE = (512, 1024)


# ============================================================
# 1. MODEL
# ============================================================


def _resize_position_embeddings(model, grid):
    """
    The checkpoint's position embeddings are a square 40x40 grid (640x640
    input). Bicubic-interpolate them to our patch grid (32x64 for 512x1024),
    the standard way of changing a ViT's input resolution.
    """
    emb = model.embeddings
    old = emb.position_embeddings.weight.data
    n, dim = old.shape
    side = int(math.sqrt(n))
    old = old.reshape(1, side, side, dim).permute(0, 3, 1, 2)
    new = F.interpolate(old.float(), size=grid, mode="bicubic", align_corners=False)
    new = new.permute(0, 2, 3, 1).reshape(-1, dim)

    emb.position_embeddings = nn.Embedding(new.shape[0], dim)
    emb.position_embeddings.weight.data.copy_(new)
    emb.register_buffer(
        "position_ids", torch.arange(new.shape[0]).unsqueeze(0), persistent=False
    )
    # Used to reshape patch tokens back into a feature map
    model.grid_size = grid


def build_model(pretrained=PRETRAINED):
    """Pretrained EoMT with a fresh 9-class head, adapted to 512x1024 input."""
    model = EomtForUniversalSegmentation.from_pretrained(
        pretrained,
        num_labels=NUM_CLASSES,
        id2label=dict(enumerate(CLASS_NAMES)),
        label2id={n: i for i, n in enumerate(CLASS_NAMES)},
        ignore_mismatched_sizes=True,  # class_predictor 134 -> 10 outputs
    )
    patch = model.config.patch_size
    _resize_position_embeddings(model, (IMG_SIZE[0] // patch, IMG_SIZE[1] // patch))
    return model


def param_groups(model, lr, llrd):
    """
    Layer-wise LR decay (official EoMT recipe): ViT layer i gets
    lr * llrd**(L-1-i), patch/position embeddings the lowest rate, final
    norm + queries + mask/class heads the full rate.
    Returns (backbone_groups, other_groups).
    """
    n_layers = model.config.num_hidden_layers
    backbone, other = {}, []
    for name, p in model.named_parameters():
        if name.startswith("embeddings."):
            depth = 0
        elif name.startswith("layers."):
            depth = int(name.split(".")[1])
        elif name.startswith("layernorm."):
            depth = n_layers - 1  # llrd**0 -> full lr, as in the official code
        else:
            other.append(p)
            continue
        backbone.setdefault(depth, []).append(p)
    backbone_groups = [
        {"params": ps, "lr": lr * llrd ** (n_layers - 1 - d)}
        for d, ps in sorted(backbone.items())
    ]
    return backbone_groups, [{"params": other, "lr": lr}]


# ============================================================
# 2. SCHEDULES
# ============================================================


def two_stage_warmup_poly(total_steps, head_warmup, vit_warmup, power=0.9):
    """
    Official EoMT LR schedule. Heads warm up linearly for `head_warmup` steps
    while the ViT is frozen (lr 0); then the ViT warms up for `vit_warmup`
    steps; both then decay with poly(power) to 0 at `total_steps`.
    Returns (backbone_factor, head_factor) for LambdaLR.
    """

    def head(step):
        if step < head_warmup:
            return step / head_warmup
        return max(0.0, 1 - (step - head_warmup) / (total_steps - head_warmup)) ** power

    def vit(step):
        if step < head_warmup:
            return 0.0
        if step < head_warmup + vit_warmup:
            return (step - head_warmup) / vit_warmup
        done = step - head_warmup - vit_warmup
        return max(0.0, 1 - done / (total_steps - head_warmup - vit_warmup)) ** power

    return vit, head


class MaskAnnealer:
    """
    EoMT trains with masked attention in its last `num_blocks` ViT blocks,
    then phases it out block by block so inference needs no masks (which is
    where its speed comes from). Probability of keeping the mask in block i
    goes 1 -> 0 with poly(0.9) between its start and end step.

    The official Cityscapes config splits training into num_blocks+2 equal
    parts: block i anneals during part i+1, and the last part is mask-free.
    We keep those proportions.
    """

    def __init__(self, model, total_steps, power=0.9):
        self.model = model
        self.power = power
        n = model.config.num_blocks
        unit = total_steps // (n + 2)
        self.starts = [(i + 1) * unit for i in range(n)]
        self.ends = [(i + 2) * unit for i in range(n)]
        self.step_count = 0
        self._apply()

    def _prob(self, start, end):
        if self.step_count < start:
            return 1.0
        if self.step_count >= end:
            return 0.0
        return (1 - (self.step_count - start) / (end - start)) ** self.power

    def _apply(self):
        for i, (s, e) in enumerate(zip(self.starts, self.ends)):
            self.model.attn_mask_probs[i] = self._prob(s, e)

    def step(self):
        self.step_count += 1
        self._apply()

    @property
    def done(self):
        return self.step_count >= self.ends[-1]


def validate_mask_free(model, loader, device):
    """
    Validate as the model will be deployed: masked attention off in every
    block, whatever the annealing stage. Restores the training probabilities.
    """
    saved = model.attn_mask_probs.clone()
    model.attn_mask_probs.zero_()
    try:
        return validate(model, loader, device)
    finally:
        model.attn_mask_probs.copy_(saved)


# ============================================================
# 3. MAIN
# ============================================================


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--llrd", type=float, default=0.8, help="layer-wise LR decay")
    p.add_argument(
        "--accum-steps",
        type=int,
        default=2,
        help="gradient accumulation; default 2 x batch 2 = effective batch 4 (as baseline)",
    )
    # Official: 500 + 1000 of ~19.9k steps on Cityscapes; same fractions of our 14,880
    p.add_argument("--head-warmup", type=int, default=375)
    p.add_argument("--vit-warmup", type=int, default=750)
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
    checkpoint_path = os.path.join(project_root, "model_backups", "eomt_base_best.pth")

    train_loader, val_loader = create_dataloaders(
        img_root,
        mask_root,
        batch_size=args.batch_size,
        balanced=True,
        img_size=IMG_SIZE,
        num_workers=args.num_workers,
    )
    if args.limit:
        train_loader = _Limited(train_loader, args.limit)
        val_loader = _Limited(val_loader, args.limit)

    model = build_model().to(device)

    steps_per_epoch = math.ceil(len(train_loader) / args.accum_steps)
    total_steps = steps_per_epoch * args.epochs

    backbone_groups, other_groups = param_groups(model, args.lr, args.llrd)
    optimizer = torch.optim.AdamW(backbone_groups + other_groups, weight_decay=0.05)
    vit_factor, head_factor = two_stage_warmup_poly(
        total_steps, args.head_warmup, args.vit_warmup
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        [vit_factor] * len(backbone_groups) + [head_factor] * len(other_groups),
    )
    annealer = MaskAnnealer(model, total_steps)
    print(f"Mask annealing: starts {annealer.starts}, ends {annealer.ends} (steps)")

    os.chdir(project_root)
    mlflow.set_tracking_uri(f"sqlite:///{project_root}/mlflow.db")
    mlflow.set_experiment("urban-segmentation")

    with mlflow.start_run(run_name="EoMT-B"):
        mlflow.log_params(
            {
                "epochs": args.epochs,
                "patience": args.patience,
                "batch_size": args.batch_size,
                "accum_steps": args.accum_steps,
                "effective_batch_size": args.batch_size * args.accum_steps,
                "lr": args.lr,
                "llrd": args.llrd,
                "lr_schedule": (
                    f"two-stage warm-up (heads {args.head_warmup}, "
                    f"ViT {args.vit_warmup} steps) + poly(0.9)"
                ),
                "total_optimizer_steps": total_steps,
                "mask_annealing": f"starts {annealer.starts}, ends {annealer.ends}",
                "img_size": f"{IMG_SIZE[0]}x{IMG_SIZE[1]}",
                "num_classes": NUM_CLASSES,
                "optimizer": "AdamW(wd=0.05)",
                "architecture": "EoMT-B",
                "pretrained": PRETRAINED,
                "loss": "EoMT (CE + mask BCE + Dice, Hungarian matching)",
                "sampler": "WeightedRandom",
                "precision": "bf16" if device == "cuda" else "fp32",
                "val_mode": "mask-free (as deployed)",
                "limit_batches": args.limit,
            }
        )

        best_val_miou = 0.0
        epochs_no_improve = 0

        for epoch in range(1, args.epochs + 1):
            print(f"\n=== EPOCH {epoch}/{args.epochs} ===")

            train_loss, train_m = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                device,
                args.accum_steps,
                on_step=annealer.step,
            )
            val_loss, val_m = validate_mask_free(model, val_loader, device)

            train_miou = train_m.mean_iou()
            val_miou = val_m.mean_iou()
            val_pix_acc = val_m.pixel_accuracy()
            val_iou = val_m.iou_per_class()
            train_iou = train_m.iou_per_class()
            probs = model.attn_mask_probs.tolist()

            print(f"Train loss: {train_loss:.4f}  mIoU: {train_miou:.4f}")
            print(
                f"Val   loss: {val_loss:.4f}  mIoU: {val_miou:.4f}  PixAcc: {val_pix_acc:.4f}"
            )
            print("Attention-mask probs per block:", [round(p, 3) for p in probs])
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
                    "lr_heads": optimizer.param_groups[-1]["lr"],
                    **{f"attn_mask_prob_{i}": p for i, p in enumerate(probs)},
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
            elif not annealer.done:
                # While masks are being phased out, mask-free val can dip
                # temporarily; don't let that trigger early stopping.
                print(
                    f"  → No improvement (best val mIoU={best_val_miou:.4f}); "
                    "early stopping paused until mask annealing ends"
                )
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
