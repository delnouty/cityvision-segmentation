import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import mlflow
import mlflow.pytorch

sys.path.insert(0, os.path.dirname(__file__))
from dataloader import create_dataloaders

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cityvision.constants import CLASS_NAMES, NUM_CLASSES  # noqa: E402,F401
from cityvision.models import SegNet, remap_mask  # noqa: E402

# ============================================================
# 3. LOSS  —  CE + Dice
# ============================================================

class_weights = torch.tensor(
    [
        0.5,  # background
        1.0,  # road
        1.0,  # building
        1.0,  # vegetation
        1.0,  # sky
        2.5,  # person
        1.0,  # car
        3.0,  # traffic_sign
        3.0,  # bicycle
    ],
    dtype=torch.float32,
)


class DiceLoss(nn.Module):
    """Weighted Dice: per-class scores are averaged using class_weights,
    so rare classes (motorcycle, bus) pull the loss up more than background."""

    def __init__(self, class_weights, smooth=1.0):
        super().__init__()
        w = class_weights / class_weights.sum() * len(class_weights)
        self.register_buffer("weights", w)
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        one_hot = torch.zeros_like(probs).scatter_(1, targets.unsqueeze(1), 1.0)
        dims = (0, 2, 3)
        inter = (probs * one_hot).sum(dim=dims)
        denom = (probs + one_hot).sum(dim=dims)
        dice = (2.0 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - (dice * self.weights).sum() / self.weights.sum()


class CombinedLoss(nn.Module):
    """CE(weighted) + Dice(weighted) — both terms are class-sensitive."""

    def __init__(self, class_weights):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = DiceLoss(class_weights)

    def forward(self, logits, targets):
        return self.ce(logits, targets) + self.dice(logits, targets)


criterion = CombinedLoss(class_weights)


# ============================================================
# 4. METRICS  (identical to other scripts)
# ============================================================


class SegmentationMetrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        preds = preds.cpu().numpy().ravel()
        targets = targets.cpu().numpy().ravel()
        mask = (targets >= 0) & (targets < self.num_classes)
        combined = self.num_classes * targets[mask].astype(np.int64) + preds[
            mask
        ].astype(np.int64)
        self.confusion += np.bincount(
            combined, minlength=self.num_classes**2
        ).reshape(self.num_classes, self.num_classes)

    def pixel_accuracy(self):
        correct = np.diag(self.confusion).sum()
        total = self.confusion.sum()
        return float(correct) / float(total) if total > 0 else 0.0

    def iou_per_class(self):
        tp = np.diag(self.confusion)
        fp = self.confusion.sum(axis=0) - tp
        fn = self.confusion.sum(axis=1) - tp
        denom = tp + fp + fn
        return np.where(denom > 0, tp / denom, np.nan)

    def mean_iou(self):
        return float(np.nanmean(self.iou_per_class()))

    def reset(self):
        self.confusion[:] = 0


# ============================================================
# 5. TRAIN / VALIDATE
# ============================================================


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    metrics = SegmentationMetrics(NUM_CLASSES)

    for imgs, masks in tqdm(loader, desc="Train"):
        imgs = imgs.to(device)
        masks = remap_mask(masks).to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        metrics.update(logits.argmax(dim=1), masks)

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

            logits = model(imgs)
            total_loss += criterion(logits, masks).item()
            metrics.update(logits.argmax(dim=1), masks)

    return total_loss / len(loader), metrics


# ============================================================
# 6. MAIN
# ============================================================


def main():
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
    checkpoint_path = os.path.join(project_root, "backend", "model", "segnet_best.pth")

    batch_size = 4
    lr = 1e-3  # higher LR — no pretrained weights, Adam + BN converges fast
    epochs = 20
    patience = 7  # stop if val mIoU does not improve for this many epochs
    # (> LR-scheduler patience so LR is reduced before stopping)

    train_loader, val_loader = create_dataloaders(
        img_root, mask_root, batch_size=batch_size, balanced=True
    )

    model = SegNet(num_classes=NUM_CLASSES).to(device)
    criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Reduce LR by ×0.5 if val mIoU stops improving for 4 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4
    )

    os.chdir(project_root)
    mlflow.set_tracking_uri(f"sqlite:///{project_root}/mlflow.db")
    mlflow.set_experiment("urban-segmentation")

    with mlflow.start_run(run_name="SegNet"):
        mlflow.log_params(
            {
                "epochs": epochs,
                "patience": patience,
                "batch_size": batch_size,
                "lr": lr,
                "img_size": "512x1024",
                "num_classes": NUM_CLASSES,
                "optimizer": "Adam",
                "scheduler": "ReduceLROnPlateau(patience=4)",
                "architecture": "SegNet",
                "pretrained": False,
                "loss": "CE+Dice(alpha=0.5)",
                "sampler": "WeightedRandom",
            }
        )

        best_val_miou = 0.0
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            print(f"\n=== EPOCH {epoch}/{epochs} ===")

            train_loss, train_m = train_one_epoch(
                model, train_loader, optimizer, device
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

            scheduler.step(val_miou)
            current_lr = optimizer.param_groups[0]["lr"]

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_mIoU": train_miou,
                    "val_loss": val_loss,
                    "val_mIoU": val_miou,
                    "val_pixel_accuracy": val_pix_acc,
                    "learning_rate": current_lr,
                },
                step=epoch,
            )

            for cls_idx, name in enumerate(CLASS_NAMES):
                v_iou = val_iou[cls_idx]
                t_iou = train_iou[cls_idx]
                if not np.isnan(v_iou):
                    mlflow.log_metric(f"val_iou_{name}", float(v_iou), step=epoch)
                if not np.isnan(t_iou):
                    mlflow.log_metric(f"train_iou_{name}", float(t_iou), step=epoch)

            if val_miou > best_val_miou:
                best_val_miou = val_miou
                epochs_no_improve = 0
                torch.save(model.state_dict(), checkpoint_path)
                mlflow.log_artifact(checkpoint_path)
                print(f"  → New best model saved (val mIoU={val_miou:.4f})")
            else:
                epochs_no_improve += 1
                print(
                    f"  → No improvement for {epochs_no_improve}/{patience} epoch(s) "
                    f"(best val mIoU={best_val_miou:.4f})"
                )
                if epochs_no_improve >= patience:
                    print(
                        f"\nEarly stopping triggered at epoch {epoch} "
                        f"(no improvement for {patience} epochs)."
                    )
                    break

        print(f"\nTraining complete. Best val mIoU: {best_val_miou:.4f}")
        mlflow.log_metric("best_val_mIoU", best_val_miou)


if __name__ == "__main__":
    main()
