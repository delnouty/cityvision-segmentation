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

# ============================================================
# 1. REMAP TO 8 OBJECTS
# ============================================================
print("FILE EXECUTED")

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


def remap_mask(mask):
    """mask: tensor HxW (Cityscapes labelIds) → remapped to 0..8"""
    new_mask = torch.zeros_like(mask)
    for src, dst in TARGET_CLASSES.items():
        new_mask[mask == src] = dst
    return new_mask


# ============================================================
# 2. U-NET
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()

        self.down1 = DoubleConv(3, 32)            # shallow — no dropout
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(32, 64)           # shallow — no dropout
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(64, 128, dropout=dropout)
        self.pool3 = nn.MaxPool2d(2)

        self.down4 = DoubleConv(128, 256, dropout=dropout)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(256, 512, dropout=dropout)

        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv4 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv1 = DoubleConv(64, 32)

        self.out = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        c1 = self.down1(x)
        p1 = self.pool1(c1)

        c2 = self.down2(p1)
        p2 = self.pool2(c2)

        c3 = self.down3(p2)
        p3 = self.pool3(c3)

        c4 = self.down4(p3)
        p4 = self.pool4(c4)

        bn = self.bottleneck(p4)

        u4 = self.up4(bn)
        u4 = torch.cat([u4, c4], dim=1)
        c5 = self.conv4(u4)

        u3 = self.up3(c5)
        u3 = torch.cat([u3, c3], dim=1)
        c6 = self.conv3(u3)

        u2 = self.up2(c6)
        u2 = torch.cat([u2, c2], dim=1)
        c7 = self.conv2(u2)

        u1 = self.up1(c7)
        u1 = torch.cat([u1, c1], dim=1)
        c8 = self.conv1(u1)

        return self.out(c8)


# ============================================================
# 3. LOSS  —  CE + Dice
# ============================================================

class_weights = torch.tensor([
    0.5,   # background
    1.0,   # road
    1.0,   # building
    1.0,   # vegetation
    1.0,   # sky
    2.5,   # person
    1.0,   # car
    3.0,   # traffic_sign
    3.0,   # bicycle
], dtype=torch.float32)


class DiceLoss(nn.Module):
    """Weighted Dice: per-class scores are averaged using class_weights,
    so rare classes (motorcycle, bus) pull the loss up more than background."""
    def __init__(self, class_weights, smooth=1.0):
        super().__init__()
        # Normalise so weights sum to num_classes — keeps Dice on unit scale.
        w = class_weights / class_weights.sum() * len(class_weights)
        self.register_buffer("weights", w)
        self.smooth = smooth

    def forward(self, logits, targets):
        probs   = torch.softmax(logits, dim=1)
        one_hot = torch.zeros_like(probs).scatter_(1, targets.unsqueeze(1), 1.0)
        dims  = (0, 2, 3)
        inter = (probs * one_hot).sum(dim=dims)        # (C,)
        denom = (probs + one_hot).sum(dim=dims)        # (C,)
        dice  = (2.0 * inter + self.smooth) / (denom + self.smooth)  # (C,)
        return 1.0 - (dice * self.weights).sum() / self.weights.sum()


class CombinedLoss(nn.Module):
    """CE(weighted) + Dice(weighted) — both terms are class-sensitive."""
    def __init__(self, class_weights):
        super().__init__()
        self.ce   = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = DiceLoss(class_weights)

    def forward(self, logits, targets):
        return self.ce(logits, targets) + self.dice(logits, targets)


criterion = CombinedLoss(class_weights)


# ============================================================
# 4. METRICS
# ============================================================

class SegmentationMetrics:
    """Accumulates confusion matrix across batches, computes IoU and pixel accuracy."""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """preds: BxHxW (argmax), targets: BxHxW"""
        preds = preds.cpu().numpy().ravel()
        targets = targets.cpu().numpy().ravel()
        mask = (targets >= 0) & (targets < self.num_classes)
        combined = self.num_classes * targets[mask].astype(np.int64) + preds[mask].astype(np.int64)
        self.confusion += np.bincount(combined, minlength=self.num_classes ** 2).reshape(
            self.num_classes, self.num_classes
        )

    def pixel_accuracy(self):
        correct = np.diag(self.confusion).sum()
        total = self.confusion.sum()
        return float(correct) / float(total) if total > 0 else 0.0

    def iou_per_class(self):
        tp = np.diag(self.confusion)
        fp = self.confusion.sum(axis=0) - tp
        fn = self.confusion.sum(axis=1) - tp
        denom = tp + fp + fn
        iou = np.where(denom > 0, tp / denom, np.nan)
        return iou  # shape (num_classes,)

    def mean_iou(self):
        iou = self.iou_per_class()
        return float(np.nanmean(iou))

    def reset(self):
        self.confusion[:] = 0


# ============================================================
# 5. TRAIN / VALIDATE
# ============================================================

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
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
        preds = logits.argmax(dim=1)
        metrics.update(preds, masks)

    return total_loss / len(loader), metrics


def validate(model, loader, device):
    model.eval()
    total_loss = 0
    metrics = SegmentationMetrics(NUM_CLASSES)

    if len(loader) == 0:
        return float("nan"), metrics

    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Val"):
            imgs = imgs.to(device)
            masks = remap_mask(masks).to(device)

            logits = model(imgs)
            loss = criterion(logits, masks)
            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            metrics.update(preds, masks)

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
    img_root = os.path.join(project_root, "data/cityscapes/P8_Cityscapes_leftImg8bit_trainvaltest/leftImg8bit")
    mask_root = os.path.join(project_root, "data/cityscapes/P8_Cityscapes_gtFine_trainvaltest/gtFine")
    checkpoint_path = os.path.join(project_root, "backend", "model", "unet_best.pth")
    batch_size = 8
    lr = 1e-4
    epochs = 60

    train_loader, val_loader = create_dataloaders(img_root, mask_root, batch_size=batch_size, balanced=True)

    dropout = 0.3
    model = UNet(dropout=dropout).to(device)
    criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(project_root)
    mlflow.set_tracking_uri(f"sqlite:///{project_root}/mlflow.db")
    mlflow.set_experiment("urban-segmentation")

    with mlflow.start_run(run_name="model_train_unet"):
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "img_size": "256x512",
            "num_classes": NUM_CLASSES,
            "optimizer": "Adam",
            "architecture": "UNet",
            "loss": "CE+Dice(weighted)",
            "sampler": "WeightedRandom",
            "dropout": dropout,
        })

        best_val_miou = 0.0

        for epoch in range(1, epochs + 1):
            print(f"\n=== EPOCH {epoch}/{epochs} ===")

            train_loss, train_m = train_one_epoch(model, train_loader, optimizer, device)
            val_loss, val_m = validate(model, val_loader, device)

            train_miou = train_m.mean_iou()
            val_miou = val_m.mean_iou()
            val_pix_acc = val_m.pixel_accuracy()
            val_iou = val_m.iou_per_class()
            train_iou = train_m.iou_per_class()

            print(f"Train loss: {train_loss:.4f}  mIoU: {train_miou:.4f}")
            print(f"Val   loss: {val_loss:.4f}  mIoU: {val_miou:.4f}  PixAcc: {val_pix_acc:.4f}")
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

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_mIoU": train_miou,
                "val_loss": val_loss,
                "val_mIoU": val_miou,
                "val_pixel_accuracy": val_pix_acc,
            }, step=epoch)

            for cls_idx, name in enumerate(CLASS_NAMES):
                v_iou = val_iou[cls_idx]
                t_iou = train_iou[cls_idx]
                if not np.isnan(v_iou):
                    mlflow.log_metric(f"val_iou_{name}", float(v_iou), step=epoch)
                if not np.isnan(t_iou):
                    mlflow.log_metric(f"train_iou_{name}", float(t_iou), step=epoch)

            # Save best model by val mIoU
            if val_miou > best_val_miou:
                best_val_miou = val_miou
                torch.save(model.state_dict(), checkpoint_path)
                mlflow.log_artifact(checkpoint_path)
                print(f"  → New best model saved (val mIoU={val_miou:.4f})")

        print(f"\nTraining complete. Best val mIoU: {best_val_miou:.4f}")
        mlflow.log_metric("best_val_mIoU", best_val_miou)


if __name__ == "__main__":
    main()
