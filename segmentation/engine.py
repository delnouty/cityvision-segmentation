"""Shared training engine — one loop drives every architecture."""

import os

import numpy as np
import torch
import torch.optim as optim
import mlflow
import mlflow.pytorch
from tqdm import tqdm

from .config import TrainConfig
from .data import CLASS_NAMES, NUM_CLASSES, create_dataloaders, remap_mask
from .losses import build_criterion
from .metrics import SegmentationMetrics
from .models import build_model


def _select_device() -> str:
    if torch.cuda.is_available():
        print("Device:", torch.cuda.get_device_name(0))
        return "cuda"
    print("Device: CPU (CUDA not available, training will be slow)")
    return "cpu"


def train_one_epoch(model, loader, criterion, optimizer, device):
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


def validate(model, loader, criterion, device):
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


def _print_iou_table(train_iou, val_iou):
    print(f"\n  {'Class':<15s}  {'Train IoU':>9}  {'Val IoU':>9}")
    print(f"  {'-'*15}  {'-'*9}  {'-'*9}")
    for cls_idx, name in enumerate(CLASS_NAMES):
        t, v = train_iou[cls_idx], val_iou[cls_idx]
        t_str = f"{t:.4f}" if not np.isnan(t) else "   n/a "
        v_str = f"{v:.4f}" if not np.isnan(v) else "   n/a "
        bar = "#" * int(v * 20) if not np.isnan(v) else ""
        print(f"  {name:<15s}  {t_str:>9}  {v_str:>9}  {bar}")
    print()


def fit(config: TrainConfig):
    """Run a full training session described by ``config``."""
    device = _select_device()
    print(f"Architecture: {config.label}  ({config.arch})")

    train_loader, val_loader = create_dataloaders(
        config.img_root,
        config.mask_root,
        batch_size=config.batch_size,
        balanced=config.balanced,
        img_size=config.img_size,
    )

    model = build_model(
        config.arch,
        num_classes=NUM_CLASSES,
        pretrained=config.pretrained,
        dropout=config.dropout,
    ).to(device)

    criterion = build_criterion().to(device)
    optimizer = optim.Adam(model.param_groups(config.lr, config.encoder_lr_mult))

    scheduler = None
    if config.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=4
        )

    os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
    os.chdir(config.project_root)
    mlflow.set_tracking_uri(f"sqlite:///{config.project_root}/mlflow.db")
    mlflow.set_experiment(config.experiment)

    with mlflow.start_run(run_name=config.run_name):
        mlflow.log_params(config.to_mlflow_params())

        best_val_miou = 0.0
        epochs_no_improve = 0

        for epoch in range(1, config.epochs + 1):
            print(f"\n=== EPOCH {epoch}/{config.epochs} ===")

            train_loss, train_m = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_m = validate(model, val_loader, criterion, device)

            train_miou = train_m.mean_iou()
            val_miou = val_m.mean_iou()
            val_pix_acc = val_m.pixel_accuracy()
            train_iou = train_m.iou_per_class()
            val_iou = val_m.iou_per_class()

            print(f"Train loss: {train_loss:.4f}  mIoU: {train_miou:.4f}")
            print(
                f"Val   loss: {val_loss:.4f}  mIoU: {val_miou:.4f}  PixAcc: {val_pix_acc:.4f}"
            )
            _print_iou_table(train_iou, val_iou)

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_mIoU": train_miou,
                    "val_loss": val_loss,
                    "val_mIoU": val_miou,
                    "val_pixel_accuracy": val_pix_acc,
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

            if scheduler is not None and not np.isnan(val_miou):
                scheduler.step(val_miou)

            if val_miou > best_val_miou:
                best_val_miou = val_miou
                epochs_no_improve = 0
                torch.save(model.state_dict(), config.checkpoint_path)
                mlflow.log_artifact(config.checkpoint_path)
                print(f"  → New best model saved (val mIoU={val_miou:.4f})")
            else:
                epochs_no_improve += 1
                if config.patience > 0:
                    print(
                        f"  → No improvement for {epochs_no_improve}/{config.patience} epoch(s) "
                        f"(best val mIoU={best_val_miou:.4f})"
                    )
                    if epochs_no_improve >= config.patience:
                        print(
                            f"\nEarly stopping triggered at epoch {epoch} "
                            f"(no improvement for {config.patience} epochs)."
                        )
                        break

        print(f"\nTraining complete. Best val mIoU: {best_val_miou:.4f}")
        mlflow.log_metric("best_val_mIoU", best_val_miou)

        # Register this run's best model as a new version in the MLflow Model
        # Registry. Reload the best checkpoint first so the registered model is
        # the best epoch, not the final one.
        if os.path.exists(config.checkpoint_path):
            model.load_state_dict(
                torch.load(config.checkpoint_path, map_location=device)
            )
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name=config.registered_model_name,
        )
        print(
            f"Registered model '{config.registered_model_name}' "
            f"(MLflow Model Registry, new version)."
        )

    return best_val_miou
