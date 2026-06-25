"""Confusion-matrix based segmentation metrics (IoU + pixel accuracy)."""

import numpy as np
import torch


class SegmentationMetrics:
    """Accumulates a confusion matrix across batches, computes IoU and accuracy."""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """preds: BxHxW (argmax), targets: BxHxW."""
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
        return np.where(denom > 0, tp / denom, np.nan)  # shape (num_classes,)

    def mean_iou(self):
        return float(np.nanmean(self.iou_per_class()))

    def reset(self):
        self.confusion[:] = 0
