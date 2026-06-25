"""CE + Dice loss, both weighted to emphasise rare foreground classes."""

import torch
import torch.nn as nn

# Default per-class weights (background down-weighted, rare objects up-weighted).
DEFAULT_CLASS_WEIGHTS = torch.tensor(
    [
        0.5,  # background
        1.0,  # road
        1.0,  # building
        1.0,  # vegetation
        1.0,  # sky
        2.0,  # person
        1.0,  # car
        2.0,  # traffic_sign
        2.0,  # bicycle
    ],
    dtype=torch.float32,
)


class DiceLoss(nn.Module):
    """Weighted Dice: per-class scores averaged using ``class_weights`` so rare
    classes pull the loss up more than background."""

    def __init__(self, class_weights, smooth=1.0):
        super().__init__()
        # Normalise so weights sum to num_classes — keeps Dice on unit scale.
        w = class_weights / class_weights.sum() * len(class_weights)
        self.register_buffer("weights", w)
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        one_hot = torch.zeros_like(probs).scatter_(1, targets.unsqueeze(1), 1.0)
        dims = (0, 2, 3)
        inter = (probs * one_hot).sum(dim=dims)  # (C,)
        denom = (probs + one_hot).sum(dim=dims)  # (C,)
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


def build_criterion(class_weights=None) -> CombinedLoss:
    """Construct the combined CE+Dice criterion (uses default weights if none)."""
    if class_weights is None:
        class_weights = DEFAULT_CLASS_WEIGHTS.clone()
    return CombinedLoss(class_weights)
