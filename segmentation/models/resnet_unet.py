"""ResNet (34 / 50) encoder + U-Net decoder.

ResNet34 uses BasicBlocks; ResNet50 uses Bottleneck blocks whose feature maps
are 4x wider. The decoder channel widths adapt automatically to the encoder
channels reported by the chosen depth, so one class covers both depths.

Feature map sizes (input 512x1024):
    stem  (conv1+bn+relu):  64ch,  H/2  — captured before maxpool
    layer1:                 C1ch,  H/4
    layer2:                 C2ch,  H/8
    layer3:                 C3ch,  H/16
    layer4:                 C4ch,  H/32
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models

from ..data import NUM_CLASSES
from .base import BaseSegModel

# Per-depth: torchvision constructor, default weights, and encoder channel widths
# (layer1..layer4) emitted by that backbone.
_RESNET_SPECS = {
    34: {
        "ctor":     tv_models.resnet34,
        "weights":  lambda: tv_models.ResNet34_Weights.IMAGENET1K_V1,
        "channels": (64, 128, 256, 512),
    },
    50: {
        "ctor":     tv_models.resnet50,
        "weights":  lambda: tv_models.ResNet50_Weights.IMAGENET1K_V2,
        "channels": (256, 512, 1024, 2048),
    },
}


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ResNetUNet(BaseSegModel):
    """ResNet encoder + U-Net decoder with skip connections at every stride."""

    encoder_modules = ("stem", "pool", "layer1", "layer2", "layer3", "layer4")

    def __init__(self, num_classes=NUM_CLASSES, depth=34, pretrained=True, **_unused):
        super().__init__()
        if depth not in _RESNET_SPECS:
            raise ValueError(f"Unsupported ResNet depth {depth}; choose from {sorted(_RESNET_SPECS)}")
        spec = _RESNET_SPECS[depth]

        weights = spec["weights"]() if pretrained else None
        backbone = spec["ctor"](weights=weights)
        c1, c2, c3, c4 = spec["channels"]

        # Encoder — split into stages to capture skip features
        self.stem   = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # 64ch, H/2
        self.pool   = backbone.maxpool                                            # H/4
        self.layer1 = backbone.layer1   # c1, H/4
        self.layer2 = backbone.layer2   # c2, H/8
        self.layer3 = backbone.layer3   # c3, H/16
        self.layer4 = backbone.layer4   # c4, H/32

        # Decoder — widths derived from encoder channels so 34 & 50 both work
        self.up4  = nn.ConvTranspose2d(c4, c3 // 2, 2, stride=2)
        self.dec4 = DoubleConv(c3 // 2 + c3, c3 // 2)

        self.up3  = nn.ConvTranspose2d(c3 // 2, c2 // 2, 2, stride=2)
        self.dec3 = DoubleConv(c2 // 2 + c2, c2 // 2)

        self.up2  = nn.ConvTranspose2d(c2 // 2, c1 // 2, 2, stride=2)
        self.dec2 = DoubleConv(c1 // 2 + c1, 128)

        self.up1  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(64 + 64, 32)       # cat stem (64ch, H/2)

        # Final upsample H/2 → H
        self.up0  = nn.ConvTranspose2d(32, 32, 2, stride=2)
        self.out  = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        s0 = self.stem(x)                # 64ch,  H/2
        s1 = self.layer1(self.pool(s0))  # c1,    H/4
        s2 = self.layer2(s1)             # c2,    H/8
        s3 = self.layer3(s2)             # c3,    H/16
        s4 = self.layer4(s3)             # c4,    H/32

        x = self.dec4(torch.cat([self.up4(s4), s3], dim=1))
        x = self.dec3(torch.cat([self.up3(x),  s2], dim=1))
        x = self.dec2(torch.cat([self.up2(x),  s1], dim=1))
        x = self.dec1(torch.cat([self.up1(x),  s0], dim=1))

        x = self.up0(x)
        return self.out(x)
