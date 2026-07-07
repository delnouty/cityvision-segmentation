"""
cityvision.models
=================

Single source of truth for the segmentation architectures used across the
project (training scripts and the serving backend). The module structure of
each class is preserved exactly, so checkpoints trained by the original
``src/training_*.py`` scripts load unchanged.

Imports torch/torchvision — do not import this from torch-free consumers
(use ``cityvision.constants`` for the palette/class names instead).
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models

from cityvision.constants import NUM_CLASSES, TARGET_CLASSES


def remap_mask(mask: torch.Tensor) -> torch.Tensor:
    """Map raw Cityscapes labelIds to the 9-class scheme (torch tensor)."""
    new_mask = torch.zeros_like(mask)
    for src, dst in TARGET_CLASSES.items():
        new_mask[mask == src] = dst
    return new_mask


# ============================================================
# UNet (from scratch)
# ============================================================


class _DoubleConv(nn.Module):
    """Conv-ReLU-Conv-ReLU (+ optional Dropout2d). Used by UNet."""

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

        self.down1 = _DoubleConv(3, 32)  # shallow — no dropout
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = _DoubleConv(32, 64)  # shallow — no dropout
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = _DoubleConv(64, 128, dropout=dropout)
        self.pool3 = nn.MaxPool2d(2)

        self.down4 = _DoubleConv(128, 256, dropout=dropout)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = _DoubleConv(256, 512, dropout=dropout)

        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv4 = _DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv3 = _DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = _DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv1 = _DoubleConv(64, 32)

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
# VGG16-UNet (pretrained VGG16 encoder)
# ============================================================


class _DoubleConvPlain(nn.Module):
    """Conv-ReLU-Conv-ReLU (no BN). Used by VGGUNet."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class VGGUNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()

        weights = tv_models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        vgg = tv_models.vgg16(weights=weights)
        f = list(vgg.features.children())

        self.enc1 = nn.Sequential(*f[0:5])  # 64ch,  stride 2
        self.enc2 = nn.Sequential(*f[5:10])  # 128ch, stride 4
        self.enc3 = nn.Sequential(*f[10:17])  # 256ch, stride 8
        self.enc4 = nn.Sequential(*f[17:24])  # 512ch, stride 16
        self.enc5 = nn.Sequential(*f[24:31])  # 512ch, stride 32

        self.up5 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.dec5 = _DoubleConvPlain(512 + 512, 256)

        self.up4 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.dec4 = _DoubleConvPlain(256 + 256, 128)

        self.up3 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.dec3 = _DoubleConvPlain(128 + 128, 64)

        self.up2 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec2 = _DoubleConvPlain(64 + 64, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.out = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        s1 = self.enc1(x)  # 64,  H/2
        s2 = self.enc2(s1)  # 128, H/4
        s3 = self.enc3(s2)  # 256, H/8
        s4 = self.enc4(s3)  # 512, H/16
        s5 = self.enc5(s4)  # 512, H/32

        x = self.up5(s5)
        x = torch.cat([x, s4], dim=1)
        x = self.dec5(x)

        x = self.up4(x)
        x = torch.cat([x, s3], dim=1)
        x = self.dec4(x)

        x = self.up3(x)
        x = torch.cat([x, s2], dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat([x, s1], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = self.dec1(x)

        return self.out(x)


# ============================================================
# ResNet-UNet (pretrained ResNet34 / ResNet50 encoder)
# ============================================================


class _DoubleConvBN(nn.Module):
    """Conv-BN-ReLU-Conv-BN-ReLU. Used by the ResNet U-Nets."""

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


class ResNetUNet(nn.Module):
    """ResNet34 encoder + U-Net decoder."""

    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()

        weights = tv_models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tv_models.resnet34(weights=weights)

        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu
        )  # 64ch, H/2
        self.pool = backbone.maxpool  # H/4
        self.layer1 = backbone.layer1  # 64ch,  H/4
        self.layer2 = backbone.layer2  # 128ch, H/8
        self.layer3 = backbone.layer3  # 256ch, H/16
        self.layer4 = backbone.layer4  # 512ch, H/32

        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = _DoubleConvBN(256 + 256, 256)  # cat layer3

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = _DoubleConvBN(128 + 128, 128)  # cat layer2

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = _DoubleConvBN(64 + 64, 64)  # cat layer1

        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1 = _DoubleConvBN(64 + 64, 32)  # cat stem (H/2)

        self.up0 = nn.ConvTranspose2d(32, 32, 2, stride=2)  # H/2 -> H
        self.out = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        s0 = self.stem(x)  # 64ch, H/2
        s1 = self.layer1(self.pool(s0))  # 64ch, H/4
        s2 = self.layer2(s1)  # 128ch, H/8
        s3 = self.layer3(s2)  # 256ch, H/16
        s4 = self.layer4(s3)  # 512ch, H/32

        x = self.up4(s4)
        x = self.dec4(torch.cat([x, s3], dim=1))

        x = self.up3(x)
        x = self.dec3(torch.cat([x, s2], dim=1))

        x = self.up2(x)
        x = self.dec2(torch.cat([x, s1], dim=1))

        x = self.up1(x)
        x = self.dec1(torch.cat([x, s0], dim=1))

        x = self.up0(x)
        return self.out(x)


class ResNet50UNet(nn.Module):
    """ResNet50 encoder + U-Net decoder."""

    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()

        weights = tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = tv_models.resnet50(weights=weights)

        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu
        )  # 64ch,  H/2
        self.pool = backbone.maxpool  # H/4
        self.layer1 = backbone.layer1  # 256ch,  H/4
        self.layer2 = backbone.layer2  # 512ch,  H/8
        self.layer3 = backbone.layer3  # 1024ch, H/16
        self.layer4 = backbone.layer4  # 2048ch, H/32

        self.up4 = nn.ConvTranspose2d(2048, 512, 2, stride=2)
        self.dec4 = _DoubleConvBN(512 + 1024, 512)  # cat layer3 (1024ch)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = _DoubleConvBN(256 + 512, 256)  # cat layer2 (512ch)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = _DoubleConvBN(128 + 256, 128)  # cat layer1 (256ch)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = _DoubleConvBN(64 + 64, 32)  # cat stem (64ch, H/2)

        self.up0 = nn.ConvTranspose2d(32, 32, 2, stride=2)  # H/2 -> H
        self.out = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        s0 = self.stem(x)  # 64ch,   H/2
        s1 = self.layer1(self.pool(s0))  # 256ch,  H/4
        s2 = self.layer2(s1)  # 512ch,  H/8
        s3 = self.layer3(s2)  # 1024ch, H/16
        s4 = self.layer4(s3)  # 2048ch, H/32

        x = self.up4(s4)
        x = self.dec4(torch.cat([x, s3], dim=1))

        x = self.up3(x)
        x = self.dec3(torch.cat([x, s2], dim=1))

        x = self.up2(x)
        x = self.dec2(torch.cat([x, s1], dim=1))

        x = self.up1(x)
        x = self.dec1(torch.cat([x, s0], dim=1))

        x = self.up0(x)
        return self.out(x)


# ============================================================
# SegNet (from scratch, max-unpool decoder)
# ============================================================


def _conv_block(in_ch: int, out_ch: int, n_convs: int) -> nn.Sequential:
    """Conv-BN-ReLU repeated n_convs times; first conv changes channel depth."""
    layers = []
    for i in range(n_convs):
        layers += [
            nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
    return nn.Sequential(*layers)


class SegNet(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        self.enc1 = _conv_block(3, 64, 2)  # 3  → 64
        self.enc2 = _conv_block(64, 128, 2)  # 64 → 128
        self.enc3 = _conv_block(128, 256, 3)  # 128→ 256
        self.enc4 = _conv_block(256, 512, 3)  # 256→ 512
        self.enc5 = _conv_block(512, 512, 3)  # 512→ 512

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.dec5 = _conv_block(512, 512, 3)
        self.dec4 = _conv_block(512, 256, 3)
        self.dec3 = _conv_block(256, 128, 3)
        self.dec2 = _conv_block(128, 64, 2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.enc1(x)
        x, idx1 = self.pool(x)  # H/2
        x = self.enc2(x)
        x, idx2 = self.pool(x)  # H/4
        x = self.enc3(x)
        x, idx3 = self.pool(x)  # H/8
        x = self.enc4(x)
        x, idx4 = self.pool(x)  # H/16
        x = self.enc5(x)
        x, idx5 = self.pool(x)  # H/32

        x = self.unpool(x, idx5)
        x = self.dec5(x)  # H/16
        x = self.unpool(x, idx4)
        x = self.dec4(x)  # H/8
        x = self.unpool(x, idx3)
        x = self.dec3(x)  # H/4
        x = self.unpool(x, idx2)
        x = self.dec2(x)  # H/2
        x = self.unpool(x, idx1)
        x = self.dec1(x)  # H (original)

        return x


# ============================================================
# Factory
# ============================================================

_ARCH_TO_CLASS = {
    "UNet": UNet,
    "VGG16-UNet": VGGUNet,
    "SegNet": SegNet,
    "ResNet34-UNet": ResNetUNet,
    "ResNet50-UNet": ResNet50UNet,
}


def build_model(arch: str, num_classes: int = NUM_CLASSES, pretrained: bool = False):
    """Instantiate a model by architecture name.

    `pretrained` only applies to the encoder-pretrained architectures
    (VGG16-UNet, ResNet34/50-UNet); ignored by UNet and SegNet.
    """
    if arch == "UNet":
        return UNet(num_classes=num_classes)
    if arch == "SegNet":
        return SegNet(num_classes=num_classes)
    if arch == "VGG16-UNet":
        return VGGUNet(num_classes=num_classes, pretrained=pretrained)
    if arch == "ResNet34-UNet":
        return ResNetUNet(num_classes=num_classes, pretrained=pretrained)
    if arch == "ResNet50-UNet":
        return ResNet50UNet(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"Unknown architecture: {arch!r}")
