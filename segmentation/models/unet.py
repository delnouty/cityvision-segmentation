"""From-scratch U-Net (no pretrained backbone)."""

import torch
import torch.nn as nn

from ..data import NUM_CLASSES
from .base import BaseSegModel


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


class UNet(BaseSegModel):
    """Classic encoder-decoder U-Net with skip connections, trained from scratch."""

    # No pretrained encoder → single learning rate for the whole network.
    encoder_modules = ()

    def __init__(self, num_classes=NUM_CLASSES, dropout=0.3, **_unused):
        super().__init__()

        self.down1 = DoubleConv(3, 32)  # shallow — no dropout
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(32, 64)  # shallow — no dropout
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

        u4 = self.conv4(torch.cat([self.up4(bn), c4], dim=1))
        u3 = self.conv3(torch.cat([self.up3(u4), c3], dim=1))
        u2 = self.conv2(torch.cat([self.up2(u3), c2], dim=1))
        u1 = self.conv1(torch.cat([self.up1(u2), c1], dim=1))

        return self.out(u1)
