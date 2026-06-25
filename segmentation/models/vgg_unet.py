"""VGG16 encoder (ImageNet-pretrained) + symmetric U-Net decoder."""

import torch
import torch.nn as nn
import torchvision.models as tv_models

from ..data import NUM_CLASSES
from .base import BaseSegModel


class DoubleConv(nn.Module):
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


class VGGUNet(BaseSegModel):
    """VGG16 encoder with a symmetric U-Net decoder and skip connections.

    Encoder blocks (each ends in a MaxPool2d):
        enc1 → 64ch H/2,  enc2 → 128ch H/4,  enc3 → 256ch H/8,
        enc4 → 512ch H/16, enc5 → 512ch H/32.
    """

    encoder_modules = ("enc1", "enc2", "enc3", "enc4", "enc5")

    def __init__(self, num_classes=NUM_CLASSES, pretrained=True, **_unused):
        super().__init__()

        weights = tv_models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        vgg = tv_models.vgg16(weights=weights)
        f = list(vgg.features.children())

        self.enc1 = nn.Sequential(*f[0:5])  # 64ch,  stride 2
        self.enc2 = nn.Sequential(*f[5:10])  # 128ch, stride 4
        self.enc3 = nn.Sequential(*f[10:17])  # 256ch, stride 8
        self.enc4 = nn.Sequential(*f[17:24])  # 512ch, stride 16
        self.enc5 = nn.Sequential(*f[24:31])  # 512ch, stride 32

        # Decoder: upsample → concat skip → DoubleConv
        self.up5 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.dec5 = DoubleConv(512 + 512, 256)

        self.up4 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.dec4 = DoubleConv(256 + 256, 128)

        self.up3 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.dec3 = DoubleConv(128 + 128, 64)

        self.up2 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec2 = DoubleConv(64 + 64, 64)

        # Final upsample H/2 → H (no skip — VGG enc1 already maxpooled)
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

        x = self.dec5(torch.cat([self.up5(s5), s4], dim=1))
        x = self.dec4(torch.cat([self.up4(x), s3], dim=1))
        x = self.dec3(torch.cat([self.up3(x), s2], dim=1))
        x = self.dec2(torch.cat([self.up2(x), s1], dim=1))

        x = self.dec1(self.up1(x))
        return self.out(x)
