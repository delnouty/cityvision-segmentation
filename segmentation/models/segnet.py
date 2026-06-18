"""SegNet — encoder/decoder with max-pool index unpooling, trained from scratch.

No skip connections; the decoder restores spatial detail using the max-pool
*indices* saved during encoding. BatchNorm after every convolution.
"""

import torch.nn as nn

from ..data import NUM_CLASSES
from .base import BaseSegModel


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


class SegNet(BaseSegModel):
    """SegNet-inspired encoder-decoder trained fully from scratch."""

    encoder_modules = ()  # no pretrained backbone → single learning rate

    def __init__(self, num_classes: int = NUM_CLASSES, **_unused):
        super().__init__()

        # ---- Encoder ----
        self.enc1 = _conv_block(3,   64,  2)
        self.enc2 = _conv_block(64,  128, 2)
        self.enc3 = _conv_block(128, 256, 3)
        self.enc4 = _conv_block(256, 512, 3)
        self.enc5 = _conv_block(512, 512, 3)

        # Shared pool / unpool objects (stateless, reused)
        self.pool   = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        # ---- Decoder (mirrors encoder in reverse) ----
        self.dec5 = _conv_block(512, 512, 3)
        self.dec4 = _conv_block(512, 256, 3)
        self.dec3 = _conv_block(256, 128, 3)
        self.dec2 = _conv_block(128, 64,  2)
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
        # --- Encode (save pool indices for unpooling) ---
        x = self.enc1(x); x, idx1 = self.pool(x)  # H/2
        x = self.enc2(x); x, idx2 = self.pool(x)  # H/4
        x = self.enc3(x); x, idx3 = self.pool(x)  # H/8
        x = self.enc4(x); x, idx4 = self.pool(x)  # H/16
        x = self.enc5(x); x, idx5 = self.pool(x)  # H/32

        # --- Decode (restore resolution via index-based unpooling) ---
        x = self.dec5(self.unpool(x, idx5))  # H/16
        x = self.dec4(self.unpool(x, idx4))  # H/8
        x = self.dec3(self.unpool(x, idx3))  # H/4
        x = self.dec2(self.unpool(x, idx2))  # H/2
        x = self.dec1(self.unpool(x, idx1))  # H (original)

        return x
