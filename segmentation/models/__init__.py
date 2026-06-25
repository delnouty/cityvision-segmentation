"""Architecture registry.

``build_model(name, ...)`` returns the requested model. ``ARCHITECTURES`` maps
the user-facing name to a builder callable; this is the single place to register
a new architecture so it becomes selectable from the CLI/config.
"""

from ..data import NUM_CLASSES
from .base import BaseSegModel
from .unet import UNet
from .resnet_unet import ResNetUNet
from .segnet import SegNet
from .vgg_unet import VGGUNet

# name → builder(num_classes, pretrained, dropout, **kwargs) -> nn.Module
ARCHITECTURES = {
    "unet": lambda **kw: UNet(**kw),
    "resnet34": lambda **kw: ResNetUNet(depth=34, **kw),
    "resnet50": lambda **kw: ResNetUNet(depth=50, **kw),
    "segnet": lambda **kw: SegNet(**kw),
    "vgg": lambda **kw: VGGUNet(**kw),
}


def build_model(
    name: str,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    dropout: float = 0.3,
) -> BaseSegModel:
    """Instantiate an architecture by name.

    Extra kwargs (pretrained, dropout) are accepted by every model — those that
    don't apply (e.g. dropout for ResNet, pretrained for SegNet) are ignored.
    """
    if name not in ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture '{name}'. Choose from: {', '.join(sorted(ARCHITECTURES))}"
        )
    return ARCHITECTURES[name](
        num_classes=num_classes, pretrained=pretrained, dropout=dropout
    )


__all__ = ["ARCHITECTURES", "build_model", "BaseSegModel"]
