"""Configurable semantic-segmentation training package.

The architecture of the solution is selected at run time (CLI flag or config),
sharing one training engine, loss, metric and data pipeline across every model.

    python -m segmentation.train --arch resnet50 --epochs 20

See ``segmentation/README.md`` for the full list of architectures and options.
"""

from .config import TrainConfig, make_config, ARCH_PRESETS
from .models import build_model, ARCHITECTURES

__all__ = [
    "TrainConfig",
    "make_config",
    "ARCH_PRESETS",
    "build_model",
    "ARCHITECTURES",
]
