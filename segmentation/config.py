"""Training configuration — the single object that describes a run.

Every architecture ships sensible defaults in ``ARCH_PRESETS`` (carried over
from the original per-model training scripts). ``make_config`` starts from the
preset for the chosen architecture and applies any explicit overrides, so the
"configuration of the solution" is just: pick an architecture, then tweak.
"""

import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@dataclass
class TrainConfig:
    # --- architecture ---
    arch: str = "unet"
    label: str = "UNet"  # human-readable name logged to MLflow
    pretrained: bool = False  # use ImageNet weights for the encoder
    dropout: float = 0.3  # only used by architectures that support it

    # --- optimisation ---
    epochs: int = 20
    batch_size: int = 4
    lr: float = 1e-4
    encoder_lr_mult: float = 0.1  # encoder LR = lr * this (pretrained models)
    optimizer: str = "Adam"
    scheduler: Optional[str] = None  # None or "plateau" (ReduceLROnPlateau)

    # --- early stopping ---
    patience: int = 0  # 0 disables early stopping

    # --- data ---
    img_size: Tuple[int, int] = (512, 1024)  # (H, W)
    balanced: bool = True
    img_root: str = field(
        default_factory=lambda: os.path.join(
            _PROJECT_ROOT,
            "data/cityscapes/P8_Cityscapes_leftImg8bit_trainvaltest/leftImg8bit",
        )
    )
    mask_root: str = field(
        default_factory=lambda: os.path.join(
            _PROJECT_ROOT, "data/cityscapes/P8_Cityscapes_gtFine_trainvaltest/gtFine"
        )
    )

    # --- bookkeeping ---
    checkpoint_name: str = "model_best.pth"
    experiment: str = "urban-segmentation"
    run_name: Optional[str] = None
    # Name under which each run's best model is registered in the MLflow Model
    # Registry. Each run adds a new version. Defaults (per arch) in make_config.
    registered_model_name: Optional[str] = None

    @property
    def project_root(self) -> str:
        return _PROJECT_ROOT

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(_PROJECT_ROOT, "backend", "model", self.checkpoint_name)

    def to_mlflow_params(self) -> dict:
        d = asdict(self)
        d["img_size"] = f"{self.img_size[0]}x{self.img_size[1]}"
        # Drop absolute paths from the logged params — not useful as hyperparams.
        for k in ("img_root", "mask_root"):
            d.pop(k, None)
        return d


# Default registered-model name per architecture (one registry entry per arch;
# every training run registers a new version under it).
def _default_registered_name(arch: str) -> str:
    return f"cityvision-segmentation-{arch}"


# Per-architecture defaults, mirroring the original standalone training scripts.
ARCH_PRESETS = {
    "unet": dict(
        label="UNet",
        pretrained=False,
        dropout=0.3,
        epochs=60,
        batch_size=8,
        lr=1e-4,
        patience=0,
        checkpoint_name="unet_best.pth",
        run_name="model_train_unet",
    ),
    "resnet34": dict(
        label="ResNet34-UNet",
        pretrained=True,
        epochs=20,
        batch_size=4,
        lr=1e-4,
        encoder_lr_mult=0.1,
        patience=5,
        checkpoint_name="resnet_best.pth",
        run_name="model_train_resnet",
    ),
    "resnet50": dict(
        label="ResNet50-UNet",
        pretrained=True,
        epochs=20,
        batch_size=4,
        lr=1e-4,
        encoder_lr_mult=0.1,
        patience=5,
        checkpoint_name="resnet50_best.pth",
        run_name="model_train_resnet50",
    ),
    "segnet": dict(
        label="SegNet",
        pretrained=False,
        epochs=20,
        batch_size=4,
        lr=1e-3,
        patience=7,
        scheduler="plateau",
        checkpoint_name="segnet_best.pth",
        run_name="model_train_segnet",
    ),
    "vgg": dict(
        label="VGG16-UNet",
        pretrained=True,
        epochs=20,
        batch_size=4,
        lr=1e-4,
        encoder_lr_mult=0.1,
        patience=5,
        checkpoint_name="vgg_best.pth",
        run_name="model_train_vgg",
    ),
}


def make_config(arch: str, **overrides) -> TrainConfig:
    """Build a config from an architecture preset plus explicit overrides.

    Only keys explicitly provided in ``overrides`` (and not None) win over the
    preset, so partial CLI overrides behave intuitively.
    """
    if arch not in ARCH_PRESETS:
        raise ValueError(
            f"Unknown architecture '{arch}'. Choose from: {', '.join(sorted(ARCH_PRESETS))}"
        )
    params = {"arch": arch, **ARCH_PRESETS[arch]}
    for key, value in overrides.items():
        if value is not None:
            params[key] = value
    params.setdefault("registered_model_name", _default_registered_name(arch))
    return TrainConfig(**params)
