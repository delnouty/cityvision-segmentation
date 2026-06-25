"""Command-line entry point for configurable segmentation training.

Examples
--------
    # Train with a preset architecture
    python -m segmentation.train --arch resnet50

    # Override hyperparameters on top of the preset
    python -m segmentation.train --arch unet --epochs 30 --batch-size 4 --lr 5e-5

    # Train a from-scratch (non-pretrained) ResNet34 encoder
    python -m segmentation.train --arch resnet34 --no-pretrained
"""

import argparse

from .config import ARCH_PRESETS, make_config
from .engine import fit


def _parse_img_size(value: str):
    """Parse 'HxW' (e.g. '512x1024') into a (H, W) tuple."""
    try:
        h, w = value.lower().split("x")
        return (int(h), int(w))
    except Exception:
        raise argparse.ArgumentTypeError("img-size must look like '512x1024'")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train a semantic-segmentation model. The architecture is "
        "selected with --arch; every other flag overrides that "
        "architecture's preset."
    )

    p.add_argument(
        "--arch",
        required=True,
        choices=sorted(ARCH_PRESETS),
        help="Architecture of the solution to train.",
    )

    # Hyperparameter overrides — default None means "keep the preset value".
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument(
        "--encoder-lr-mult", type=float, default=None, dest="encoder_lr_mult"
    )
    p.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early-stopping patience (0 disables).",
    )
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument(
        "--img-size",
        type=_parse_img_size,
        default=None,
        dest="img_size",
        help="Input resolution as HxW, e.g. 512x1024.",
    )
    p.add_argument(
        "--scheduler",
        choices=["none", "plateau"],
        default=None,
        help="LR scheduler. 'plateau' = ReduceLROnPlateau on val mIoU.",
    )

    pre = p.add_mutually_exclusive_group()
    pre.add_argument(
        "--pretrained",
        action="store_true",
        default=None,
        help="Use ImageNet-pretrained encoder weights.",
    )
    pre.add_argument(
        "--no-pretrained",
        action="store_false",
        default=None,
        dest="pretrained",
        help="Train the encoder from scratch.",
    )

    bal = p.add_mutually_exclusive_group()
    bal.add_argument(
        "--balanced",
        action="store_true",
        default=None,
        help="Use the class-balanced WeightedRandomSampler.",
    )
    bal.add_argument(
        "--no-balanced",
        action="store_false",
        default=None,
        dest="balanced",
        help="Plain random shuffling instead.",
    )

    p.add_argument(
        "--checkpoint-name",
        default=None,
        dest="checkpoint_name",
        help="Filename under backend/model/ for the best checkpoint.",
    )
    p.add_argument("--run-name", default=None, dest="run_name", help="MLflow run name.")
    p.add_argument("--experiment", default=None, help="MLflow experiment name.")
    p.add_argument(
        "--registered-model-name",
        default=None,
        dest="registered_model_name",
        help="MLflow Model Registry name (default: cityvision-segmentation-<arch>).",
    )
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    # Scheduler is handled separately because "disable" must be able to override
    # a preset that enables it — make_config() ignores None overrides.
    overrides = {k: v for k, v in vars(args).items() if k not in ("arch", "scheduler")}

    config = make_config(args.arch, **overrides)
    if args.scheduler is not None:
        config.scheduler = None if args.scheduler == "none" else args.scheduler

    fit(config)


if __name__ == "__main__":
    main()
