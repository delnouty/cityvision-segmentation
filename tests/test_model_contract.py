"""Contract tests between the model definitions, the notebooks and the
checkpoints on disk.

Motivation: the notebooks redefine the architectures inline (to display them),
while `backend/inference.py` rebuilds them from `cityvision.models` when serving.
Nothing tied the two together, so a notebook decoder that drifted from the
package produced a `.pth` the API could not load — and the failure only surfaced
at deployment, hours after the run finished.

These tests close that gap:
  * every architecture in the checkpoint map can be built and returns the right
    output shape;
  * each notebook's inline definitions have the same `state_dict` signature as
    the package's;
  * every checkpoint present on disk loads into its declared architecture.

The checkpoint test skips when the weights are absent (they are gitignored, so CI
never sees them); the others run everywhere.
"""

import json
import os
from functools import lru_cache

import pytest
import torch
import torch.nn as nn
import torchvision.models as tv_models

from backend.inference import CKPT_BY_ARCH, MODEL_DIR
from cityvision.constants import NUM_CLASSES
from cityvision import models as pkg

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NOTEBOOK_DIR = os.path.join(PROJECT_ROOT, "notebooks")

# Small enough to stay fast on CPU, but both dimensions must survive five
# stride-2 stages: 64/32 = 2 and 128/32 = 4.
PROBE_SHAPE = (1, 3, 64, 128)

# architecture name -> builder from the package (pretrained=False: no download)
PKG_BUILDERS = {
    "UNet": lambda: pkg.UNet(NUM_CLASSES),
    "VGG16-UNet": lambda: pkg.VGGUNet(NUM_CLASSES, pretrained=False),
    "SegNet": lambda: pkg.SegNet(NUM_CLASSES),
    "ResNet34-UNet": lambda: pkg.ResNetUNet(NUM_CLASSES, pretrained=False),
    "ResNet50-UNet": lambda: pkg.ResNet50UNet(NUM_CLASSES, pretrained=False),
}

# How the same architecture is spelled with the notebooks' inline classes.
NOTEBOOK_BUILDERS = {
    "UNet": "UNet(NUM_CLASSES)",
    "VGG16-UNet": "VGGUNet(NUM_CLASSES, pretrained=False)",
    "SegNet": "SegNet(NUM_CLASSES)",
    "ResNet34-UNet": "ResNetUNet(NUM_CLASSES, depth=34, pretrained=False)",
    "ResNet50-UNet": "ResNetUNet(NUM_CLASSES, depth=50, pretrained=False)",
}

NOTEBOOKS = [
    "training_segnet.ipynb",
    "training_comparison.ipynb",
    "training_augmented.ipynb",
]


def signature(model: nn.Module) -> dict:
    """Parameter/buffer name -> shape. This is exactly what `load_state_dict`
    compares, so two models with equal signatures are checkpoint-compatible."""
    return {k: tuple(v.shape) for k, v in model.state_dict().items()}


@lru_cache(maxsize=None)
def package_signature(arch: str) -> tuple:
    # dicts are unhashable, so lru_cache stores sorted items.
    return tuple(sorted(signature(PKG_BUILDERS[arch]()).items()))


@lru_cache(maxsize=None)
def notebook_namespace(filename: str) -> dict:
    """Execute a notebook's '§5 architecture definitions' cell and return the
    resulting namespace. Only class definitions are executed — no training, no
    dataset, no weight download."""
    path = os.path.join(NOTEBOOK_DIR, filename)
    if not os.path.exists(path):
        pytest.skip(f"notebook not found: {filename}")

    cells = json.load(open(path, encoding="utf-8"))["cells"]
    source = next(
        (
            "".join(c["source"])
            for c in cells
            if c["cell_type"] == "code" and "_RESNET_SPECS" in "".join(c["source"])
        ),
        None,
    )
    if source is None:
        pytest.skip(f"no architecture cell in {filename}")

    ns = {
        "torch": torch,
        "nn": nn,
        "tv_models": tv_models,
        "NUM_CLASSES": NUM_CLASSES,
        "__name__": "notebook_arch",
    }
    exec(compile(source, f"{filename}:architectures", "exec"), ns)
    return ns


# ============================================================
# the checkpoint map itself
# ============================================================


def test_checkpoint_map_matches_known_architectures():
    """inference.CKPT_BY_ARCH and the builders here must describe the same set —
    otherwise an architecture is servable but untested, or vice versa."""
    assert set(CKPT_BY_ARCH) == set(PKG_BUILDERS)
    assert set(CKPT_BY_ARCH) == set(NOTEBOOK_BUILDERS)


def test_checkpoint_filenames_are_distinct():
    """Two architectures sharing a filename would silently overwrite each other."""
    names = list(CKPT_BY_ARCH.values())
    assert len(names) == len(set(names))


# ============================================================
# the package models
# ============================================================


@pytest.mark.parametrize("arch", sorted(PKG_BUILDERS))
def test_architecture_builds_and_output_shape(arch):
    model = PKG_BUILDERS[arch]()
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(PROBE_SHAPE))
    n, _, h, w = PROBE_SHAPE
    assert out.shape == (n, NUM_CLASSES, h, w), f"{arch} returned {tuple(out.shape)}"


# ============================================================
# notebooks vs package — the regression test
# ============================================================


@pytest.mark.parametrize("notebook", NOTEBOOKS)
@pytest.mark.parametrize("arch", sorted(NOTEBOOK_BUILDERS))
def test_notebook_architecture_matches_package(notebook, arch):
    """A notebook-trained checkpoint must be loadable by backend/inference.py,
    which rebuilds from cityvision.models. Equal state_dict signatures is the
    condition for that."""
    ns = notebook_namespace(notebook)
    nb_model = eval(NOTEBOOK_BUILDERS[arch], dict(ns))  # noqa: S307 - fixed strings

    nb_sig = signature(nb_model)
    pkg_sig = dict(package_signature(arch))

    missing = sorted(set(pkg_sig) - set(nb_sig))
    extra = sorted(set(nb_sig) - set(pkg_sig))
    mismatched = {
        k: (nb_sig[k], pkg_sig[k])
        for k in set(nb_sig) & set(pkg_sig)
        if nb_sig[k] != pkg_sig[k]
    }

    assert not missing, f"{notebook} / {arch}: absent from the notebook: {missing[:5]}"
    assert not extra, f"{notebook} / {arch}: only in the notebook: {extra[:5]}"
    assert not mismatched, (
        f"{notebook} / {arch}: shape mismatch (notebook, package): "
        f"{dict(list(mismatched.items())[:5])}"
    )


# ============================================================
# checkpoints on disk
# ============================================================


@pytest.mark.parametrize("arch", sorted(CKPT_BY_ARCH))
def test_checkpoint_loads_into_declared_architecture(arch):
    """Weights are gitignored, so this skips in CI and guards local runs — run it
    before pushing an image or a Space."""
    path = os.path.join(MODEL_DIR, CKPT_BY_ARCH[arch])
    if not os.path.exists(path):
        pytest.skip(f"no checkpoint for {arch}: {os.path.basename(path)}")

    state = torch.load(path, map_location="cpu")
    model = PKG_BUILDERS[arch]()
    # strict=True: a missing or unexpected key means the API would refuse it.
    model.load_state_dict(state, strict=True)
