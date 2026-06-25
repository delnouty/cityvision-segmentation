"""Shared pytest fixtures for backend + frontend tests."""

import io
import os
import sys

import numpy as np
import pytest
from PIL import Image

# Make project packages importable: backend.*, and the frontend `utils` module.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "frontend")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from backend.inference import Segmenter, PALETTE, CLASS_NAMES, NUM_CLASSES, IMG_SIZE


class DummySegmenter:
    """Lightweight stand-in for the real Segmenter so API tests don't load a
    160 MB checkpoint. Reuses the real rendering/static methods for fidelity."""

    arch = "DummyNet"
    checkpoint_path = "dummy.pth"
    val_miou = 0.5
    device = "cpu"

    def predict(self, image: Image.Image) -> np.ndarray:
        w, h = image.size
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[:, : w // 2] = 1  # left half -> "road"
        mask[: h // 2, w // 2 :] = 4  # top-right -> "sky"
        return mask

    def colorize(self, mask):
        return Segmenter.colorize(mask)

    def overlay(self, image, mask, alpha=0.5):
        return Segmenter.overlay(image, mask, alpha)

    def class_summary(self, mask):
        return Segmenter.class_summary(mask)

    def rle_encode(self, mask):
        return Segmenter.rle_encode(mask)

    def info(self):
        return {
            "architecture": self.arch,
            "checkpoint": self.checkpoint_path,
            "val_mIoU": self.val_miou,
            "device": self.device,
            "num_classes": NUM_CLASSES,
            "classes": CLASS_NAMES,
            "palette": [[int(c) for c in PALETTE[i]] for i in range(NUM_CLASSES)],
            "input_size": {"height": IMG_SIZE[0], "width": IMG_SIZE[1]},
        }


@pytest.fixture
def png_bytes() -> bytes:
    """A small valid PNG image."""
    buf = io.BytesIO()
    Image.new("RGB", (64, 32), (120, 130, 140)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def png_file(tmp_path, png_bytes) -> str:
    """A small valid PNG written to disk; returns its path."""
    p = tmp_path / "sample.png"
    p.write_bytes(png_bytes)
    return str(p)


@pytest.fixture
def client(monkeypatch):
    """FastAPI TestClient with the model replaced by DummySegmenter."""
    import backend.app as appmod
    from fastapi.testclient import TestClient

    monkeypatch.setattr(appmod, "Segmenter", DummySegmenter)
    with TestClient(appmod.app) as c:  # context triggers startup -> DummySegmenter()
        yield c
