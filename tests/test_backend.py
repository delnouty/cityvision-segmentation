"""Unit tests for the backend FastAPI service (backend/app.py) and the
inference helpers (backend/inference.py)."""

import io

import numpy as np
import pytest
from PIL import Image

from backend import inference
from backend.inference import Segmenter, PALETTE, CLASS_NAMES, NUM_CLASSES

# ============================================================
# inference.py — pure helpers (no model load)
# ============================================================


def test_palette_and_classes_aligned():
    assert len(CLASS_NAMES) == NUM_CLASSES
    assert PALETTE.shape == (NUM_CLASSES, 3)


def test_colorize_shape_and_color():
    mask = np.zeros((4, 5), dtype=np.uint8)
    mask[0, 0] = 1  # road
    img = Segmenter.colorize(mask)
    arr = np.array(img)
    assert img.mode == "RGB"
    assert arr.shape == (4, 5, 3)
    assert tuple(arr[0, 0]) == tuple(int(c) for c in PALETTE[1])
    assert tuple(arr[1, 1]) == tuple(int(c) for c in PALETTE[0])  # background


def test_overlay_size_matches_base():
    base = Image.new("RGB", (20, 10), (0, 0, 0))
    mask = np.ones((10, 20), dtype=np.uint8)  # all road
    out = Segmenter.overlay(base, mask, alpha=0.5)
    assert out.size == base.size
    assert out.mode == "RGB"


def test_class_summary_sorted_and_complete():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:, :5] = 1  # 50% road
    mask[:2, 5:] = 6  # 10% car
    summary = Segmenter.class_summary(mask)
    # sorted by pixel_count desc
    counts = [c["pixel_count"] for c in summary]
    assert counts == sorted(counts, reverse=True)
    # percentages sum to ~100
    assert abs(sum(c["percentage"] for c in summary) - 100.0) < 0.5
    names = {c["class_name"] for c in summary}
    assert {"background", "road", "car"} <= names


def test_arch_from_filename():
    assert inference._arch_from_filename("resnet50_best.pth") == "ResNet50-UNet"
    assert inference._arch_from_filename("/x/y/unet_best.pth") == "UNet"
    assert inference._arch_from_filename("nope.pth") is None


def test_build_model_returns_correct_classes():
    # pretrained=False -> no weight download; just checks the class wiring.
    assert type(inference.build_model("UNet")).__name__ == "UNet"
    assert type(inference.build_model("SegNet")).__name__ == "SegNet"
    assert type(inference.build_model("ResNet34-UNet")).__name__ == "ResNetUNet"
    assert type(inference.build_model("ResNet50-UNet")).__name__ == "ResNet50UNet"
    assert type(inference.build_model("VGG16-UNet")).__name__ == "VGGUNet"
    with pytest.raises(ValueError):
        inference.build_model("NotAModel")


def test_select_checkpoint_respects_env_arch(monkeypatch):
    monkeypatch.setenv("CITYVISION_ARCH", "ResNet50-UNet")
    monkeypatch.delenv("CITYVISION_CHECKPOINT", raising=False)
    arch, ckpt, miou = inference.select_checkpoint()
    assert arch == "ResNet50-UNet"
    assert ckpt.endswith("resnet50_best.pth")


def test_select_checkpoint_unknown_env_arch(monkeypatch):
    monkeypatch.delenv("CITYVISION_CHECKPOINT", raising=False)
    monkeypatch.setenv("CITYVISION_ARCH", "Bogus")
    with pytest.raises(ValueError):
        inference.select_checkpoint()


# ============================================================
# app.py — HTTP endpoints (model stubbed via `client` fixture)
# ============================================================


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["architecture"] == "DummyNet"
    assert body["num_classes"] == NUM_CLASSES
    assert len(body["palette"]) == NUM_CLASSES
    assert body["classes"] == CLASS_NAMES


def test_index_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "CityVision" in r.text


def _files(png_bytes):
    return {"file": ("sample.png", png_bytes, "image/png")}


def test_predict_color_png(client, png_bytes):
    r = client.post("/predict", params={"format": "color"}, files=_files(png_bytes))
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    img = Image.open(io.BytesIO(r.content))
    assert img.mode == "RGB"
    assert img.size == (64, 32)  # original image size


def test_predict_overlay_png(client, png_bytes):
    r = client.post(
        "/predict", params={"format": "overlay", "alpha": 0.4}, files=_files(png_bytes)
    )
    assert r.status_code == 200
    img = Image.open(io.BytesIO(r.content))
    assert img.size == (64, 32)


def test_predict_raw_is_single_channel(client, png_bytes):
    r = client.post("/predict", params={"format": "raw"}, files=_files(png_bytes))
    assert r.status_code == 200
    img = Image.open(io.BytesIO(r.content))
    assert img.mode == "L"
    assert set(np.unique(np.array(img))).issubset(set(range(NUM_CLASSES)))


def test_predict_default_format_is_color(client, png_bytes):
    r = client.post("/predict", files=_files(png_bytes))
    assert r.status_code == 200
    assert Image.open(io.BytesIO(r.content)).mode == "RGB"


def test_predict_invalid_format_rejected(client, png_bytes):
    r = client.post("/predict", params={"format": "rainbow"}, files=_files(png_bytes))
    assert r.status_code == 422  # fails the regex query validation


def test_predict_invalid_file_returns_400(client):
    r = client.post(
        "/predict", files={"file": ("x.txt", b"not an image", "text/plain")}
    )
    assert r.status_code == 400


def test_predict_classes_json(client, png_bytes):
    r = client.post("/predict/classes", files=_files(png_bytes))
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "DummyNet"
    assert body["image_size"] == {"width": 64, "height": 32}
    classes = body["detected_classes"]
    assert len(classes) >= 1
    for c in classes:
        assert {
            "class_id",
            "class_name",
            "color",
            "pixel_count",
            "percentage",
        } <= c.keys()
