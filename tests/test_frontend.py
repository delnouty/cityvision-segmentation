"""Unit tests for the frontend helpers (frontend/utils.py).

The Streamlit UI script itself is not imported (it renders on import); all the
testable logic lives in utils.py.
"""

import os

import numpy as np
import pytest
import requests

import utils  # frontend/ is on sys.path via conftest

# ============================================================
# Pure helpers
# ============================================================


def test_image_id():
    p = os.path.join("a", "b", "berlin", "berlin_000000_000019_leftImg8bit.png")
    assert utils.image_id(p) == "berlin_000000_000019"


def test_remap_labels():
    raw = np.array([[7, 11, 0], [21, 99, 33]], dtype=np.int32)
    out = utils.remap_labels(raw)
    assert out.dtype == np.uint8
    # 7->1 (road), 11->2 (building), 0->0, 21->3 (veg), 99->0 (unmapped), 33->8 (bicycle)
    assert out.tolist() == [[1, 2, 0], [3, 0, 8]]


# ============================================================
# API client (requests mocked)
# ============================================================


class _FakeResp:
    def __init__(self, *, json_data=None, content=b"", status=200):
        self._json = json_data
        self.content = content
        self.status = status

    def raise_for_status(self):
        if self.status >= 400:
            raise requests.HTTPError(f"status {self.status}")

    def json(self):
        return self._json


def test_api_health(monkeypatch):
    captured = {}

    def fake_get(url, timeout=None):
        captured["url"] = url
        return _FakeResp(json_data={"architecture": "ResNet50-UNet", "device": "cpu"})

    monkeypatch.setattr(utils.requests, "get", fake_get)
    out = utils.api_health("http://api:8000")
    assert out["architecture"] == "ResNet50-UNet"
    assert captured["url"] == "http://api:8000/health"


def test_api_predict_posts_file_and_params(monkeypatch, png_file):
    captured = {}

    def fake_post(url, params=None, files=None, timeout=None):
        captured.update(url=url, params=params, has_file="file" in (files or {}))
        return _FakeResp(content=b"\x89PNG-bytes")

    monkeypatch.setattr(utils.requests, "post", fake_post)
    content = utils.api_predict("http://api:8000", png_file, fmt="overlay", alpha=0.3)
    assert content == b"\x89PNG-bytes"
    assert captured["url"] == "http://api:8000/predict"
    assert captured["params"] == {"format": "overlay", "alpha": 0.3}
    assert captured["has_file"] is True


def test_api_classes(monkeypatch, png_file):
    def fake_post(url, files=None, timeout=None):
        return _FakeResp(
            json_data={
                "model": "DummyNet",
                "detected_classes": [{"class_name": "road", "percentage": 50.0}],
            }
        )

    monkeypatch.setattr(utils.requests, "post", fake_post)
    out = utils.api_classes("http://api:8000", png_file)
    assert out["model"] == "DummyNet"
    assert out["detected_classes"][0]["class_name"] == "road"


def test_api_predict_raises_on_http_error(monkeypatch, png_file):
    monkeypatch.setattr(utils.requests, "post", lambda *a, **k: _FakeResp(status=500))
    with pytest.raises(requests.HTTPError):
        utils.api_predict("http://api:8000", png_file)


# ============================================================
# Dataset-backed helpers (skipped if the dataset is absent)
# ============================================================


def _has_split(split: str) -> bool:
    return os.path.isdir(os.path.join(utils.IMG_ROOT, split))


@pytest.mark.skipif(not _has_split("val"), reason="Cityscapes val split not present")
def test_build_pairs_val():
    pairs = utils.build_pairs("val")
    assert len(pairs) > 0
    some_id = next(iter(pairs))
    img_path, mask_path = pairs[some_id]
    assert os.path.exists(img_path) and os.path.exists(mask_path)
    assert utils.image_id(img_path) == some_id


@pytest.mark.skipif(not _has_split("val"), reason="Cityscapes val split not present")
def test_colorize_gt_val_has_real_labels():
    pairs = utils.build_pairs("val")
    _, mask_path = next(iter(pairs.values()))
    img, has_real = utils.colorize_gt(mask_path)
    assert img.mode == "RGB"
    assert has_real is True


@pytest.mark.skipif(not _has_split("test"), reason="Cityscapes test split not present")
def test_colorize_gt_test_has_no_real_labels():
    pairs = utils.build_pairs("test")
    _, mask_path = next(iter(pairs.values()))
    _, has_real = utils.colorize_gt(mask_path)
    assert has_real is False  # test GT is withheld -> all background
