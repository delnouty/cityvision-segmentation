"""
Smoke test for the Mask2Former PoC — run before any long training.

  1. Load   — ADE20K checkpoint loads with a fresh 9-class head; forward shapes.
  2. Sanity — the Cityscapes-trained checkpoint, pushed through *our*
              post-processing + 19->9 remap + metric, must score high on a few
              val images. Proves semantic_logits() and the metric are wired right.
  3. Memory — one full training step at 512x1024 fits on the GPU.
  4. Overfit — a few images, many steps: loss must fall and train mIoU rise.
              Proves targets, loss and optimiser are wired right.

Usage:  python scripts/smoke_test_mask2former.py
"""

import os
import sys
import time

import torch
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from dataloader import CityscapesDataset, get_cityscapes_pairs  # noqa: E402
from training_mask2former import (  # noqa: E402
    _autocast,
    backbone_params,
    build_model,
    semantic_logits,
    to_mask2former_targets,
)
from training_resnet50 import SegmentationMetrics  # noqa: E402
from transformers import Mask2FormerForUniversalSegmentation  # noqa: E402

from cityvision.constants import CLASS_NAMES, NUM_CLASSES  # noqa: E402
from cityvision.models import remap_mask  # noqa: E402

IMG_ROOT = os.path.join(
    ROOT, "data/cityscapes/P8_Cityscapes_leftImg8bit_trainvaltest/leftImg8bit"
)
MASK_ROOT = os.path.join(
    ROOT, "data/cityscapes/P8_Cityscapes_gtFine_trainvaltest/gtFine"
)
CITYSCAPES_CKPT = "facebook/mask2former-swin-tiny-cityscapes-semantic"

# Cityscapes 19 trainId names -> our 9-class index (everything else = background)
_TRAINID_NAME_TO_OURS = {
    "road": 1,
    "building": 2,
    "vegetation": 3,
    "sky": 4,
    "person": 5,
    "car": 6,
    "traffic sign": 7,
    "bicycle": 8,
}

device = "cuda" if torch.cuda.is_available() else "cpu"


def loader(split, n, batch_size=2):
    pairs = get_cityscapes_pairs(IMG_ROOT, MASK_ROOT, split=split)[:n]
    return DataLoader(CityscapesDataset(pairs), batch_size=batch_size, shuffle=False)


def header(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def step_load():
    header("1. Load ADE20K checkpoint with 9-class head")
    model = build_model().to(device).eval()
    n_all = sum(p.numel() for p in model.parameters())
    n_enc = sum(p.numel() for p in backbone_params(model))
    print(f"params: {n_all / 1e6:.1f} M total, {n_enc / 1e6:.1f} M Swin backbone")

    imgs, _ = next(iter(loader("val", 1, batch_size=1)))
    imgs = imgs.to(device)
    with torch.no_grad(), _autocast(device):
        model(pixel_values=imgs)  # warm-up
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(pixel_values=imgs)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    logits = semantic_logits(out, imgs.shape[-2:])
    print(
        f"class_queries_logits {tuple(out.class_queries_logits.shape)}  (Q, {NUM_CLASSES}+1)"
    )
    print(
        f"masks_queries_logits {tuple(out.masks_queries_logits.shape)}  (Q, H/4, W/4)"
    )
    print(f"semantic logits      {tuple(logits.shape)}")
    print(f"inference 1 image 512x1024: {dt * 1000:.0f} ms")
    assert logits.shape == (1, NUM_CLASSES, *imgs.shape[-2:])
    del model


def step_sanity(n_images=10):
    header(f"2. Post-processing sanity: Cityscapes checkpoint on {n_images} val images")
    model = (
        Mask2FormerForUniversalSegmentation.from_pretrained(CITYSCAPES_CKPT)
        .to(device)
        .eval()
    )
    id2label = model.config.id2label
    lut = torch.zeros(len(id2label), dtype=torch.long, device=device)
    for i, name in id2label.items():
        lut[int(i)] = _TRAINID_NAME_TO_OURS.get(name, 0)
    missing = set(_TRAINID_NAME_TO_OURS) - set(id2label.values())
    assert not missing, f"labels not found in checkpoint: {missing}"

    metrics = SegmentationMetrics(NUM_CLASSES)
    with torch.no_grad():
        for imgs, masks in loader("val", n_images):
            imgs, masks = imgs.to(device), remap_mask(masks).to(device)
            with _autocast(device):
                out = model(pixel_values=imgs)
            preds19 = semantic_logits(out, masks.shape[-2:]).argmax(dim=1)
            metrics.update(lut[preds19], masks)
    iou = metrics.iou_per_class()
    for name, v in zip(CLASS_NAMES, iou):
        print(f"  {name:<15s} {v:.3f}")
    print(f"mIoU (9-class, remapped from 19): {metrics.mean_iou():.3f}")
    # Not the paper's ~0.8: this checkpoint was trained at 1024x2048 with void
    # pixels ignored, whereas our scheme scores them as "background".
    print(
        "  expect ~0.7 at 512x1024; far below (<0.5) means post-processing or remap is wrong"
    )
    del model


def step_memory():
    header("3. One training step, batch 2, 512x1024")
    model = build_model().to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    imgs, masks = next(iter(loader("train", 2)))
    imgs, masks = imgs.to(device), remap_mask(masks).to(device)
    mask_labels, class_labels = to_mask2former_targets(masks)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with _autocast(device):
        out = model(
            pixel_values=imgs, mask_labels=mask_labels, class_labels=class_labels
        )
    out.loss.backward()
    opt.step()
    print(f"loss: {out.loss.item():.3f}")
    if device == "cuda":
        peak = torch.cuda.max_memory_allocated() / 2**30
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        print(f"peak GPU memory: {peak:.2f} / {total:.2f} GiB")
    del model, opt


def step_overfit(n_images=8, steps=150):
    header(f"4. Overfit {n_images} train images for {steps} steps")
    torch.manual_seed(0)
    model = build_model().to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    batches = [
        (i.to(device), remap_mask(m).to(device)) for i, m in loader("train", n_images)
    ]

    losses = []
    for step in range(1, steps + 1):
        imgs, masks = batches[step % len(batches)]
        mask_labels, class_labels = to_mask2former_targets(masks)
        opt.zero_grad()
        with _autocast(device):
            out = model(
                pixel_values=imgs, mask_labels=mask_labels, class_labels=class_labels
            )
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
        opt.step()
        losses.append(out.loss.item())
        if step % 25 == 0:
            print(f"  step {step:4d}  loss {sum(losses[-25:]) / 25:.3f}")

    model.eval()
    metrics = SegmentationMetrics(NUM_CLASSES)
    with torch.no_grad():
        for imgs, masks in batches:
            with _autocast(device):
                out = model(pixel_values=imgs)
            metrics.update(semantic_logits(out, masks.shape[-2:]).argmax(dim=1), masks)
    first, last = sum(losses[:10]) / 10, sum(losses[-10:]) / 10
    print(
        f"loss {first:.3f} -> {last:.3f}   train mIoU on these images: {metrics.mean_iou():.3f}"
    )
    assert last < first, "loss did not decrease — training loop is broken"


if __name__ == "__main__":
    print("Device:", torch.cuda.get_device_name(0) if device == "cuda" else "CPU")
    step_load()
    step_sanity()
    step_memory()
    step_overfit()
    print("\nSmoke test finished.")
