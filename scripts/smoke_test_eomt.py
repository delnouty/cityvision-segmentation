"""
Smoke test for the EoMT PoC — run before any long training.

  1. Load    — COCO checkpoint loads with a fresh 9-class head at 512x1024;
               output shapes; mask-free inference speed.
  2. Resize  — the *pretrained* COCO model (its own 133 classes), run at
               512x1024 with interpolated position embeddings, must still find
               road / car / person... on Cityscapes val images. Proves the
               40x40 -> 32x64 embedding interpolation keeps what it learned.
  3. Memory  — one training step at 512x1024, masked attention on (the
               heaviest mode), fits on the GPU.
  4. Overfit — a few images, with mask annealing compressed into the run:
               loss must fall, and once annealing is over, mask-free inference
               must score as well as masked inference. Proves the annealing
               hands over to a mask-free model, as it will in the real run.

Usage:  python scripts/smoke_test_eomt.py
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
from training_eomt import (  # noqa: E402
    PRETRAINED,
    MaskAnnealer,
    _resize_position_embeddings,
    build_model,
    validate_mask_free,
)
from training_mask2former import (  # noqa: E402
    _autocast,
    semantic_logits,
    to_mask2former_targets,
)
from training_resnet50 import SegmentationMetrics  # noqa: E402
from transformers import EomtForUniversalSegmentation  # noqa: E402

from cityvision.constants import CLASS_NAMES, NUM_CLASSES  # noqa: E402
from cityvision.models import remap_mask  # noqa: E402

IMG_ROOT = os.path.join(
    ROOT, "data/cityscapes/P8_Cityscapes_leftImg8bit_trainvaltest/leftImg8bit"
)
MASK_ROOT = os.path.join(
    ROOT, "data/cityscapes/P8_Cityscapes_gtFine_trainvaltest/gtFine"
)

# COCO panoptic names -> our 9-class index. COCO has no "traffic sign" class,
# so that class is left out of the step-2 check.
_COCO_TO_OURS = {
    "road": 1,
    "building-other-merged": 2,
    "house": 2,
    "tree-merged": 3,
    "sky-other-merged": 4,
    "person": 5,
    "car": 6,
    "bicycle": 8,
}

device = "cuda" if torch.cuda.is_available() else "cpu"


def loader(split, n, batch_size=2):
    pairs = get_cityscapes_pairs(IMG_ROOT, MASK_ROOT, split=split)[:n]
    return DataLoader(CityscapesDataset(pairs), batch_size=batch_size, shuffle=False)


def header(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def evaluate(model, batches, mask_free=True):
    """mIoU on a list of (imgs, masks) batches, with or without masked attention."""
    saved = model.attn_mask_probs.clone()
    model.eval()
    if mask_free:
        model.attn_mask_probs.zero_()
    else:
        model.attn_mask_probs.fill_(1.0)
    metrics = SegmentationMetrics(NUM_CLASSES)
    with torch.no_grad():
        for imgs, masks in batches:
            with _autocast(device):
                out = model(pixel_values=imgs)
            metrics.update(semantic_logits(out, masks.shape[-2:]).argmax(dim=1), masks)
    model.attn_mask_probs.copy_(saved)
    return metrics


def step_load():
    header("1. Load COCO checkpoint with 9-class head, 512x1024")
    model = build_model().to(device).eval()
    model.attn_mask_probs.zero_()
    n_all = sum(p.numel() for p in model.parameters())
    print(f"params: {n_all / 1e6:.1f} M, {model.config.num_blocks} query blocks")
    print(
        f"position embeddings: {tuple(model.embeddings.position_embeddings.weight.shape)}"
    )

    imgs, _ = next(iter(loader("val", 1, batch_size=1)))
    imgs = imgs.to(device)
    with torch.no_grad(), _autocast(device):
        for _ in range(3):
            model(pixel_values=imgs)  # warm-up
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            out = model(pixel_values=imgs)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 10
    logits = semantic_logits(out, imgs.shape[-2:])
    print(
        f"class_queries_logits {tuple(out.class_queries_logits.shape)}  (Q, {NUM_CLASSES}+1)"
    )
    print(
        f"masks_queries_logits {tuple(out.masks_queries_logits.shape)}  (Q, H/4, W/4)"
    )
    print(f"semantic logits      {tuple(logits.shape)}")
    print(f"mask-free inference 1 image 512x1024: {dt * 1000:.0f} ms")
    assert logits.shape == (1, NUM_CLASSES, *imgs.shape[-2:])
    del model


def step_resize(n_images=10):
    header(f"2. Interpolated position embeddings: COCO model on {n_images} val images")
    model = EomtForUniversalSegmentation.from_pretrained(PRETRAINED)
    patch = model.config.patch_size
    _resize_position_embeddings(model, (512 // patch, 1024 // patch))
    model = model.to(device).eval()
    model.attn_mask_probs.zero_()

    names = model.config.id2label
    lut = torch.zeros(len(names), dtype=torch.long, device=device)
    for i, name in names.items():
        lut[int(i)] = _COCO_TO_OURS.get(name, 0)

    metrics = SegmentationMetrics(NUM_CLASSES)
    with torch.no_grad():
        for imgs, masks in loader("val", n_images):
            imgs, masks = imgs.to(device), remap_mask(masks).to(device)
            with _autocast(device):
                out = model(pixel_values=imgs)
            preds = lut[semantic_logits(out, masks.shape[-2:]).argmax(dim=1)]
            metrics.update(preds, masks)
    iou = metrics.iou_per_class()
    checked = [
        i for i, n in enumerate(CLASS_NAMES) if n not in ("background", "traffic_sign")
    ]
    for i in checked:
        print(f"  {CLASS_NAMES[i]:<15s} {iou[i]:.3f}")
    miou = float(torch.tensor(iou[checked]).nanmean())
    print(f"mIoU over these 7 classes (zero-shot, COCO labels): {miou:.3f}")
    print(
        "  expect clearly > 0.5 (road/car/sky high); near 0 means the resize broke the model"
    )
    del model


def step_memory():
    header("3. One training step, batch 2, 512x1024, masked attention on")
    model = build_model().to(device).train()
    model.attn_mask_probs.fill_(1.0)
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


def step_overfit(n_images=8, steps=250):
    header(f"4. Overfit {n_images} train images, {steps} steps, annealing compressed")
    torch.manual_seed(0)
    model = build_model().to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    annealer = MaskAnnealer(model, total_steps=steps)
    print(f"annealing starts {annealer.starts}, ends {annealer.ends}")
    raw = list(loader("train", n_images))  # validate() remaps labels itself
    batches = [(i.to(device), remap_mask(m).to(device)) for i, m in raw]

    losses = []
    for step in range(1, steps + 1):
        imgs, masks = batches[step % len(batches)]
        mask_labels, class_labels = to_mask2former_targets(masks)
        model.train()
        opt.zero_grad()
        with _autocast(device):
            out = model(
                pixel_values=imgs, mask_labels=mask_labels, class_labels=class_labels
            )
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
        opt.step()
        annealer.step()
        losses.append(out.loss.item())
        if step % 50 == 0:
            probs = [round(p, 2) for p in model.attn_mask_probs.tolist()]
            print(
                f"  step {step:4d}  loss {sum(losses[-50:]) / 50:.3f}  mask probs {probs}"
            )

    assert annealer.done and float(model.attn_mask_probs.max()) == 0.0
    free = evaluate(model, batches, mask_free=True).mean_iou()
    masked = evaluate(model, batches, mask_free=False).mean_iou()
    first, last = sum(losses[:10]) / 10, sum(losses[-10:]) / 10
    print(f"loss {first:.3f} -> {last:.3f}")
    print(f"train mIoU on these images: mask-free {free:.3f} | masked {masked:.3f}")
    assert last < first, "loss did not decrease — training loop is broken"

    # validate_mask_free() is what the training script calls every epoch
    val_loss, val_m = validate_mask_free(model, raw, device)
    print(
        f"validate_mask_free on the same batches: loss {val_loss:.3f}, mIoU {val_m.mean_iou():.3f}"
    )
    assert float(model.attn_mask_probs.max()) == 0.0, "probs not restored"


if __name__ == "__main__":
    print("Device:", torch.cuda.get_device_name(0) if device == "cuda" else "CPU")
    step_load()
    step_resize()
    step_memory()
    step_overfit()
    print("\nSmoke test finished.")
