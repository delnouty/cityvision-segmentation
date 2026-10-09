---
title: CityVision PoC Dashboard
emoji: 🛣️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# CityVision — Proof-of-concept dashboard

Semantic segmentation of urban street scenes (Cityscapes, 9 classes). This
dashboard compares the project's **ResNet50-UNet** baseline with two recent
transformers, **Mask2Former** (CVPR 2022) and **EoMT** (Kerssies et al., 2025,
arXiv:2503.19108).

- **Dataset**: counts per split, city and class; example images by city and by
  class; image transformations (histogram equalisation, CLAHE, blur,
  auto-contrast, model normalisation).
- **Prediction**: choose an image and a model; see the predicted mask next to the
  real one, with per-class scores. The three models run inside this Space, on CPU
  (a few seconds per image).
- **Model comparison**: accuracy, speed, bootstrap confidence intervals, training
  curves.

Charts follow the essential WCAG criteria: text alternatives, data tables,
colour-blind-safe series colours, keyboard-operable controls.

Source: https://github.com/delnouty/cityvision-segmentation (branch `poc-mask2former`)
