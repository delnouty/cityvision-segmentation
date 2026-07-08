---
title: CityVision Segmentation API
emoji: 🛣️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# CityVision — Segmentation API

FastAPI endpoint serving the trained urban-scene segmentation model.

- `POST /predict` — image → predicted mask (JSON)
- `POST /predict/image` — image → predicted mask (PNG)
- `POST /predict/classes` — image → detected classes
- Interactive docs at `/docs`

Source: https://github.com/delnouty/cityvision-segmentation
