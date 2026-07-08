---
title: CityVision Segmentation (Demo UI)
emoji: 🛣️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# CityVision — Demo UI

Streamlit app that **consumes the CityVision prediction API**: lists available
image IDs, sends the selected one to the API, and shows the real image, the real
mask, and the predicted mask.

Set the `CITYVISION_API` env var (or edit the Dockerfile) to point at your
deployed API Space.

Source: https://github.com/delnouty/cityvision-segmentation
