# HF Space (Docker) — CityVision prediction API (FastAPI), publicly callable.
# Build context = project root:  docker build -f deploy/hf/api/Dockerfile -t cityvision-api .
FROM python:3.10-slim

WORKDIR /app

# CPU PyTorch (deps from PyPI).
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    torch==2.12.0 torchvision==0.27.0

# API runtime deps (no streamlit — this Space only serves the API).
RUN pip install --no-cache-dir \
    fastapi==0.136.3 uvicorn==0.49.0 python-multipart==0.0.32 \
    numpy==2.2.6 pillow==12.2.0

# Shared package + API code + baked-in model weight.
# (backend/model/<arch>_best.pth must be present — commit it via git-lfs.)
COPY cityvision/ cityvision/
COPY backend/ backend/

ENV CITYVISION_ARCH=ResNet50-UNet

# HF routes external traffic to app_port (7860). Serve the API there so it's
# publicly reachable (Swagger at /docs, prediction at /predict).
EXPOSE 7860
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]
