# Backend image: FastAPI prediction API + the trained model weights.
FROM python:3.10-slim

WORKDIR /app

# CPU-only PyTorch (far smaller than the default CUDA build).
# index-url = PyTorch CPU wheels; extra-index-url = PyPI for the dependencies
# (torch's deps aren't all mirrored cleanly on the PyTorch index).
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    torch==2.12.0 torchvision==0.27.0

# Runtime deps. The model classes now live in the shared `cityvision` package
# (torch/torchvision only), so the API no longer pulls in mlflow/tqdm.
RUN pip install --no-cache-dir \
    fastapi==0.136.3 uvicorn==0.49.0 python-multipart==0.0.32 \
    numpy==2.2.6 pillow==12.2.0

# Shared package (constants + model architectures), API code, baked-in weights.
COPY cityvision/ cityvision/
COPY backend/ backend/

# Pick the served model explicitly (no mlflow.db in the image to fall back on).
ENV CITYVISION_ARCH=ResNet50-UNet

EXPOSE 8000
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
