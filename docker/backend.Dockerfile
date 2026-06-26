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

# Runtime deps. mlflow + tqdm are needed because the model classes live in the
# training modules (src/training_*.py), which import them at module load.
RUN pip install --no-cache-dir \
    fastapi==0.136.3 uvicorn==0.49.0 python-multipart==0.0.32 \
    numpy==2.2.6 pillow==12.2.0 tqdm==4.68.2 mlflow==2.22.5

# Code + model classes + baked-in weights (backend/model/*.pth).
COPY backend/ backend/
COPY src/ src/

# Pick the served model explicitly (no mlflow.db in the image to fall back on).
ENV CITYVISION_ARCH=ResNet50-UNet

EXPOSE 8000
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
