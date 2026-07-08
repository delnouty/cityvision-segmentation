# Hugging Face Space (Docker SDK) — backend + frontend in ONE container.
# HF exposes a single port (app_port in README frontmatter, default 7860):
# Streamlit listens there; the FastAPI backend runs internally on :8000 and the
# frontend reaches it via localhost.
#
# Build context is the project root:
#   docker build -f deploy/hf/Dockerfile -t cityvision-hf .
FROM python:3.10-slim

WORKDIR /app

# CPU PyTorch (deps from PyPI).
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    torch==2.12.0 torchvision==0.27.0

# Backend + frontend runtime deps.
RUN pip install --no-cache-dir \
    fastapi==0.136.3 uvicorn==0.49.0 python-multipart==0.0.32 \
    numpy==2.2.6 pillow==12.2.0 streamlit==1.58.0 requests==2.34.2

# Shared package + both services + sample images + the model weight.
# NOTE: backend/model/<arch>_best.pth must be present (commit it via git-lfs
# in the Space repo — see the deploy steps).
COPY cityvision/ cityvision/
COPY backend/ backend/
COPY frontend/ frontend/
COPY data/samples/ data/samples/

ENV CITYVISION_ARCH=ResNet50-UNet
ENV CITYVISION_API=http://localhost:8000

EXPOSE 7860

# Start the API (internal :8000), then the Streamlit UI on the HF port.
CMD ["bash", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port 8000 & streamlit run frontend/streamlit_app.py --server.address=0.0.0.0 --server.port=7860 --server.headless=true --browser.gatherUsageStats=false"]
