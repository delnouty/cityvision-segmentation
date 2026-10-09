# HF Space (Docker) — CityVision PoC dashboard (Streamlit) with the three models
# running in-process on CPU: ResNet50-UNet (baseline), Mask2Former, EoMT.
# The Space repo mirrors the project layout; fill it with deploy/hf/poc/stage.ps1.
# Local test, from the staged folder:  docker build -t cityvision-poc . && docker run -p 7860:7860 cityvision-poc
FROM python:3.10-slim

# HF_HUB_OFFLINE: models are built from the bundled configs
# (cityvision/hf_configs) and the baked-in checkpoints, never from the Hub.
ENV PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/tmp/hf

# OpenCV (headless) needs GLib at runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU PyTorch (the free/basic Space hardware has no GPU)
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    torch==2.12.0 torchvision==0.27.0

# Dashboard + model runtime (versions = local PoC environment)
RUN pip install --no-cache-dir \
    transformers==5.17.0 scipy==1.15.3 \
    streamlit==1.58.0 altair==6.2.2 pandas==2.3.3 \
    numpy==2.2.6 pillow==12.2.0 opencv-python-headless==5.0.0.93 tqdm==4.68.2

# Code, bundled data (EDA stats, 30 sample images, results) and the 3 checkpoints
COPY cityvision/ cityvision/
COPY src/ src/
COPY scripts/evaluate_poc.py scripts/
COPY poc_app/ poc_app/
COPY backend/model/resnet50_best.pth backend/model/
COPY model_backups/mask2former_swint_best.pth model_backups/eomt_base_best.pth model_backups/

# HF routes public traffic to app_port 7860
EXPOSE 7860
CMD ["bash", "-c", "streamlit run poc_app/app.py --server.address=0.0.0.0 --server.port=7860 --server.headless=true --browser.gatherUsageStats=false"]
