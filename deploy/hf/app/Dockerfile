# HF Space (Docker) — CityVision demo UI (Streamlit) that consumes the API.
# Build context = project root:  docker build -f deploy/hf/app/Dockerfile -t cityvision-app .
# No torch — this Space only runs the UI and calls the deployed API over HTTP.
FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    streamlit==1.58.0 requests==2.34.2 numpy==2.2.6 pillow==12.2.0

# Shared (torch-free) constants + UI code + bundled sample images.
COPY cityvision/ cityvision/
COPY frontend/ frontend/
COPY data/samples/ data/samples/

# URL of the deployed API Space. EDIT this to match your API Space's URL.
ENV CITYVISION_API=https://daryael-cityvision-api.hf.space

EXPOSE 7860
CMD ["bash", "-c", "streamlit run frontend/streamlit_app.py --server.address=0.0.0.0 --server.port=7860 --server.headless=true --browser.gatherUsageStats=false"]
