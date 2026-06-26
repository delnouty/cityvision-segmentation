# Frontend image: Streamlit UI + bundled sample images. No torch (lean).
FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    streamlit==1.58.0 requests==2.34.2 numpy==2.2.6 pillow==12.2.0

# UI code + the committed sample set (data/samples/).
COPY frontend/ frontend/
COPY data/samples/ data/samples/

# Backend URL — overridden by docker-compose / Azure to point at the API.
ENV CITYVISION_API=http://backend:8000

EXPOSE 8501
CMD ["streamlit", "run", "frontend/streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
