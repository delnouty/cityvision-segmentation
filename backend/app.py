"""
backend/app.py
==============

FastAPI service exposing the trained CityVision segmentation model.

Endpoints
---------
GET  /                 Minimal HTML page to upload an image and view the mask.
GET  /health           Service + loaded-model info (JSON).
POST /predict          Upload an image -> predicted mask as JSON.
                       Returns: {width, height, classes, palette,
                                 mask (HxW class indices), detected_classes}.
POST /predict/image    Upload an image -> predicted mask rendered as a PNG.
                       Query: format=color|overlay|raw (default color),
                              alpha=0.0..1.0 (overlay blend, default 0.5).
POST /predict/classes  Upload an image -> JSON list of detected classes
                       with pixel coverage (no per-pixel mask).

Run
---
    uvicorn backend.app:app --host 0.0.0.0 --port 8000
    # then open http://localhost:8000/
"""

import io
import os
import json
import time
import logging

from fastapi import FastAPI, File, UploadFile, Query, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from PIL import Image, UnidentifiedImageError

from backend.inference import Segmenter, CLASS_NAMES, PALETTE, NUM_CLASSES

# Reject uploads larger than this (guards against OOM from huge images).
MAX_UPLOAD_MB = float(os.environ.get("CITYVISION_MAX_UPLOAD_MB", "10"))
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)


class _JsonFormatter(logging.Formatter):
    """Emit one JSON object per log line (structured logs)."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        data = getattr(record, "data", None)
        if isinstance(data, dict):
            payload.update(data)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _make_logger() -> logging.Logger:
    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())
    lg = logging.getLogger("cityvision")
    lg.setLevel(logging.INFO)
    lg.handlers = [handler]
    lg.propagate = False
    return lg


log = _make_logger()

app = FastAPI(
    title="CityVision Segmentation API",
    description="Upload an urban scene image and receive the predicted segmentation mask.",
    version="1.0.0",
)


@app.middleware("http")
async def _limit_and_log(request: Request, call_next):
    # Fast-reject oversized uploads before buffering the whole body.
    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > MAX_UPLOAD_BYTES + 4096:
        log.warning(
            "upload_rejected",
            extra={"data": {"path": request.url.path, "content_length": int(cl)}},
        )
        return JSONResponse(
            status_code=413,
            content={"detail": f"Payload too large (limit {MAX_UPLOAD_MB:g} MB)."},
        )
    start = time.perf_counter()
    response = await call_next(request)
    log.info(
        "request",
        extra={
            "data": {
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "latency_ms": round((time.perf_counter() - start) * 1000, 1),
            }
        },
    )
    return response


# Load the model once at startup (heavy — do it lazily-once, not per request).
segmenter: Segmenter | None = None


@app.on_event("startup")
def _load_model() -> None:
    global segmenter
    segmenter = Segmenter()
    info = segmenter.info()
    log.info(
        "model_loaded",
        extra={
            "data": {
                "architecture": info["architecture"],
                "checkpoint": info["checkpoint"],
                "device": info["device"],
                "max_upload_mb": MAX_UPLOAD_MB,
            }
        },
    )


async def _read_upload(file: UploadFile) -> bytes:
    """Read the upload, enforcing the size cap (defense-in-depth vs the
    Content-Length check in the middleware, e.g. for chunked requests)."""
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413, detail=f"Payload too large (limit {MAX_UPLOAD_MB:g} MB)."
        )
    return data


def _read_image(data: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(data))
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400, detail="Uploaded file is not a valid image."
        )


def _png_response(image: Image.Image) -> StreamingResponse:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.get("/health")
def health() -> dict:
    if segmenter is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return {"status": "ok", **segmenter.info()}


@app.post("/predict")
async def predict(
    file: UploadFile = File(..., description="Image to segment."),
    encoding: str = Query("dense", pattern="^(dense|rle)$"),
) -> dict:
    """Take an image and return the predicted segmentation mask as JSON.

    The mask is a height x width grid of integer class indices (0..8). Use
    `classes` and `palette` for the index -> name / RGB-colour mapping.

    `encoding`:
      - `dense` (default): `mask` is the full 2-D grid (one int per pixel).
      - `rle`: `mask_rle.runs` is a compact list of [class_id, run_length]
        pairs (row-major) — much smaller for typical masks.
    """
    if segmenter is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    image = _read_image(await _read_upload(file))

    def _compute() -> dict:
        # CPU-bound (model inference + mask encoding) — runs in a worker thread
        # so the async event loop stays free to serve other requests.
        mask = segmenter.predict(image)
        height, width = mask.shape
        payload = {
            "model": segmenter.arch,
            "width": int(width),
            "height": int(height),
            "num_classes": NUM_CLASSES,
            "classes": CLASS_NAMES,
            "palette": [[int(c) for c in PALETTE[i]] for i in range(NUM_CLASSES)],
            "encoding": encoding,
            "detected_classes": segmenter.class_summary(mask),
        }
        if encoding == "rle":
            payload["mask_rle"] = {
                "shape": [int(height), int(width)],
                "order": "row-major",
                "runs": segmenter.rle_encode(mask),
            }
        else:
            payload["mask"] = mask.tolist()
        return payload

    return await run_in_threadpool(_compute)


@app.post("/predict/image")
async def predict_image(
    file: UploadFile = File(..., description="Image to segment."),
    format: str = Query("color", pattern="^(color|overlay|raw)$"),
    alpha: float = Query(0.5, ge=0.0, le=1.0),
):
    """Return the predicted mask rendered as a PNG (for visualisation)."""
    if segmenter is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    image = _read_image(await _read_upload(file))

    def _render() -> Image.Image:
        # CPU-bound (inference + rendering) — offloaded to a worker thread.
        mask = segmenter.predict(image)
        if format == "color":
            return segmenter.colorize(mask)
        if format == "overlay":
            return segmenter.overlay(image, mask, alpha=alpha)
        return Image.fromarray(mask, mode="L")  # raw class-index PNG (0..8)

    result = await run_in_threadpool(_render)
    return _png_response(result)


@app.post("/predict/classes")
async def predict_classes(file: UploadFile = File(...)) -> dict:
    """Return the detected classes and their pixel coverage as JSON."""
    if segmenter is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    image = _read_image(await _read_upload(file))

    def _compute() -> list:
        # CPU-bound (inference + summary) — offloaded to a worker thread.
        return segmenter.class_summary(segmenter.predict(image))

    detected = await run_in_threadpool(_compute)
    return {
        "model": segmenter.arch,
        "image_size": {"width": image.size[0], "height": image.size[1]},
        "detected_classes": detected,
    }


_INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>CityVision Segmentation</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 0; background: #0f1117; color: #e6e6e6; }
    .wrap { max-width: 960px; margin: 0 auto; padding: 24px; }
    h1 { font-size: 1.4rem; }
    .row { display: flex; gap: 16px; flex-wrap: wrap; margin-top: 16px; }
    .col { flex: 1 1 300px; }
    img { max-width: 100%; border-radius: 8px; border: 1px solid #2a2f3a; background:#000; }
    label { display:block; margin: 8px 0 4px; font-size: .9rem; color:#9aa4b2; }
    select, input[type=range] { width: 100%; }
    button { margin-top: 12px; padding: 10px 16px; border: 0; border-radius: 8px;
             background: #3b82f6; color: white; font-weight: 600; cursor: pointer; }
    button:disabled { opacity: .5; cursor: default; }
    .muted { color: #9aa4b2; font-size: .85rem; }
    .card { background:#161a22; border:1px solid #2a2f3a; border-radius:12px; padding:16px; }
    .legend { display:flex; flex-wrap:wrap; gap:10px 18px; margin-top:16px; }
    .legend .item { display:flex; align-items:center; gap:8px; font-size:.85rem; }
    .legend .swatch { width:16px; height:16px; border-radius:4px; border:1px solid #00000055;
                      flex:0 0 auto; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>CityVision — Urban Scene Segmentation</h1>
    <p class="muted" id="modelinfo">Loading model info…</p>
    <div class="card">
      <input id="file" type="file" accept="image/*"/>
      <label for="fmt">Output</label>
      <select id="fmt">
        <option value="overlay">Overlay (mask on image)</option>
        <option value="color">Colour mask</option>
        <option value="raw">Raw class indices</option>
      </select>
      <label for="alpha">Overlay opacity: <span id="alphaval">0.5</span></label>
      <input id="alpha" type="range" min="0" max="1" step="0.05" value="0.5"/>
      <button id="go" disabled>Segment</button>
    </div>
    <div class="row">
      <div class="col"><label>Input</label><img id="src"/></div>
      <div class="col"><label>Prediction</label><img id="out"/></div>
    </div>
    <label>Classes</label>
    <div class="legend" id="legend"></div>
  </div>
  <script>
    const $ = (id) => document.getElementById(id);
    fetch('/health').then(r => r.json()).then(d => {
      $('modelinfo').textContent =
        `Model: ${d.architecture} (${d.checkpoint}) · device: ${d.device}` +
        (d.val_mIoU ? ` · val mIoU ${d.val_mIoU.toFixed(4)}` : '');
      const legend = $('legend');
      d.classes.forEach((name, i) => {
        const [r, g, b] = d.palette[i];
        const item = document.createElement('div');
        item.className = 'item';
        item.innerHTML =
          `<span class="swatch" style="background:rgb(${r},${g},${b})"></span>${name}`;
        legend.appendChild(item);
      });
    });
    $('alpha').addEventListener('input', e => $('alphaval').textContent = e.target.value);
    $('file').addEventListener('change', e => {
      const f = e.target.files[0];
      $('go').disabled = !f;
      if (f) $('src').src = URL.createObjectURL(f);
    });
    $('go').addEventListener('click', async () => {
      const f = $('file').files[0];
      if (!f) return;
      $('go').disabled = true; $('go').textContent = 'Segmenting…';
      const fd = new FormData(); fd.append('file', f);
      const url = `/predict/image?format=${$('fmt').value}&alpha=${$('alpha').value}`;
      const r = await fetch(url, { method: 'POST', body: fd });
      if (r.ok) { const b = await r.blob(); $('out').src = URL.createObjectURL(b); }
      else { alert('Error: ' + (await r.text())); }
      $('go').disabled = false; $('go').textContent = 'Segment';
    });
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _INDEX_HTML
