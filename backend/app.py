"""
backend/app.py
==============

FastAPI service exposing the trained CityVision segmentation model.

Endpoints
---------
GET  /                 Minimal HTML page to upload an image and view the mask.
GET  /health           Service + loaded-model info (JSON).
POST /predict          Upload an image -> predicted mask.
                       Query: format=color|overlay|raw (default color),
                              alpha=0.0..1.0 (overlay blend, default 0.5).
                       Returns: image/png.
POST /predict/classes  Upload an image -> JSON list of detected classes
                       with pixel coverage.

Run
---
    uvicorn backend.app:app --host 0.0.0.0 --port 8000
    # then open http://localhost:8000/
"""

import io

from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from PIL import Image, UnidentifiedImageError

from backend.inference import Segmenter

app = FastAPI(
    title="CityVision Segmentation API",
    description="Upload an urban scene image and receive the predicted segmentation mask.",
    version="1.0.0",
)

# Load the model once at startup (heavy — do it lazily-once, not per request).
segmenter: Segmenter | None = None


@app.on_event("startup")
def _load_model() -> None:
    global segmenter
    segmenter = Segmenter()
    info = segmenter.info()
    print(f"[CityVision] Loaded {info['architecture']} "
          f"({info['checkpoint']}) on {info['device']}.")


def _read_image(data: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(data))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")


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
    format: str = Query("color", pattern="^(color|overlay|raw)$"),
    alpha: float = Query(0.5, ge=0.0, le=1.0),
):
    """Return the predicted mask as a PNG."""
    if segmenter is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    image = _read_image(await file.read())
    mask = segmenter.predict(image)

    if format == "color":
        result = segmenter.colorize(mask)
    elif format == "overlay":
        result = segmenter.overlay(image, mask, alpha=alpha)
    else:  # raw: single-channel class-index PNG (0..8)
        result = Image.fromarray(mask, mode="L")

    return _png_response(result)


@app.post("/predict/classes")
async def predict_classes(file: UploadFile = File(...)) -> dict:
    """Return the detected classes and their pixel coverage as JSON."""
    if segmenter is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    image = _read_image(await file.read())
    mask = segmenter.predict(image)
    return {
        "model": segmenter.arch,
        "image_size": {"width": image.size[0], "height": image.size[1]},
        "detected_classes": segmenter.class_summary(mask),
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
      const url = `/predict?format=${$('fmt').value}&alpha=${$('alpha').value}`;
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
