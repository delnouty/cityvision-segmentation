# CityVision — Segmentation API

A FastAPI service that exposes the trained urban-scene segmentation model.
It receives an image and returns the predicted segmentation mask.

## Layout

```
backend/
├── app.py          # FastAPI app + endpoints + minimal upload UI
├── inference.py    # model selection, loading, preprocessing, prediction
└── model/          # trained checkpoints (*.pth) — already present
```

The API rebuilds the **same** model classes from `src/` that produced the
checkpoints and applies the **same** ImageNet preprocessing as
`src/dataloader.py`, so inference matches training.

## Install

```bash
pip install -r requirements.txt   # includes fastapi, uvicorn, python-multipart
```

## Run

From the **project root** (so the `backend` package imports correctly):

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Then open <http://localhost:8000/> for a simple upload page, or
<http://localhost:8000/docs> for the interactive Swagger UI.

## Which model is served?

On startup the API picks the model in this priority order:

1. `CITYVISION_CHECKPOINT` — explicit path to a `.pth` file
2. `CITYVISION_ARCH` — architecture name (`UNet`, `ResNet34-UNet`,
   `ResNet50-UNet`, `VGG16-UNet`, `SegNet`)
3. Best run recorded in `mlflow.db` (highest `best_val_mIoU` with an available
   checkpoint) — currently **ResNet50-UNet** (val mIoU ≈ 0.816)
4. First available checkpoint by quality priority

Examples:

```bash
# serve a specific architecture
CITYVISION_ARCH=VGG16-UNet uvicorn backend.app:app --port 8000

# serve a specific checkpoint file
CITYVISION_CHECKPOINT=backend/model/unet_best.pth uvicorn backend.app:app --port 8000
```

On Windows PowerShell:

```powershell
$env:CITYVISION_ARCH = "VGG16-UNet"; uvicorn backend.app:app --port 8000
```

## Endpoints

| Method | Path               | Description |
|--------|--------------------|-------------|
| GET    | `/`                | HTML upload page |
| GET    | `/health`          | Service + loaded-model info (JSON) |
| GET    | `/docs`            | Swagger UI |
| POST   | `/predict`         | Image → predicted mask as **JSON** |
| POST   | `/predict/image`   | Image → predicted mask rendered as a **PNG** |
| POST   | `/predict/classes` | Image → detected classes + pixel coverage (JSON) |

### `POST /predict` (JSON — the predicted mask)

Form field `file` (the image). Query param `encoding`: `dense` (default) or `rle`.

**`encoding=dense`** — `mask` is the full grid:

```jsonc
{
  "model": "ResNet50-UNet",
  "width": 2048,
  "height": 1024,
  "num_classes": 9,
  "classes": ["background", "road", ...],      // index -> name
  "palette": [[0,0,0], [128,64,128], ...],      // index -> RGB
  "encoding": "dense",
  "mask": [[0, 0, 1, ...], ...],                // height x width class indices
  "detected_classes": [ {"class_name": "road", "percentage": 39.1, ...}, ... ]
}
```

The **`mask`** is the predicted segmentation: a 2-D grid (height × width) of
integer class indices (0–8), at the original image resolution.

**`encoding=rle`** — compact run-length encoding (much smaller; recommended for
large images). Instead of `mask`, you get `mask_rle`:

```jsonc
{
  ...,
  "encoding": "rle",
  "mask_rle": {
    "shape": [1024, 2048],            // [height, width]
    "order": "row-major",
    "runs": [[0, 1500], [1, 320], ...]  // [class_id, run_length] pairs
  }
}
```

Reconstruct it with NumPy:

```python
import numpy as np
runs = resp["mask_rle"]["runs"]
h, w = resp["mask_rle"]["shape"]
mask = np.concatenate([np.full(n, c) for c, n in runs]).reshape(h, w)
```

Use `classes` / `palette` to map an index to its name / colour.

### `POST /predict/image` (PNG — for visualisation)

Form field `file`. Query params:

- `format`: `color` (default) · `overlay` · `raw`
  - `color` — colourised mask (one colour per class)
  - `overlay` — mask blended over the original image
  - `raw` — single-channel PNG of class indices (0–8)
- `alpha`: `0.0`–`1.0` overlay opacity (default `0.5`)

Returns `image/png` at the **original image resolution**.

## Examples

```bash
# predicted mask as JSON
curl -X POST "http://localhost:8000/predict" -F "file=@scene.png"

# predicted mask as a PNG image
curl -X POST "http://localhost:8000/predict/image?format=color" \
     -F "file=@scene.png" -o mask.png

# overlay PNG
curl -X POST "http://localhost:8000/predict/image?format=overlay&alpha=0.6" \
     -F "file=@scene.png" -o overlay.png

# detected classes only (lightweight JSON, no per-pixel mask)
curl -X POST "http://localhost:8000/predict/classes" -F "file=@scene.png"
```

## Classes

`background, road, building, vegetation, sky, person, car, traffic_sign, bicycle`
