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
| POST   | `/predict`         | Image → predicted mask (PNG) |
| POST   | `/predict/classes` | Image → detected classes + pixel coverage (JSON) |

### `POST /predict`

Form field `file` (the image). Query params:

- `format`: `color` (default) · `overlay` · `raw`
  - `color` — colourised mask (one colour per class)
  - `overlay` — mask blended over the original image
  - `raw` — single-channel PNG of class indices (0–8)
- `alpha`: `0.0`–`1.0` overlay opacity (default `0.5`)

Returns `image/png` at the **original image resolution**.

## Examples

```bash
# colour mask
curl -X POST "http://localhost:8000/predict?format=color" \
     -F "file=@scene.png" -o mask.png

# overlay
curl -X POST "http://localhost:8000/predict?format=overlay&alpha=0.6" \
     -F "file=@scene.png" -o overlay.png

# detected classes as JSON
curl -X POST "http://localhost:8000/predict/classes" -F "file=@scene.png"
```

## Classes

`background, road, building, vegetation, sky, person, car, traffic_sign, bicycle`
