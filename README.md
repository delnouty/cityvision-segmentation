# CityVision — Urban Scene Semantic Segmentation

[![CI](https://github.com/delnouty/cityvision-segmentation/actions/workflows/ci.yml/badge.svg)](https://github.com/delnouty/cityvision-segmentation/actions/workflows/ci.yml)

Semantic segmentation of urban street scenes (Cityscapes), served through a
REST API and a web UI. The project covers the full pipeline: training several
CNN architectures, tracking them with MLflow, serving the best model via a
FastAPI prediction API, and presenting results in a Streamlit app.

**9 classes:** `background, road, building, vegetation, sky, person, car,
traffic_sign, bicycle`.

---

## Architecture

```
┌─────────────────────┐   HTTP (image → mask)   ┌──────────────────────────┐
│ Streamlit frontend  │ ──────────────────────► │ FastAPI backend          │
│  (data/samples/)    │ ◄────────────────────── │  (baked-in model weights)│
└─────────────────────┘     JSON / PNG mask     └──────────────────────────┘
        :8501                                              :8000
                        shared: cityvision/ (classes, palette, model defs)
```

- **Backend** — takes an image, returns the predicted segmentation mask (JSON or PNG).
- **Frontend** — lists sample images, calls the API, shows real image · real mask · predicted mask.
- **`cityvision/`** — single source of truth for the class taxonomy, colour palette, and model architectures (shared by training and serving).

---

## Results

Validation mIoU on Cityscapes (9-class), best checkpoint per architecture:

| Architecture      | Encoder                | Val mIoU |
|-------------------|------------------------|:--------:|
| **ResNet50-UNet** | ResNet50 (pretrained)  | **0.816** ⭐ |
| ResNet34-UNet     | ResNet34 (pretrained)  | 0.805 |
| VGG16-UNet        | VGG16 (pretrained)     | 0.798 |
| UNet              | from scratch           | 0.684 |
| SegNet            | from scratch           | _not trained_ |

The pretrained encoders clearly outperform training from scratch. The backend
serves the best model (**ResNet50-UNet**) by default. Metrics are tracked in
MLflow; light data augmentation (h-flip + colour jitter) is compared in
`notebooks/training_augmented.ipynb`.

---

## Repository layout

```
cityvision/     shared package — constants.py (palette/classes) + models.py (architectures)
src/            training scripts (UNet, VGG16-UNet, SegNet, ResNet34/50-UNet) + dataloader.py
backend/        FastAPI app (app.py), inference (inference.py), model/ (checkpoints)
frontend/       Streamlit UI (streamlit_app.py) + helpers (utils.py)
scripts/        make_samples.py (build data/samples), visualize_best.py (qualitative eval)
tests/          pytest suite (backend + frontend)
docker/         backend/frontend Dockerfiles
notebooks/      training_comparison.ipynb, training_augmented.ipynb
doc/            technical note
data/samples/   small committed sample set used by the frontend (full dataset is gitignored)
```

---

## Quick start (Docker — recommended)

Requires Docker Desktop running. From the project root:

```bash
docker compose up --build -d      # first run builds the images (downloads PyTorch)
```

Then open:
- **App (frontend):** http://localhost:8501
- **API docs (backend):** http://localhost:8000/docs

Stop with:

```bash
docker compose down
```

See [docker/](docker/) and [doc/docker_instructions.tex](doc/docker_instructions.tex) for details.

---

## Run locally (without Docker)

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows; use bin/activate on Linux/macOS
# PyTorch is installed separately (CPU wheels shown; see requirements.in for CUDA):
pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Two terminals from the project root:

```bash
uvicorn backend.app:app --port 8000        # terminal 1 — API
streamlit run frontend/streamlit_app.py     # terminal 2 — UI
```

---

## The prediction API

| Method | Path               | Description |
|--------|--------------------|-------------|
| GET    | `/health`          | Service + loaded-model info |
| GET    | `/docs`            | Swagger UI |
| POST   | `/predict`         | Image → predicted mask as **JSON** (`encoding=dense\|rle`) |
| POST   | `/predict/image`   | Image → predicted mask as **PNG** (`format=color\|overlay\|raw`) |
| POST   | `/predict/classes` | Image → detected classes + pixel coverage (JSON) |

Example:

```bash
curl -X POST "http://localhost:8000/predict" -F "file=@scene.png"          # JSON mask
curl -X POST "http://localhost:8000/predict/image?format=overlay" \
     -F "file=@scene.png" -o overlay.png                                    # PNG overlay
```

Config via env vars: `CITYVISION_ARCH` / `CITYVISION_CHECKPOINT` (which model to serve),
`CITYVISION_MAX_UPLOAD_MB` (upload size cap, default 10). Full reference: [backend/README.md](backend/README.md).

---

## Training

Each architecture has a script under `src/` (models come from `cityvision.models`):

```bash
python src/training.py            # UNet
python src/training_vgg.py        # VGG16-UNet
python src/training_segnet.py     # SegNet
python src/training_resnet.py     # ResNet34-UNet
python src/training_resnet50.py   # ResNet50-UNet
```

Runs are tracked in MLflow (`sqlite:///mlflow.db`, experiment `urban-segmentation`);
each saves its best checkpoint to `backend/model/`. View runs:

```bash
python run_mlflow.py              # MLflow UI
```

The notebooks (`notebooks/`) reproduce training with and without light data augmentation.
Qualitative check of the best model:

```bash
python scripts/visualize_best.py --split val
```

---

## Development

```bash
pytest                 # test suite
black .                # format
flake8                 # lint
pre-commit install     # run black + flake8 on each commit
```

CI (GitHub Actions) runs lint + tests on every push/PR ([.github/workflows/ci.yml](.github/workflows/ci.yml)).

---

## Notes

- The **full Cityscapes dataset is gitignored** (large). Only a small curated sample set
  (`data/samples/`, 2 image+mask pairs per city, built by `scripts/make_samples.py`) is
  committed so the frontend/demo works without it.
- Model checkpoints (`*.pth`) are gitignored; the backend loads them from `backend/model/`.
