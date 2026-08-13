# CityVision — Urban Scene Semantic Segmentation

[![CI](https://github.com/delnouty/cityvision-segmentation/actions/workflows/ci.yml/badge.svg)](https://github.com/delnouty/cityvision-segmentation/actions/workflows/ci.yml)

Semantic segmentation of urban street scenes (Cityscapes), served through a
REST API and a web UI. The project covers the full pipeline: training several
CNN architectures, tracking them with MLflow, serving the best model via a
FastAPI prediction API, and presenting results in a Streamlit app.

**9 classes:** `background, road, building, vegetation, sky, person, car,
traffic_sign, bicycle`.

## 🚀 Live demo (Hugging Face Spaces)

- **App (Streamlit):** https://huggingface.co/spaces/DaryaEL/cityvision
- **Prediction API (FastAPI):** https://huggingface.co/spaces/DaryaEL/cityvision-api — interactive docs at [`/docs`](https://daryael-cityvision-api.hf.space/docs)

The app consumes the API. Deployment recipe: [`deploy/hf/`](deploy/hf/) — full
step-by-step guide in [`deploy/hf/DEPLOYMENT.md`](deploy/hf/DEPLOYMENT.md)
([PDF](deploy/hf/DEPLOYMENT.pdf)).

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
| ResNet34-UNet     | ResNet34 (pretrained)  | 0.809 |
| VGG16-UNet        | VGG16 (pretrained)     | 0.798 |
| SegNet            | from scratch           | 0.718 |
| UNet              | from scratch           | 0.684 |

The pretrained encoders clearly outperform training from scratch. The backend
serves the best model (**ResNet50-UNet**) by default. Metrics are tracked in
MLflow; light data augmentation (h-flip + colour jitter) is compared in
`notebooks/training_augmented.ipynb` (UNet + augmentation: 0.682).

> **Not comparable to published Cityscapes benchmarks.** Those score 19 classes
> at full resolution; this project remaps to 9 classes at 512×1024, which is a
> substantially easier task. The numbers above are only meaningful *relative to
> each other*.

---

## Repository layout

```
cityvision/     shared package — constants.py (palette/classes) + models.py (architectures)
src/            training scripts (UNet, VGG16-UNet, SegNet, ResNet34/50-UNet) + dataloader.py
backend/        FastAPI app (app.py), inference (inference.py), model/ (checkpoints)
frontend/       Streamlit UI (streamlit_app.py) + helpers (utils.py)
scripts/        make_samples.py (build data/samples), visualize_best.py (qualitative eval)
tests/          pytest suite (backend + frontend + model/notebook contract)
docker/         backend/frontend Dockerfiles
notebooks/      training_comparison.ipynb, training_augmented.ipynb, training_segnet.ipynb
notebook_env/   self-contained bundle to run the notebooks (see its README)
doc/            technical note
data/samples/   small committed sample set used by the frontend (full dataset is gitignored)
```

---

## Step 1 — get the model weights (required)

Checkpoints are **not** in the repository (`*.pth` is gitignored), so a fresh
clone has none and the backend refuses to start with
`RuntimeError: No model checkpoints found`. Download the served model from the
release — it is attached there, so it does not inflate the repository:

```bash
mkdir -p backend/model
curl -L -o backend/model/resnet50_best.pth \
  https://github.com/delnouty/cityvision-segmentation/releases/download/v2.2.0/resnet50_best.pth
```

```powershell
# Windows PowerShell
New-Item -ItemType Directory -Force backend\model | Out-Null
curl -L -o backend\model\resnet50_best.pth `
  https://github.com/delnouty/cityvision-segmentation/releases/download/v2.2.0/resnet50_best.pth
```

That is 156 MiB and the only checkpoint needed to run the API — it serves
**ResNet50-UNet** by default (`CITYVISION_ARCH=ResNet50-UNet`). All releases:
[releases](https://github.com/delnouty/cityvision-segmentation/releases).

---

## Step 2 — Quick start (Docker — recommended)

Requires Docker Desktop running, and the weight from step 1 in place. From the
project root:

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

Needs the weight from [step 1](#step-1--get-the-model-weights-required) as well —
the API loads it at startup either way.

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

> **Project requirement — "a Flask/FastAPI API, deployed on the Cloud, that takes an
> image and returns the predicted mask":** fulfilled by this **FastAPI** service
> ([`backend/app.py`](backend/app.py)), deployed on **Hugging Face Spaces** (see
> [Live demo](#-live-demo-hugging-face-spaces)). The **image → predicted mask**
> contract is `POST /predict` (mask as JSON) and `POST /predict/image` (mask as PNG)
> — the core inference is `segmenter.predict(image)` in
> [`backend/app.py`](backend/app.py).

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

### Get the dataset first

Training needs the full Cityscapes set, which is **not** in the repository. It
requires a free account at [cityscapes-dataset.com](https://www.cityscapes-dataset.com/downloads/);
download `leftImg8bit_trainvaltest.zip` (11 GB) and `gtFine_trainvaltest.zip`
(241 MB), then unpack them into this exact layout — the paths are hardcoded in
the training scripts' `img_root` / `mask_root` and in the notebooks' `CONFIG` cell:

```
data/cityscapes/
├── P8_Cityscapes_leftImg8bit_trainvaltest/
│   └── leftImg8bit/{train,val,test}/<city>/*_leftImg8bit.png
└── P8_Cityscapes_gtFine_trainvaltest/
    └── gtFine/{train,val,test}/<city>/*_gtFine_labelIds.png
```

The `P8_…` prefixes are part of the expected path, not decoration. Verify with:

```bash
python src/dataloader.py
```

Expect `Train batches: 744 | Val batches: 125` at the default batch size of 4 —
that is 2975 train and 500 val pairs. Any `WARNING: mask not found` line means an
image has no matching mask, usually a half-finished unpack.

This check takes a couple of minutes: its `__main__` uses `balanced=True`, which
opens all 2975 masks to compute the class-balanced sampling weights.

### Run a training

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

### Notebooks

`notebooks/` reproduces the same training interactively: architecture comparison,
light data augmentation, and a dedicated SegNet run that documents the data
generator (streaming, class-balanced sampling, dataloader throughput).

They need the Jupyter stack, which `requirements.txt` does **not** declare — every
notebook does `from tqdm.notebook import tqdm`, which fails without `ipywidgets`.
Either install it directly, or use the self-contained bundle:

```bash
pip install -r notebook_env/requirements-notebooks.txt
jupyter lab notebooks/
```

[`notebook_env/`](notebook_env/) holds copies of the notebooks plus the only
module they import, so the folder runs on its own — see
[its README](notebook_env/README.md). Keep the copies fresh with
`notebook_env/sync.ps1`.

Each notebook checks its GPU budget before training (peak memory, competing GPU
processes) and verifies that its inline architectures still match
`cityvision.models`, so a checkpoint trained there stays loadable by the API.

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
  committed so the frontend/demo works without it. To train, fetch the dataset as
  described under [Training](#training).
- Model checkpoints (`*.pth`) are gitignored; the backend loads them from `backend/model/`.
  Get the served one from the [latest release](https://github.com/delnouty/cityvision-segmentation/releases)
  (see [step 1](#step-1--get-the-model-weights-required)) — release assets do not count
  towards the repository size.
- Which checkpoint gets served is decided by `select_checkpoint()` in
  [backend/inference.py](backend/inference.py), in this order: `CITYVISION_CHECKPOINT`
  (explicit path) → `CITYVISION_ARCH` (architecture name) → best run recorded in
  `mlflow.db` → first available by quality priority. The deployed API pins
  `CITYVISION_ARCH=ResNet50-UNet`, so a new local training run cannot silently change
  what production serves.
