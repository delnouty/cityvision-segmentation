
# segmentation — configurable training pipeline

One training engine, loss, metric and data pipeline shared across every<>
architecture. You **configure the architecture of the solution** (and its
hyperparameters) instead of maintaining a separate script per model.

This replaces the five near-duplicate scripts in `src/` (`training.py`,
`training_resnet.py`, `training_resnet50.py`, `training_segnet.py`,
`training_vgg.py`) with a single parameterised package.

## How to run

All commands are run **from the project root**
(`c:\Users\DFILATOVA\Desktop\CityVision`) so that `python -m segmentation.train`
resolves the package.

### 1. Prerequisites

- The Cityscapes data must be present in the expected layout (same one the
  `src/` scripts use):
  ```
  data/cityscapes/P8_Cityscapes_leftImg8bit_trainvaltest/leftImg8bit/{train,val}/...
  data/cityscapes/P8_Cityscapes_gtFine_trainvaltest/gtFine/{train,val}/...
  ```
- Dependencies are already installed in the project's `.venv`
  (torch, torchvision, mlflow, tqdm, numpy, pillow).

### 2. Activate the virtual environment (PowerShell)

```powershell
.venv\Scripts\Activate.ps1
```

Alternatively, skip activation and prefix each command with
`.venv\Scripts\python.exe` instead of `python`.

### 3. Train a model

Pick an architecture with `--arch` (`unet`, `resnet34`, `resnet50`, `segnet`,
`vgg`):

```powershell
python -m segmentation.train --arch resnet50
```

Each architecture has a preset (carried over from the original scripts). Any
flag overrides that preset:

```powershell
python -m segmentation.train --arch unet --epochs 30 --batch-size 4 --lr 5e-5
python -m segmentation.train --arch resnet34 --no-pretrained
python -m segmentation.train --arch segnet --scheduler none --patience 10
```

Quick CPU smoke test (no GPU needed, proves the pipeline works end-to-end):

```powershell
python -m segmentation.train --arch unet --epochs 1 --batch-size 1 --img-size 256x512
```

List every flag:

```powershell
python -m segmentation.train --help
```

The script auto-detects CUDA and prints the device it uses; on CPU it runs but
is slow.

### 4. View results in MLflow

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open <http://127.0.0.1:5000> — the run appears under the
**urban-segmentation** experiment, and the registered model under **Models** as
`cityvision-segmentation-<arch>`.

## Available architectures

| `--arch`   | Model            | Encoder      | Preset (epochs / batch / lr) |
|------------|------------------|--------------|------------------------------|
| `unet`     | U-Net            | from scratch | 60 / 8 / 1e-4                |
| `resnet34` | ResNet34 + U-Net | pretrained   | 20 / 4 / 1e-4                |
| `resnet50` | ResNet50 + U-Net | pretrained   | 20 / 4 / 1e-4                |
| `segnet`   | SegNet           | from scratch | 20 / 4 / 1e-3 (+plateau LR)  |
| `vgg`      | VGG16 + U-Net    | pretrained   | 20 / 4 / 1e-4                |

## Common flags

| Flag                  | Meaning                                              |
|-----------------------|------------------------------------------------------|
| `--epochs N`          | Number of epochs                                     |
| `--batch-size N`      | Batch size                                           |
| `--lr F`              | Base (decoder) learning rate                         |
| `--encoder-lr-mult F` | Encoder LR = `lr * F` (pretrained models)            |
| `--patience N`        | Early-stopping patience on val mIoU (`0` disables)   |
| `--dropout F`         | Dropout (U-Net only)                                 |
| `--img-size HxW`      | Input resolution, e.g. `512x1024`                    |
| `--scheduler {none,plateau}` | LR scheduler                                  |
| `--pretrained` / `--no-pretrained` | Toggle ImageNet encoder weights         |
| `--balanced` / `--no-balanced`     | Toggle class-balanced sampler           |
| `--checkpoint-name F` | Output filename under `backend/model/`               |
| `--run-name` / `--experiment` | MLflow naming                                |
| `--registered-model-name F`   | MLflow Model Registry name                   |

## MLflow tracking & registration

Every run is tracked to `sqlite:///mlflow.db` (experiment `urban-segmentation`):
params, per-epoch metrics, per-class IoU, and the best checkpoint artifact.

In addition, at the end of each run the **best** model is registered in the
**MLflow Model Registry** as a new version under
`cityvision-segmentation-<arch>` (override with `--registered-model-name`). The
SQLite-backed store is required for the registry — the file store does not
support it, which is already the case here.

Best checkpoints are also written to `backend/model/`.

## Package layout

```
segmentation/
├── train.py        # CLI entry point  (python -m segmentation.train)
├── config.py       # TrainConfig dataclass + per-architecture presets
├── engine.py       # shared train / validate / fit loop + MLflow logging
├── losses.py       # weighted CE + Dice
├── metrics.py      # confusion-matrix IoU / pixel accuracy
├── data.py         # class remap + bridge to src/dataloader.py
└── models/
    ├── base.py         # BaseSegModel (encoder/decoder param groups)
    ├── unet.py
    ├── resnet_unet.py  # ResNet34 & ResNet50
    ├── segnet.py
    ├── vgg_unet.py
    └── __init__.py     # ARCHITECTURES registry + build_model()
```

### Adding a new architecture

1. Add a model class in `models/` subclassing `BaseSegModel` (set
   `encoder_modules` if it has a pretrained encoder to fine-tune).
2. Register it in `models/__init__.py` (`ARCHITECTURES`).
3. Add a preset in `config.py` (`ARCH_PRESETS`).

It is then selectable via `--arch <name>` with no changes to the engine.
