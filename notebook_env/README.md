# notebook_env — scripts required to run the training notebooks

Copies of the only files the notebooks in [notebooks/](../notebooks/) need in
order to run. Nothing here is imported by the rest of the repo; the originals
are untouched.

**Repository:** https://github.com/delnouty/cityvision-segmentation
**This folder:** https://github.com/delnouty/cityvision-segmentation/tree/main/notebook_env

| File | GitHub |
|---|---|
| `requirements-notebooks.txt` | https://github.com/delnouty/cityvision-segmentation/blob/main/notebook_env/requirements-notebooks.txt |
| `src/dataloader.py` | https://github.com/delnouty/cityvision-segmentation/blob/main/notebook_env/src/dataloader.py |
| `notebooks/training_segnet.ipynb` | https://github.com/delnouty/cityvision-segmentation/blob/main/notebook_env/notebooks/training_segnet.ipynb |
| `notebooks/training_comparison.ipynb` | https://github.com/delnouty/cityvision-segmentation/blob/main/notebook_env/notebooks/training_comparison.ipynb |
| `notebooks/training_augmented.ipynb` | https://github.com/delnouty/cityvision-segmentation/blob/main/notebook_env/notebooks/training_augmented.ipynb |

Clone just this folder:

```powershell
git clone --filter=blob:none --sparse https://github.com/delnouty/cityvision-segmentation.git
cd cityvision-segmentation
git sparse-checkout set notebook_env
```

```
notebook_env/
  requirements-notebooks.txt   notebook-only deps + the Jupyter stack
  src/dataloader.py            copy of src/dataloader.py
  notebooks/
    training_segnet.ipynb      copies of the three training notebooks
    training_comparison.ipynb
    training_augmented.ipynb
```

## Why only one Python file

All three notebooks (`training_segnet.ipynb`, `training_comparison.ipynb`,
`training_augmented.ipynb`) import exactly one local module — `dataloader` —
and define their models, losses and metrics inline in their own cells. They do
not import `cityvision/`. `dataloader.py` itself imports only third-party
packages, so the dependency chain stops there.

## Why the `src/` subfolder name matters

Each notebook locates the project root by walking up from the working directory
until it finds a folder containing `src/`, then adds that `src/` to `sys.path`.
Keeping the copy at `notebook_env/src/dataloader.py` makes `notebook_env/` the
project root the notebooks resolve to, so they import `dataloader` from here —
not from the parent repo — with no edits to any cell.

Renaming `src/`, or moving the notebooks out of `notebooks/`, breaks that lookup.

## Install

PyTorch must be installed first and separately — it is accelerator-specific,
and the default index gives you the CPU build:

```powershell
pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-notebooks.txt
```

`requirements-notebooks.txt` adds `jupyterlab`, `ipykernel` and `ipywidgets`,
which the repo's top-level `requirements.txt` does not declare. `ipywidgets` is
required, not optional: every notebook does `from tqdm.notebook import tqdm`,
which fails without it.

Launch from this folder so the root resolves correctly:

```powershell
jupyter lab notebooks/
```

## Not included

The Cityscapes dataset and the `.pth` checkpoints are data, not scripts, so they
are not copied here. The notebooks read their paths from the `CONFIG` cell —
point `IMG_ROOT` / `MASK_ROOT` at wherever the dataset lives, and `EPOCHS` /
`BATCH_SIZE` are set in the same cell.

Because `PROJECT_ROOT` now resolves to `notebook_env/`, the notebooks will look
for the dataset at `notebook_env/data/...` and write checkpoints to
`notebook_env/backend/model/` by default. Either edit those paths in the `CONFIG`
cell to absolute paths, or symlink the real folders in.

## Keeping the copies fresh — `sync.ps1`

These notebooks are copies, so edits here do not propagate back to
[notebooks/](../notebooks/) and vice versa. In practice they drift every time a
notebook is re-run and saved in the IDE.

`sync.ps1` re-copies the four tracked files from the repository and verifies each
one by SHA256:

```powershell
cd notebook_env
.\sync.ps1            # copy whatever drifted, then verify
.\sync.ps1 -Check     # report only; exit code 1 if anything is stale
```

`-Check` writes nothing and exits non-zero when something is out of sync, so it
works as a pre-commit guard. The repository is always the source: the script
never writes back into `notebooks/` or `src/`.

Tracked files: the three notebooks and `src/dataloader.py`. If you add another
shared module, add it to the `$files` map at the top of the script.
