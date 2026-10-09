"""
scripts/export_hf_configs.py
============================

Saves the Hugging Face configurations of the two PoC transformers, already
set to the project's 9 classes, to ``cityvision/hf_configs/``.

Why: to load a *trained* checkpoint, only the architecture is needed. With
these small JSON files bundled in the repo, ``build_model_from_config()`` in
``src/training_mask2former.py`` / ``src/training_eomt.py`` creates the model
without downloading the pretrained weights from the Hub (which the checkpoint
overwrites anyway). That keeps evaluation, the PoC app and a future cloud
image self-contained and faster to start.

Run once (needs Hub access), or again if PRETRAINED changes in a training script:
    python scripts/export_hf_configs.py
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from transformers import AutoConfig  # noqa: E402

import training_eomt  # noqa: E402
import training_mask2former  # noqa: E402
from cityvision.constants import CLASS_NAMES  # noqa: E402

TARGETS = {
    training_mask2former.CONFIG_PATH: training_mask2former.PRETRAINED,
    training_eomt.CONFIG_PATH: training_eomt.PRETRAINED,
}


def main():
    for path, repo in TARGETS.items():
        config = AutoConfig.from_pretrained(
            repo,
            id2label=dict(enumerate(CLASS_NAMES)),
            label2id={n: i for i, n in enumerate(CLASS_NAMES)},
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        config.to_json_file(path)
        print(f"{repo} -> {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
