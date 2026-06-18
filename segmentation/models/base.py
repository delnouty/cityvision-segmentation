"""Base class shared by every segmentation model."""

import torch.nn as nn


class BaseSegModel(nn.Module):
    """Common behaviour for all architectures.

    ``encoder_modules`` lists the attribute names of submodules that form the
    (optionally pretrained) encoder. When non-empty, ``param_groups`` returns
    two optimizer groups so the pretrained encoder can be fine-tuned at a lower
    learning rate than the freshly initialised decoder. When empty, the whole
    network trains at a single learning rate.
    """

    encoder_modules: tuple = ()

    def param_groups(self, lr: float, encoder_lr_mult: float = 0.1):
        if not self.encoder_modules:
            return [{"params": list(self.parameters()), "lr": lr}]

        enc_params = []
        for name in self.encoder_modules:
            enc_params += list(getattr(self, name).parameters())
        enc_ids = {id(p) for p in enc_params}
        dec_params = [p for p in self.parameters() if id(p) not in enc_ids]
        return [
            {"params": enc_params, "lr": lr * encoder_lr_mult},
            {"params": dec_params, "lr": lr},
        ]
