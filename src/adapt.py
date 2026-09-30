"""Label-free calibration for a new athlete, using only their UNLABELED day-1 windows.

- input_norm: standardize inputs with the athlete's own per-channel mean/SD instead of the team's.
- adabn:      recompute every BatchNorm layer's running mean/variance on the athlete's windows
              (Li et al., "Adaptive Batch Normalization"); weights unchanged, no gradients, no labels.
Both only need forward passes, so they could run on the device itself.
"""
import copy

import numpy as np
import torch
from torch import nn


def athlete_stats(X):
    return X.mean(axis=(0, 2), keepdims=True), X.std(axis=(0, 2), keepdims=True) + 1e-6


def adabn(model, mu, sd, X_unlabeled):
    """Return a copy of `model` whose BatchNorm statistics are recomputed on X_unlabeled."""
    m = copy.deepcopy(model)
    bns = [mod for mod in m.modules() if isinstance(mod, nn.BatchNorm1d)]
    if not bns:
        return m
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None  # cumulative average; one full-batch pass gives the exact statistics
    m.train()
    x = torch.from_numpy(((X_unlabeled - mu) / sd).astype(np.float32))
    with torch.no_grad():
        m(x)
    m.eval()
    return m


def has_batchnorm(model):
    return any(isinstance(mod, nn.BatchNorm1d) for mod in model.modules())
