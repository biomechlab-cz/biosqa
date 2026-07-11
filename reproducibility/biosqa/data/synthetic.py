"""Synthetic SQA signals — for end-to-end smoke tests and CI (no real data needed).

Generates a *learnable* 4-level quality task: cleaner classes (higher Q) have a
stronger quasi-periodic component and less broadband noise, so a working model
reaches well above chance. Used by the Phase-0 dummy run and unit tests to prove
the config->model->train->eval->export pipeline end to end.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import TensorDataset

__all__ = ["make_synthetic_sqa", "synthetic_datasets"]


def make_synthetic_sqa(
    n: int = 1024,
    length: int = 1024,
    n_classes: int = 4,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(X [n,1,length] float32, y [n] int)``. Higher class = cleaner."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, size=n)
    t = np.linspace(0, 8 * np.pi, length)
    X = np.zeros((n, 1, length), dtype=np.float32)
    for i in range(n):
        c = y[i]
        snr = (c + 1) / n_classes                        # cleaner for higher c
        freq = 1.0 + 0.5 * c
        phase = rng.uniform(0, 2 * np.pi)
        clean = np.sin(freq * t + phase) + 0.3 * np.sin(2 * freq * t + phase)
        noise = rng.standard_normal(length) * (1.0 - snr) * 1.5
        baseline = 0.5 * (1 - snr) * np.sin(0.1 * t + rng.uniform(0, np.pi))  # wander
        X[i, 0] = (snr * clean + noise + baseline).astype(np.float32)
    return X, y.astype(np.int64)


def synthetic_datasets(
    length: int = 1024, n_classes: int = 4, n_train: int = 1024, n_val: int = 256, n_test: int = 256, seed: int = 0
):
    Xtr, ytr = make_synthetic_sqa(n_train, length, n_classes, seed)
    Xva, yva = make_synthetic_sqa(n_val, length, n_classes, seed + 1)
    Xte, yte = make_synthetic_sqa(n_test, length, n_classes, seed + 2)
    to = lambda X, y: TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return to(Xtr, ytr), to(Xva, yva), to(Xte, yte)
