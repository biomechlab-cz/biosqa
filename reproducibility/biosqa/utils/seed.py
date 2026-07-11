"""Deterministic seeding (Plan 1 §1.5 — every claim reproducible)."""
from __future__ import annotations

import os
import random

import numpy as np

__all__ = ["seed_everything", "worker_init_fn"]


def seed_everything(seed: int = 0, *, deterministic: bool = True) -> int:
    """Seed python/numpy/torch RNGs. Returns the seed for logging.

    ``deterministic=True`` sets cuDNN to deterministic mode (slower but exact);
    keep it on for confirmation runs, optionally off for large pretraining.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # opt-in exact algorithms where available; warn_only avoids hard
            # failures on ops without a deterministic impl.
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
        else:
            torch.backends.cudnn.benchmark = True
    except ImportError:
        pass
    return seed


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker seeding for reproducible augmentation."""
    base = np.random.get_state()[1][0]
    np.random.seed((int(base) + worker_id) % (2**32 - 1))
    random.seed((int(base) + worker_id) % (2**32 - 1))
