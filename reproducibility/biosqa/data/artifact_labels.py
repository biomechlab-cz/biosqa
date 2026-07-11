"""Turn the store's ``artifact_type`` column (a ``|``-joined set of canonical
:data:`harmonize.ARTIFACT_TYPES`, or None) into a multi-hot target + a mask.

The mask is the per-sample supervision flag for the artifact-type head: True when
the record's dataset carries native/synthetic type labels, False otherwise (those
samples train the quality/binary heads but not the type head).
"""
from __future__ import annotations

import numpy as np

from .harmonize import ARTIFACT_TYPES

__all__ = ["to_multihot", "ARTIFACT_TYPES"]


def to_multihot(values, types: tuple[str, ...] = ARTIFACT_TYPES) -> tuple[np.ndarray, np.ndarray]:
    """``values`` = iterable of ``"a|b"`` strings / ``"clean"`` / None. Returns
    ``(Y [N, K] float32 multi-hot, mask [N] bool)``."""
    idx = {t: i for i, t in enumerate(types)}
    values = list(values)
    n, k = len(values), len(types)
    Y = np.zeros((n, k), dtype=np.float32)
    mask = np.zeros(n, dtype=bool)
    for i, v in enumerate(values):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        mask[i] = True
        for tok in str(v).split("|"):
            j = idx.get(tok)
            if j is not None:
                Y[i, j] = 1.0
    return Y, mask
