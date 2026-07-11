"""Apply a model card's normalization contract before inference (Plan 2 §7.1/§11).

This must reproduce Plan 1's training-time normalization *exactly*. Any
divergence silently degrades predictions (Plan 2 §14) -- so this module has
exactly one job and does not "helpfully" add alternative normalizations.
"""

from __future__ import annotations

import numpy as np

from biosqa.model.model_card import ModelCard, ModelCardError, Normalization


def normalize_window(window: np.ndarray, normalization: Normalization) -> np.ndarray:
    """Apply ``normalization`` to a single window of raw samples.

    Args:
        window: 1-D array of length ``L_m`` (raw units).
        normalization: the card's normalization spec.

    Returns:
        A ``float32`` array the same shape as ``window``.
    """
    window = np.asarray(window, dtype=np.float32)

    if normalization.method == "none":
        return window
    if normalization.method == "zscore":
        mean = normalization.mean if normalization.mean is not None else 0.0
        std = normalization.std if normalization.std is not None else 1.0
        if std == 0:
            raise ModelCardError("normalization.std is 0 -- would divide by zero")
        return (window - mean) / std
    if normalization.method == "minmax":
        lo = normalization.mean  # reused fields per simple card schema; TODO(Plan2 §11):
        hi = normalization.std  # promote to explicit min/max fields if minmax is adopted
        if lo is None or hi is None or hi == lo:
            raise ModelCardError("normalization method 'minmax' requires distinct mean/std bounds")
        return (window - lo) / (hi - lo)

    raise NotImplementedError(
        f"normalize_window: unsupported normalization method {normalization.method!r} "
        "(TODO Plan2 §7.1: extend as Plan 1 adds normalization schemes)"
    )


def make_windows(signal: np.ndarray, card: ModelCard, overlap: float = 0.0) -> np.ndarray:
    """Slice a full-length signal into ``L_m``-sample windows for sliding-window inference.

    Args:
        signal: 1-D raw-sample array for one channel.
        card: the modality's validated model card (supplies ``L_m``).
        overlap: fraction in ``[0, 1)`` of overlap between consecutive windows.

    Returns:
        2-D array ``[n_windows, L_m]``. The final partial window (if any) is
        dropped -- TODO(Plan2 §7.2): decide whether to zero-pad the tail
        window instead of dropping it once real recordings are exercised.
    """
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    signal = np.asarray(signal)
    step = max(1, int(round(card.l_m * (1.0 - overlap))))
    n_windows = max(0, (signal.shape[0] - card.l_m) // step + 1)
    if n_windows == 0:
        return np.empty((0, card.l_m), dtype=np.float32)
    return np.stack(
        [signal[i * step : i * step + card.l_m] for i in range(n_windows)]
    ).astype(np.float32)
