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


class ShortRecordError(ValueError):
    """The signal is shorter than ONE model window -- not a single sample can be graded.

    Raised instead of returning an empty window array: zero windows means zero segments, which
    the UI renders as "N segments = 0" and a user reads as "no quality problems were found"
    rather than "nothing was analyzed" (Plan 2 §14 -- a silent empty state is a false clean
    bill of health). Carries the facts a caller needs to say so out loud; ``ValueError`` so
    existing broad handlers still degrade safely.
    """

    def __init__(self, n_samples: int, card: ModelCard) -> None:
        self.n_samples = int(n_samples)
        self.l_m = int(card.l_m)
        self.fs_hz = float(card.fs_hz)
        self.modality = card.modality
        self.record_sec = self.n_samples / self.fs_hz if self.fs_hz else 0.0
        self.required_sec = self.l_m / self.fs_hz if self.fs_hz else 0.0
        super().__init__(
            f"record is {self.record_sec:.1f} s ({self.n_samples} samples) but the "
            f"{self.modality} model needs {self.required_sec:.1f} s ({self.l_m} samples) "
            "for one window -- nothing was analyzed"
        )


def window_starts(
    n_samples: int, card: ModelCard, overlap: float = 0.0, *, cover_tail: bool = True
) -> np.ndarray:
    """First-sample index of every window :func:`make_windows` produces, for ``n_samples`` samples.

    The uniform grid is ``i * step``; when the signal does not tile evenly the final start is
    END-ANCHORED at ``n_samples - L_m`` (it overlaps its predecessor) so that every sample of the
    record is covered by at least one graded window. A caller that needs exact window TIMES must
    use these starts rather than assuming the uniform grid -- the last one is not on it.

    Raises:
        ShortRecordError: if ``n_samples < L_m`` (no window exists).
    """
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    n = int(n_samples)
    if n < card.l_m:
        raise ShortRecordError(n, card)
    step = max(1, int(round(card.l_m * (1.0 - overlap))))
    starts = list(range(0, n - card.l_m + 1, step))
    if cover_tail and starts[-1] + card.l_m < n:
        starts.append(n - card.l_m)
    return np.asarray(starts, dtype=np.int64)


def ungraded_tail_samples(
    n_samples: int, card: ModelCard, overlap: float = 0.0, *, cover_tail: bool = True
) -> int:
    """Samples past the end of the last window -- signal that no window covers, hence ungraded.

    Always 0 under the default ``cover_tail=True``; a caller that opts out gets the count so it
    can mark the tail as not-analyzed instead of leaving it invisible.
    """
    starts = window_starts(n_samples, card, overlap, cover_tail=cover_tail)
    return max(0, int(n_samples) - (int(starts[-1]) + int(card.l_m)))


def make_windows(
    signal: np.ndarray,
    card: ModelCard,
    overlap: float = 0.0,
    *,
    cover_tail: bool = True,
    return_starts: bool = False,
):
    """Slice a full-length signal into ``L_m``-sample windows for sliding-window inference.

    Args:
        signal: 1-D raw-sample array for one channel.
        card: the modality's validated model card (supplies ``L_m``).
        overlap: fraction in ``[0, 1)`` of overlap between consecutive windows.
        cover_tail: emit a final END-ANCHORED window covering the trailing partial window, so no
            sample goes ungraded (the default). Set ``False`` for the legacy drop-the-tail grid,
            and then use :func:`ungraded_tail_samples` to report what was not analyzed.
        return_starts: also return the per-window start indices (:func:`window_starts`).

    Returns:
        2-D array ``[n_windows, L_m]``, or ``(windows, starts)`` when ``return_starts``.

    Raises:
        ShortRecordError: if the signal is shorter than one window. It is never silently empty.
    """
    signal = np.asarray(signal)
    starts = window_starts(signal.shape[0], card, overlap, cover_tail=cover_tail)
    windows = np.stack([signal[s : s + card.l_m] for s in starts]).astype(np.float32)
    return (windows, starts) if return_starts else windows
