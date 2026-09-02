"""Recoverability: would a standard per-modality filter turn a POOR window into a USABLE one?

The quality models grade the signal *as provided* (raw). This module runs a SECOND inference pass
on a filtered copy and, per window, asks whether a poor raw grade (Q0/Q1) becomes usable (Q2/Q3)
after filtering — i.e. the corruption is band-limited noise a downstream filter would remove, not
unrecoverable signal loss. The raw grade stays the source of truth; recoverability is an ADVISORY
overlay (a "↺ recoverable" badge), never a silent re-grade — silently averaging raw+filtered would
mask true corruption, the exact failure the false-clean guard exists to prevent.

Honesty guard: a raw-trained model can be *fooled* into scoring filtered-but-still-corrupt signal as
clean (the false-clean failure, see :mod:`integrity`). So for ECG/PPG a recoverable call is
CORROBORATED by the filter-robust two-detector **bSQI** on the *filtered* window — if the QRS
detectors still disagree there, the higher filtered grade is deceptive and it is NOT called
recoverable. EEG/EDA have no beat-based SQI, so their calls are model-only (surfaced as "likely").
"""
from __future__ import annotations

import numpy as np

#: per-modality passband (Hz) for the "would a standard filter help?" pass. ``(lo, hi)`` bandpass;
#: these are the conventional clinical analysis bands (ECG 0.5–40, PPG 0.5–8, EEG 1–40, EDA ≤1.5).
RECOVERY_BANDS: dict[str, tuple[float, float]] = {
    "ecg": (0.5, 40.0),
    "ppg": (0.5, 8.0),
    "eeg": (1.0, 40.0),
    "eda": (0.05, 1.5),
}

#: ordinal index (in the card's worst→best ``class_order``) of the first "usable" grade — Q2.
USABLE_FROM = 2


def _fft_bandpass(x: np.ndarray, fs: float, lo: float, hi: float | None) -> np.ndarray:
    """Length-preserving zero-phase band limit via rFFT (scipy-free fallback)."""
    L = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(L, d=1.0 / fs)
    mask = np.ones_like(f, dtype=bool)
    if lo and lo > 0:
        mask &= f >= lo
    if hi:
        mask &= f <= hi
    X[~mask] = 0.0
    return np.fft.irfft(X, n=L)


def filter_for_modality(signal, fs: float, modality: str) -> np.ndarray:
    """Return a filtered copy of ``signal`` using the modality's standard passband (length-preserving).

    Uses a 4th-order zero-phase Butterworth (``scipy.signal.filtfilt``); falls back to an FFT band
    limit if scipy is unavailable, and returns the input unchanged when ``fs`` is too low for the
    band or the signal is too short to filter stably.
    """
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    band = RECOVERY_BANDS.get((modality or "").lower())
    fs = float(fs)
    if band is None or fs <= 0 or x.size < 32:
        return x.astype(np.float32)
    lo, hi = band
    nyq = 0.5 * fs
    hi = min(hi, 0.99 * nyq)
    if hi <= lo:                                  # band collapses at very low fs → low-pass only
        lo = 0.0
    try:
        from scipy.signal import butter, filtfilt

        if lo <= 0:
            b, a = butter(4, hi / nyq, btype="lowpass")
        else:
            b, a = butter(4, [lo / nyq, hi / nyq], btype="bandpass")
        if x.size <= 3 * max(len(a), len(b)):     # filtfilt edge-padding needs headroom
            return x.astype(np.float32)
        return np.asarray(filtfilt(b, a, x), dtype=np.float32)
    except Exception:  # noqa: BLE001 - degrade to the FFT band limit, never crash inference
        return _fft_bandpass(x, fs, lo, hi).astype(np.float32)


def _code(tier: str) -> str:
    """Short Q-code from a possibly-verbose class label ('Q0_unacceptable' -> 'Q0')."""
    return str(tier).split("_")[0]


def _grade_index(tier: str, grade_order) -> int:
    """Ordinal position of ``tier`` in the card's worst→best ``grade_order`` (-1 if absent)."""
    codes = [_code(g) for g in grade_order]
    t = _code(tier)
    return codes.index(t) if t in codes else -1


def recoverable_windows(raw_tiers, filtered_tiers, grade_order, usable_from: int = USABLE_FROM):
    """Per-window ordinal rule: recoverable when the RAW grade is poor (< ``usable_from``) and the
    FILTERED grade is usable (>= ``usable_from``).

    Returns ``(recoverable[bool array], recovered_tier[list[str]])`` — ``recovered_tier[i]`` is the
    short Q-code of the filtered grade for recoverable windows, else "".
    """
    raw_tiers = list(raw_tiers)
    filtered_tiers = list(filtered_tiers)
    n = len(raw_tiers)
    if len(filtered_tiers) != n:
        raise ValueError("raw_tiers and filtered_tiers must have the same length")
    codes = [_code(g) for g in grade_order]
    rec = np.zeros(n, dtype=bool)
    rtier = [""] * n
    for i in range(n):
        ri = _grade_index(raw_tiers[i], grade_order)
        fi = _grade_index(filtered_tiers[i], grade_order)
        if 0 <= ri < usable_from <= fi:
            rec[i] = True
            rtier[i] = codes[fi] if 0 <= fi < len(codes) else ""
    return rec, rtier
