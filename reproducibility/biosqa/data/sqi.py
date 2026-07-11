"""Signal-quality indices (SQIs) — Plan 1 §6.1/§9.3, §12.8.

Fast, vectorized, classic SQIs (kurtosis, skewness, band-power SNR proxy,
flatline/saturation fractions, HF-noise ratio) computed over a batch of windows.
Two uses:
  * **robust label cleaning (§12.8):** flag windows whose harmonized label grossly
    disagrees with the signal's SQIs (likely label errors) via per-class robust
    z-score sigma-clipping.
  * **weak-supervision label functions (Phase 4):** each SQI is a noisy LF.

These are intentionally simple/fast (no per-window R-peak detection) so they scale
to 10^5+ windows; NeuroKit2's heavier ``*_quality`` methods can be layered later.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = ["sqi_matrix", "SQI_NAMES", "clean_label_outliers"]

SQI_NAMES = ["kurtosis", "skewness", "snr_band", "flatline", "saturation", "hf_ratio"]

# physiological band (Hz) per modality for the band-power SNR proxy
_BANDS = {"ecg": (0.5, 40.0), "ppg": (0.5, 8.0), "eeg": (0.5, 45.0), "eda": (0.01, 1.0)}


def sqi_matrix(X: np.ndarray, fs: float, modality: str = "ecg") -> np.ndarray:
    """``X [N, 1, L]`` (or ``[N, L]``) -> ``[N, len(SQI_NAMES)]`` SQI features."""
    x = X[:, 0, :] if X.ndim == 3 else X
    x = np.asarray(x, dtype=np.float64)
    N, L = x.shape

    kurt = stats.kurtosis(x, axis=1)
    skew = stats.skew(x, axis=1)

    # band-power SNR proxy: in-band power / total power
    lo, hi = _BANDS.get(modality, (0.5, 40.0))
    freqs = np.fft.rfftfreq(L, d=1.0 / fs)
    P = np.abs(np.fft.rfft(x, axis=1)) ** 2
    total = P.sum(axis=1) + 1e-12
    band = P[:, (freqs >= lo) & (freqs <= hi)].sum(axis=1)
    hf = P[:, freqs > hi].sum(axis=1)
    snr_band = band / total
    hf_ratio = hf / total

    dx = np.diff(x, axis=1)
    flatline = (np.abs(dx) < 1e-4).mean(axis=1)
    rng = x.max(axis=1, keepdims=True) - x.min(axis=1, keepdims=True) + 1e-9
    near_ext = (np.abs(x - x.max(axis=1, keepdims=True)) < 0.01 * rng) | \
               (np.abs(x - x.min(axis=1, keepdims=True)) < 0.01 * rng)
    saturation = near_ext.mean(axis=1)

    M = np.column_stack([kurt, skew, snr_band, flatline, saturation, hf_ratio])
    return np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)


def clean_label_outliers(
    X: np.ndarray, y: np.ndarray, fs: float, modality: str = "ecg", z_thresh: float = 3.0
) -> np.ndarray:
    """Return a boolean keep-mask dropping windows whose SQIs are per-class robust
    outliers (|robust z| > z_thresh on any SQI) — likely mislabeled segments (§12.8).

    Robust z uses median / MAD per class, so it's insensitive to the outliers it
    is trying to find. Conservative: only drops clear disagreements.
    """
    M = sqi_matrix(X, fs, modality)
    keep = np.ones(len(y), dtype=bool)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        if len(idx) < 10:
            continue
        med = np.median(M[idx], axis=0)
        mad = stats.median_abs_deviation(M[idx], axis=0) + 1e-9
        z = np.abs((M[idx] - med) / (1.4826 * mad))
        keep[idx[(z > z_thresh).any(axis=1)]] = False
    return keep
