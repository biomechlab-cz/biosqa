"""Host-side PPG SQI feature vector for the 2-input PPG fusion model (numpy only).

Standalone port of ``biosqa.data.sqa_features.ppg_sqi_vector`` — the app has no
``biosqa`` dependency. Produces the 6-dim scale-invariant SQI vector the PPG fusion
ONNX expects as its 2nd input (``x_feat``). The ONNX graph bakes the training
feature-standardization (mean/std), so we feed the RAW vector here. numpy.fft only —
no scipy, no in-graph FFT.

Order MUST match the exporter's ``ppg_sqi_vector`` (model_card.feature_preprocessing.
feat_names): [skew, kurt, cardiac_ratio, hf_ratio, spec_entropy, pulse_regularity].
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-8

FEAT_NAMES = (
    "ppg_skew", "ppg_kurt", "ppg_cardiac_ratio", "ppg_hf_ratio",
    "ppg_spec_entropy", "ppg_pulse_regularity",
)


def _as_rows(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 3:
        X = X[:, 0, :]
    if X.ndim == 1:
        X = X[None, :]
    return X


def _zrows(X: np.ndarray) -> np.ndarray:
    mu = X.mean(1, keepdims=True)
    sd = X.std(1, keepdims=True)
    return (X - mu) / (sd + _EPS)


def _psd(X: np.ndarray, fs: float):
    L = X.shape[1]
    w = np.hanning(L)
    F = np.fft.rfft(X * w, axis=1)
    psd = F.real ** 2 + F.imag ** 2
    freqs = np.fft.rfftfreq(L, d=1.0 / fs)
    return freqs, psd


def _bandpower(freqs, psd, lo, hi):
    m = (freqs >= lo) & (freqs < hi)
    if not m.any():
        return np.zeros(psd.shape[0])
    return psd[:, m].sum(1)


def _autocorr_regularity(z: np.ndarray, fs: float, f_lo: float, f_hi: float) -> np.ndarray:
    N, L = z.shape
    lag_lo = max(1, int(np.floor(fs / f_hi)))
    lag_hi = min(L - 1, int(np.ceil(fs / f_lo)))
    if lag_hi <= lag_lo:
        return np.zeros(N)
    out = np.zeros(N)
    denom = (z ** 2).sum(1) + _EPS
    for lag in range(lag_lo, lag_hi + 1):
        c = (z[:, :L - lag] * z[:, lag:]).sum(1) / denom
        out = np.maximum(out, c)
    return out


def ppg_sqi_vector(X: np.ndarray, fs: float) -> np.ndarray:
    """``[N, L]`` or ``[N, 1, L]`` PPG windows -> ``[N, 6]`` float32 SQI vector."""
    X = _as_rows(X)
    z = _zrows(X)
    skew = (z ** 3).mean(1)
    kurt = (z ** 4).mean(1) - 3.0
    freqs, psd = _psd(z, fs)
    total = _bandpower(freqs, psd, 0.1, min(8.0, fs / 2)) + _EPS
    cardiac = _bandpower(freqs, psd, 1.0, 2.25) / total
    hf = _bandpower(freqs, psd, 8.0, fs / 2) / (_bandpower(freqs, psd, 0.1, fs / 2) + _EPS)
    band_m = (freqs >= 0.1) & (freqs < min(8.0, fs / 2))
    p = psd[:, band_m]
    p = p / (p.sum(1, keepdims=True) + _EPS)
    sent = -(p * np.log(p + _EPS)).sum(1) / np.log(max(2, int(band_m.sum())))
    reg = _autocorr_regularity(z, fs, f_lo=0.7, f_hi=2.25)
    feat = np.stack([skew, kurt, cardiac, hf, sent, reg], axis=1)
    return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
