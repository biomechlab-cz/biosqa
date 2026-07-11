"""Deploy-ready (scipy-free) port of the rich 12-lead ECG SQI bank.

This is the pure-numpy twin of :mod:`biosqa.data.ecg_sqi` — SAME signature, SAME
126-feature layout and column names — so the desktop app's numpy-only inference path
can compute the ECG rich-SQI vector without importing scipy (which the app stack
deliberately excludes, exactly like :mod:`biosqa.data.sqa_features` for EDA/PPG/EEG).

The ONLY thing scipy provided was ``scipy.signal.find_peaks`` (used by the two QRS
detectors that feed bSQI + HR-plausibility). It is replaced here by :func:`_find_peaks`,
a pure-numpy peak finder matching scipy's ``height`` + ``distance`` semantics:

  1. local maxima ``x[i-1] < x[i] >= x[i+1]``,
  2. keep candidates with value ``>= height``,
  3. greedy min-distance / refractory: sort surviving candidates by value DESCENDING,
     keep a peak only if it is ``>= distance`` samples from every already-kept peak
     (this is scipy's ``distance`` rule — remove-closer-lower-peaks-first).

Everything else (numpy.fft brick-wall band-pass, spectral/moment/flatline/saturation
features, bSQI matching, HR features, cross-lead aggregates) is byte-for-byte the same
math as the scipy reference. ``ecg_rich_sqi_matrix(X [N,12,L], fs) -> (feat [N,D], names)``.

Keep :mod:`biosqa.data.ecg_sqi` as the FROZEN scipy reference; this module must track it.
"""
from __future__ import annotations

import numpy as np

__all__ = ["ecg_rich_sqi_matrix", "ECG_RICH_LEAD_FEATS", "ECG_RICH_AGG_FEATS", "_find_peaks"]

_EPS = 1e-8
# per-lead scalar features (order fixed for reproducible column layout) — MUST equal ecg_sqi.py
ECG_RICH_LEAD_FEATS = ("pSQI", "basSQI", "hf_ratio", "skew", "kurt", "flatline",
                       "saturation", "bSQI", "hr_plaus", "rr_cov")
ECG_RICH_AGG_FEATS = ("bSQI_mean", "bSQI_min", "bSQI_frac_gt05", "hr_plaus_mean",
                      "hr_agreement", "pSQI_mean")


def _bandpass_fft(x: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    """Zero-phase FFT brick-wall band-pass of a 1-D signal (identical to the scipy ref)."""
    L = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(L, d=1.0 / fs)
    X[(f < lo) | (f > hi)] = 0.0
    return np.fft.irfft(X, n=L)


def _find_peaks(x: np.ndarray, height: float, distance: int) -> np.ndarray:
    """Pure-numpy replacement for ``scipy.signal.find_peaks(x, height=, distance=)``.

    Returns sorted indices of local maxima ``x[i-1] < x[i] >= x[i+1]`` whose value is
    ``>= height``, thinned so that every kept peak is ``>= distance`` samples from every
    other (greedy, highest-first — the scipy ``distance`` rule). Signals here are smooth
    floats (MWI / rectified envelope) so exact plateaus are negligible; with no ties the
    ``>=`` right-edge test coincides with scipy's strict ``x[i] > x[i+1]``.
    """
    n = x.shape[0]
    if n < 3:
        return np.empty(0, dtype=np.int64)
    # step 1: local maxima  (left strictly rising, right non-increasing)
    left = x[1:-1] > x[:-2]
    right = x[1:-1] >= x[2:]
    cand = np.nonzero(left & right)[0] + 1
    if cand.size == 0:
        return np.empty(0, dtype=np.int64)
    # step 2: height threshold
    cand = cand[x[cand] >= height]
    if cand.size == 0:
        return np.empty(0, dtype=np.int64)
    dist = max(1, int(distance))
    if cand.size == 1 or dist <= 1:
        return np.sort(cand).astype(np.int64)
    # step 3: greedy min-distance thinning, highest peaks first
    order = cand[np.argsort(x[cand], kind="stable")[::-1]]
    kept: list[int] = []
    for idx in order:
        ok = True
        for k in kept:
            if abs(int(idx) - k) < dist:
                ok = False
                break
        if ok:
            kept.append(int(idx))
    return np.sort(np.asarray(kept, dtype=np.int64))


def _qrs_pantompkins(x: np.ndarray, fs: float) -> np.ndarray:
    """QRS detector A: Pan-Tompkins-lite (5-15 Hz band -> derivative -> square ->
    moving-window integration -> adaptive-threshold peak pick). scipy-free."""
    b = _bandpass_fft(x, fs, 5.0, 15.0)
    d = np.diff(b, prepend=b[:1])
    sq = d * d
    w = max(1, int(round(0.15 * fs)))
    mwi = np.convolve(sq, np.ones(w) / w, mode="same")
    mx = float(mwi.max())
    if mx <= 0:
        return np.empty(0, dtype=np.int64)
    return _find_peaks(mwi, height=0.3 * mx, distance=max(1, int(0.25 * fs)))


def _qrs_energy(x: np.ndarray, fs: float) -> np.ndarray:
    """QRS detector B (independent method for bSQI): rectified 8-20 Hz envelope with a
    mean+0.5·std threshold. Different band + rule than detector A. scipy-free."""
    b = _bandpass_fft(x, fs, 8.0, 20.0)
    env = np.abs(b)
    thr = float(env.mean() + 0.5 * env.std())
    return _find_peaks(env, height=thr, distance=max(1, int(0.25 * fs)))


def _bsqi(pa: np.ndarray, pb: np.ndarray, fs: float, tol: float = 0.15) -> float:
    """bSQI: |matched peaks| / max(|A|,|B|), match within ``tol`` s. Both-empty -> 0.0."""
    if len(pa) == 0 or len(pb) == 0:
        return 0.0
    toln = tol * fs
    matched = sum(1 for p in pa if np.abs(pb - p).min() <= toln)
    return matched / max(len(pa), len(pb))


def _hr_feats(peaks: np.ndarray, fs: float):
    """From QRS peaks -> (RR-plausibility fraction, RR coefficient-of-variation, HR bpm)."""
    if len(peaks) < 2:
        return 0.0, 0.0, 0.0
    rr = np.diff(peaks) / fs
    plaus = float(((rr >= 0.3) & (rr <= 2.0)).mean())
    cov = float(rr.std() / (rr.mean() + _EPS))
    hr = 60.0 / (np.median(rr) + _EPS)
    return plaus, cov, hr


def ecg_rich_sqi_matrix(X: np.ndarray, fs: float):
    """``X [N,12,L]`` (or ``[N,L]`` single lead) -> ``(feat [N,D] float32, names)``.

    Scipy-free twin of :func:`biosqa.data.ecg_sqi.ecg_rich_sqi_matrix` with the SAME
    126-feature layout: per-lead pSQI, basSQI, hf_ratio, skew, kurt, flatline, saturation,
    bSQI, hr_plaus, rr_cov (×C) then aggregates bSQI mean/min/frac>0.5, hr_plaus mean,
    HR agreement, pSQI mean.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 2:
        X = X[:, None, :]
    N, C, L = X.shape

    # --- cheap spectral/moment features, vectorized over (N, C) [identical to ref] ---
    Xf = np.fft.rfft(X, axis=2)
    P = (Xf.real ** 2 + Xf.imag ** 2)
    f = np.fft.rfftfreq(L, d=1.0 / fs)

    def bp(lo, hi):
        m = (f >= lo) & (f < hi)
        return P[:, :, m].sum(2) if m.any() else np.zeros((N, C))

    total = P.sum(2) + _EPS
    pSQI = bp(5, 15) / (bp(5, 40) + _EPS)
    basSQI = bp(0, 1) / (bp(0, 40) + _EPS)
    hf = bp(40, fs / 2) / total
    z = (X - X.mean(2, keepdims=True)) / (X.std(2, keepdims=True) + _EPS)
    skew = (z ** 3).mean(2)
    kurt = (z ** 4).mean(2) - 3.0
    dx = np.diff(X, axis=2)
    flat = (np.abs(dx) < 1e-4).mean(2)
    rng = X.max(2, keepdims=True) - X.min(2, keepdims=True) + 1e-9
    sat = ((np.abs(X - X.max(2, keepdims=True)) < 0.01 * rng) |
           (np.abs(X - X.min(2, keepdims=True)) < 0.01 * rng)).mean(2)

    # --- peak-based features (per record, per lead) ---
    bsqi = np.zeros((N, C)); plaus = np.zeros((N, C)); rrcov = np.zeros((N, C)); hr = np.zeros((N, C))
    for i in range(N):
        for c in range(C):
            xi = X[i, c]
            pa = _qrs_pantompkins(xi, fs)
            pb = _qrs_energy(xi, fs)
            bsqi[i, c] = _bsqi(pa, pb, fs)
            plaus[i, c], rrcov[i, c], hr[i, c] = _hr_feats(pa, fs)

    per_lead = np.concatenate([pSQI, basSQI, hf, skew, kurt, flat, sat, bsqi, plaus, rrcov], axis=1)  # [N, 10C]
    hr_valid = np.where(hr > 0, hr, np.nan)
    hr_std = np.nan_to_num(np.nanstd(hr_valid, axis=1), nan=0.0)
    agg = np.column_stack([
        bsqi.mean(1), bsqi.min(1), (bsqi > 0.5).mean(1),
        plaus.mean(1), 1.0 / (1.0 + hr_std), pSQI.mean(1),
    ])  # [N, 6]
    M = np.concatenate([per_lead, agg], axis=1)
    names = [f"{feat}_L{c}" for feat in ECG_RICH_LEAD_FEATS for c in range(C)] + list(ECG_RICH_AGG_FEATS)
    return np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), names
