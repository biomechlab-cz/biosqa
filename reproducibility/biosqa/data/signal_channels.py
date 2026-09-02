"""Scale-invariant transient / morphology feature channels for the artifact-TYPE
head (research slate quick-wins #8 STA/LTA, #10 LITE trend/peak filters).

The type-loss sweep showed the artifact-type macro-F1 is SIGNAL-limited, not
loss-limited: reweighting cannot manufacture cues the raw amplitude trace does not
expose. These fixed, parameter-free channels hand the type head explicit transient
and morphology cues that mean-pooling over raw amplitude discards:

* **STA/LTA** (Allen 1978, seismology): short-term over long-term power ratio. It
  is scale-invariant (a ratio -> robust to cross-cohort gain shift) and spikes at
  abrupt onsets (motion bursts, electrode pop, lead-off transitions).
* **trend**: a slow moving average ~ baseline-wander component.
* **peak/edge**: raw minus a short moving average ~ a high-pass that exposes
  spikes/QRS/motion (LITE-style fixed peak stencil).

All are pure-numpy running means via cumulative sums -> exactly reproducible in the
onnxruntime/numpy app at deploy time (no torch, no scipy.signal). Everything traces
to ONNX too (cumulative-sum + slice + divide), but the intended deploy path is the
app recomputing the channels in numpy and feeding a multi-channel window.
"""
from __future__ import annotations

import numpy as np

__all__ = ["running_mean", "sta_lta", "trend_channel", "peak_channel", "build_channels", "CHANNELS",
           "spectral_band_channels", "MODALITY_BANDS", "band_channel_names",
           "spectral_sqi_vector", "sqi_feature_names"]

CHANNELS = ("raw", "stalta", "trend", "peak")
_EPS = 1e-8

# Frequency bands per modality (Hz) for STFT band-power channels. Chosen to isolate
# the SPECTRALLY-DEFINED artifact types: baseline-wander (lows), the physiological
# band, muscle/EMG (highs), and powerline (50/60 Hz). numpy.fft-computable at deploy.
MODALITY_BANDS = {
    "ecg": [(0.0, 0.7), (0.7, 8.0), (8.0, 40.0), (40.0, 125.0), (48.0, 62.0)],
    "ppg": [(0.0, 0.5), (0.5, 4.0), (4.0, 12.0), (12.0, 32.0)],
    "eeg": [(0.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 100.0), (48.0, 62.0)],
    "eda": [(0.0, 0.05), (0.05, 0.5), (0.5, 1.0), (1.0, 4.0)],
}


def band_channel_names(modality: str):
    return [f"band_{lo:g}-{hi:g}Hz" for (lo, hi) in MODALITY_BANDS[modality]]


def _interp_to_L(a: np.ndarray, L: int) -> np.ndarray:
    """Linear-resample each row of ``a`` [N, F] to length ``L`` -> [N, L]."""
    N, F = a.shape
    if F == L:
        return a
    xp = np.linspace(0.0, 1.0, F)
    xq = np.linspace(0.0, 1.0, L)
    return np.stack([np.interp(xq, xp, a[i]) for i in range(N)], axis=0)


def spectral_band_channels(X: np.ndarray, fs: float, bands, frame_s: float = 0.25,
                           hop_s: float = 0.0625) -> np.ndarray:
    """Time-localized log band-power channels via a pure-numpy STFT.

    ``X`` ``[N, 1, L]`` (or ``[N, L]``) -> ``[N, n_bands, L]``: a Hann-windowed STFT
    (numpy.fft.rfft) gives per-frame power; power is summed within each frequency
    band, log1p-compressed, and linearly resampled back to length ``L`` so each band
    becomes a time-series channel aligned with the raw window. This hands a 1D-CNN
    the frequency content the raw-amplitude trace hides, with the frequency axis
    REDUCED to a few physiologically-meaningful bands (cheap; no 2D image, no in-graph
    FFT). The app recomputes it in numpy from the same raw window.
    """
    if X.ndim == 3:
        X = X[:, 0, :]
    x = X.astype(np.float64)
    N, L = x.shape
    frame = max(8, int(round(frame_s * fs)))
    frame = min(frame, L)
    hop = max(1, int(round(hop_s * fs)))
    n_frames = 1 + (L - frame) // hop
    win = np.hanning(frame)
    idx = (np.arange(n_frames)[:, None] * hop) + np.arange(frame)[None, :]   # [F, frame]
    frames = x[:, idx] * win[None, None, :]                                  # [N, F, frame]
    spec = np.fft.rfft(frames, axis=-1)
    power = spec.real ** 2 + spec.imag ** 2                                  # [N, F, nfreq]
    freqs = np.fft.rfftfreq(frame, d=1.0 / fs)
    chans = []
    for lo, hi in bands:
        m = (freqs >= lo) & (freqs < hi)
        bp = power[:, :, m].sum(-1) if m.any() else np.zeros((N, n_frames))  # [N, F]
        chans.append(_interp_to_L(np.log1p(bp), L).astype(np.float32))       # [N, L]
    return np.stack(chans, axis=1)                                           # [N, n_bands, L]


def running_mean(x: np.ndarray, k: int) -> np.ndarray:
    """Centered uniform running mean over the last axis, window ``k`` (odd-ish),
    edge-padded so the output length matches the input. Pure cumulative-sum."""
    k = max(1, int(k))
    if k == 1:
        return x.astype(np.float64, copy=True)
    pad = k // 2
    xp = np.pad(x.astype(np.float64), [(0, 0)] * (x.ndim - 1) + [(pad, k - 1 - pad)], mode="edge")
    cs = np.cumsum(xp, axis=-1)
    cs = np.concatenate([np.zeros_like(cs[..., :1]), cs], axis=-1)
    out = (cs[..., k:] - cs[..., :-k]) / k
    return out[..., : x.shape[-1]]


def sta_lta(x: np.ndarray, fs: float, short_s: float = 0.05, long_s: float = 0.5) -> np.ndarray:
    """Short-term/long-term power ratio (scale-invariant onset detector)."""
    e = x.astype(np.float64) ** 2
    sta = running_mean(e, int(round(short_s * fs)))
    lta = running_mean(e, int(round(long_s * fs)))
    return sta / (lta + _EPS)


def trend_channel(x: np.ndarray, fs: float, win_s: float = 0.25) -> np.ndarray:
    """Slow moving average ~ baseline-wander component."""
    return running_mean(x, int(round(win_s * fs)))


def peak_channel(x: np.ndarray, fs: float, win_s: float = 0.05) -> np.ndarray:
    """Raw minus short moving average ~ high-pass exposing spikes/QRS/motion."""
    return x.astype(np.float64) - running_mean(x, int(round(win_s * fs)))


def tkeo_channel(x: np.ndarray, fs: float, smooth_s: float = 0.03) -> np.ndarray:
    """Teager-Kaiser energy operator envelope (Kaiser 1990), log-compressed + smoothed.

    ``psi[n] = x[n]^2 - x[n-1]*x[n+1]`` is an instantaneous AM-FM energy that SPIKES at
    transient bursts/motion and COLLAPSES on flatline/dropout — the uncovered time-domain
    cue for the residual hard type classes (burst_transient, clipping_flatline). O(L)
    numpy, deploy-reproducible. Returns a per-window z-scored energy envelope [N, L]."""
    x = x.astype(np.float64)
    psi = np.zeros_like(x)
    psi[:, 1:-1] = x[:, 1:-1] ** 2 - x[:, :-2] * x[:, 2:]
    psi = np.abs(psi)
    psi = running_mean(psi, max(1, int(round(smooth_s * fs))))
    psi = np.log1p(psi)
    return ((psi - psi.mean(-1, keepdims=True)) / (psi.std(-1, keepdims=True) + _EPS)).astype(np.float32)


def _harmonic_neighbor_ratio(psd, freqs, f0, bw=1.0, gap=1.0, nbw=3.0):
    """Energy in [f0-bw, f0+bw] over energy in the flanking neighbor bands
    (amplitude-invariant powerline/harmonic detector). psd/freqs 1D."""
    line = (freqs >= f0 - bw) & (freqs <= f0 + bw)
    lo = (freqs >= f0 - gap - nbw) & (freqs < f0 - gap)
    hi = (freqs > f0 + gap) & (freqs <= f0 + gap + nbw)
    nb = psd[lo].sum() + psd[hi].sum()
    return float(psd[line].sum() / (nb + _EPS))


def sqi_feature_names(modality: str):
    bands = MODALITY_BANDS[modality]
    names = [f"bandratio_{lo:g}-{hi:g}" for (lo, hi) in bands]
    names += ["spec_entropy", "spec_flatness", "spec_centroid", "spec_slope",
              "dom_freq", "total_logpower", "hjorth_mobility", "hjorth_complexity",
              "hn_ratio_50", "hn_ratio_60", "hn_ratio_100", "hn_ratio_120"]
    return names


def spectral_sqi_vector(X: np.ndarray, fs: float, modality: str) -> np.ndarray:
    """Per-window GLOBAL spectral-SQI feature VECTOR (numpy.rfft only) -> [N, F].

    Complementary to the time-localized band channels: a single whole-window PSD
    reduced to the textbook SQA scalars (band-power ratios ~ pSQI/basSQI, spectral
    entropy/flatness/centroid/slope, dominant frequency, total log-power, Hjorth
    mobility/complexity, and 50/60/100/120 Hz harmonic-neighbor ratios that pinpoint
    powerline without amplitude sensitivity). numpy-only -> deploy-reproducible.
    """
    if X.ndim == 3:
        X = X[:, 0, :]
    x = X.astype(np.float64)
    N, L = x.shape
    xw = (x - x.mean(1, keepdims=True)) * np.hanning(L)[None, :]
    spec = np.fft.rfft(xw, axis=-1)
    psd = spec.real ** 2 + spec.imag ** 2                          # [N, nfreq]
    freqs = np.fft.rfftfreq(L, d=1.0 / fs)
    tot = psd.sum(1, keepdims=True) + _EPS
    feats = []
    for lo, hi in MODALITY_BANDS[modality]:                        # band-power ratios
        m = (freqs >= lo) & (freqs < hi)
        feats.append((psd[:, m].sum(1, keepdims=True) / tot))
    pn = psd / tot                                                 # normalized -> spectral shape
    entropy = -(pn * np.log(pn + _EPS)).sum(1, keepdims=True)
    flatness = np.exp(np.log(psd + _EPS).mean(1, keepdims=True)) / (psd.mean(1, keepdims=True) + _EPS)
    centroid = (freqs[None, :] * pn).sum(1, keepdims=True)
    # spectral slope: linear fit of log-psd vs freq (vectorized closed form)
    lf = freqs; lp = np.log(psd + _EPS)
    fm = lf.mean(); slope = ((lp - lp.mean(1, keepdims=True)) * (lf - fm)[None, :]).sum(1, keepdims=True) / (((lf - fm) ** 2).sum() + _EPS)
    dom = freqs[np.argmax(psd, axis=1)][:, None]
    totlp = np.log(tot)
    dx = np.diff(x, axis=1); ddx = np.diff(dx, axis=1)
    v0 = x.var(1, keepdims=True) + _EPS; v1 = dx.var(1, keepdims=True) + _EPS; v2 = ddx.var(1, keepdims=True) + _EPS
    mob = np.sqrt(v1 / v0)
    comp = np.sqrt(v2 / v1) / (mob + _EPS)
    feats += [entropy, flatness, centroid, slope, dom, totlp, mob, comp]
    hn = np.zeros((N, 4))
    for i in range(N):
        for j, f0 in enumerate((50.0, 60.0, 100.0, 120.0)):
            hn[i, j] = _harmonic_neighbor_ratio(psd[i], freqs, f0) if f0 < fs / 2 else 0.0
    feats.append(hn)
    return np.concatenate([f if f.ndim == 2 else f[:, None] for f in feats], axis=1).astype(np.float32)


def build_channels(X: np.ndarray, fs: float, channels=CHANNELS) -> np.ndarray:
    """``X`` ``[N, 1, L]`` (or ``[N, L]``) -> ``[N, C, L]`` stacking the requested
    channels in order. ``raw`` is the input trace; the rest are derived. Deploy-time
    the app recomputes the derived channels in numpy from the same raw window."""
    if X.ndim == 2:
        X = X[:, None, :]
    raw = X[:, 0, :].astype(np.float64)               # [N, L]
    made = {
        "raw": raw,
        "stalta": sta_lta(raw, fs),
        "trend": trend_channel(raw, fs),
        "peak": peak_channel(raw, fs),
    }
    return np.stack([made[c] for c in channels], axis=1).astype(np.float32)  # [N, C, L]
