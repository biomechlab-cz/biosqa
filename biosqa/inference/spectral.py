"""Host-side spectral band-power channels for the two-input dual-branch ECG model.

Mirrors ``biosqa.data.signal_channels.spectral_band_channels`` exactly (the app and
the training package have disjoint deps, so this is a standalone pure-numpy copy).
The dual-branch ONNX takes ``x_raw [B,1,L]`` and ``x_spec [B,C,L]``; the app computes
``x_spec`` here from the same raw window (numpy.fft only — no scipy, no torch, and no
in-graph FFT). The graph z-scores both inputs per-window, so this returns RAW (un-
normalized) band-power channels.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-8


def _interp_to_L(a: np.ndarray, length: int) -> np.ndarray:
    n_rows, n_frames = a.shape
    if n_frames == length:
        return a
    xp = np.linspace(0.0, 1.0, n_frames)
    xq = np.linspace(0.0, 1.0, length)
    return np.stack([np.interp(xq, xp, a[i]) for i in range(n_rows)], axis=0)


def spectral_band_channels(x: np.ndarray, fs: float, bands, frame_s: float = 0.25,
                           hop_s: float = 0.0625) -> np.ndarray:
    """``x`` ``[N, 1, L]`` (or ``[N, L]``) -> ``[N, len(bands), L]`` log band-power channels.

    ``bands`` is a list of ``(lo_hz, hi_hz)`` pairs (from the model card's
    ``spectral_preprocessing.bands_hz``).
    """
    if x.ndim == 3:
        x = x[:, 0, :]
    x = np.asarray(x, dtype=np.float64)
    n, length = x.shape
    frame = max(8, int(round(frame_s * fs)))
    frame = min(frame, length)
    hop = max(1, int(round(hop_s * fs)))
    n_frames = 1 + (length - frame) // hop
    win = np.hanning(frame)
    idx = (np.arange(n_frames)[:, None] * hop) + np.arange(frame)[None, :]
    frames = x[:, idx] * win[None, None, :]
    spec = np.fft.rfft(frames, axis=-1)
    power = spec.real ** 2 + spec.imag ** 2
    freqs = np.fft.rfftfreq(frame, d=1.0 / fs)
    chans = []
    for lo, hi in bands:
        m = (freqs >= lo) & (freqs < hi)
        bp = power[:, :, m].sum(-1) if m.any() else np.zeros((n, n_frames))
        chans.append(_interp_to_L(np.log1p(bp), length).astype(np.float32))
    return np.stack(chans, axis=1).astype(np.float32)
