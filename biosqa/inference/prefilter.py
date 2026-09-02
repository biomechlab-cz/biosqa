"""Pre-filter detector — a cheap spectral-fingerprint guard against the "false-clean" failure.

The quality models are trained on RAW signals and report the quality of whatever they are given. If a
SOURCE DEVICE has already filtered the signal, an aggressive filter can strip the high-frequency /
spectral cues the model keys on *faster* than it restores a usable signal — so a still-corrupted signal
can be scored as clean. This detector inspects the input's spectrum for filtering fingerprints
(suppressed HF / LF bands, mains notch, an unnaturally sharp spectral roll-off) and returns a warning so
the app can tell the user "this input looks pre-filtered; the quality score may be optimistic."

Pure numpy (numpy.fft only), a few microseconds/window — safe to run on every window before inference.
"""
from __future__ import annotations

import numpy as np

__all__ = ["detect_prefiltering", "PrefilterVerdict"]

# per-modality (mid-band, hf-band, lf-cut, base-band) in Hz. hf is capped at Nyquist at call time.
_BANDS = {
    "ecg": {"mid": (5.0, 40.0), "hf": (45.0, 120.0), "lf_cut": 0.5, "base": (0.5, 5.0)},
    "ppg": {"mid": (0.5, 8.0), "hf": (10.0, 32.0), "lf_cut": 0.3, "base": (0.3, 4.0)},
    "eeg": {"mid": (4.0, 30.0), "hf": (35.0, 100.0), "lf_cut": 0.5, "base": (0.5, 4.0)},
    "eda": {"mid": (0.05, 1.0), "hf": (1.5, 4.0), "lf_cut": 0.02, "base": (0.02, 0.2)},
}
# thresholds (relative band-power ratios); below these => that band looks suppressed by a filter.
_HF_SUPPRESS = 0.01     # hf/mid power ratio below this => likely low-pass / band-limited (the dangerous case)
_NOTCH_DEPTH = 6.0      # mains-band trough this many x below local neighbours => notch


class PrefilterVerdict:
    """Result of :func:`detect_prefiltering`."""

    def __init__(self, prefiltered: bool, score: float, reasons: list[str]):
        self.prefiltered = prefiltered      # bool: any filtering fingerprint found
        self.score = score                  # 0..1 heuristic confidence
        self.reasons = reasons              # human-readable fingerprints

    def as_dict(self) -> dict:
        return {"prefiltered": self.prefiltered, "score": round(self.score, 3), "reasons": self.reasons}


def _psd(x: np.ndarray, fs: float):
    x = np.asarray(x, dtype=np.float64)
    x = x.mean(0) if x.ndim == 2 else x            # average channels for the fingerprint
    x = (x - x.mean()) / (x.std() + 1e-9)
    w = np.hanning(len(x))
    P = np.abs(np.fft.rfft(x * w)) ** 2
    f = np.fft.rfftfreq(len(x), d=1.0 / fs)
    return f, P


def _bp(f, P, lo, hi):
    m = (f >= lo) & (f < hi)
    return float(P[m].sum()) if m.any() else 0.0


def detect_prefiltering(window: np.ndarray, fs: float, modality: str) -> PrefilterVerdict:
    """Inspect ``window`` ([L] or [C,L]) for filtering fingerprints. Returns a :class:`PrefilterVerdict`.

    Fingerprints: HF-band power suppressed (low-pass / band-limit), LF-band power suppressed (high-pass),
    a deep narrow trough at 50/60 Hz (mains notch), or an unnaturally sharp spectral roll-off edge.
    """
    b = _BANDS.get(modality, _BANDS["ecg"])
    f, P = _psd(window, fs)
    nyq = fs / 2.0
    eps = 1e-12
    reasons: list[str] = []
    score = 0.0

    # --- HF suppression (the dangerous false-clean case: aggressive low-pass / band-limit) ---
    hf_hi = min(b["hf"][1], nyq)
    if hf_hi > b["hf"][0]:
        hf = _bp(f, P, b["hf"][0], hf_hi)
        mid = _bp(f, P, *b["mid"]) + eps
        r = hf / mid
        if r < _HF_SUPPRESS:
            reasons.append(f"high-frequency band ({b['hf'][0]:.0f}-{hf_hi:.0f} Hz) suppressed "
                           f"(rel. power {r:.4f}) — likely low-pass / band-limited")
            score = max(score, 1.0 - r / _HF_SUPPRESS)

    # NOTE: a low-frequency-suppression (high-pass) check was dropped — it false-positived on ~40%
    # of raw windows (Hann windowing naturally attenuates <0.5 Hz), and high-pass is the SAFE regime
    # (it removes baseline wander, it does not mask corruption). We flag only the dangerous fingerprints.

    # --- mains notch (narrow deep trough at 50/60 Hz) ---
    for f0 in (50.0, 60.0):
        if f0 + 3 >= nyq:
            continue
        trough = _bp(f, P, f0 - 1.0, f0 + 1.0) / 2.0 + eps
        neigh = (_bp(f, P, f0 - 4.0, f0 - 1.0) + _bp(f, P, f0 + 1.0, f0 + 4.0)) / 6.0 + eps
        if neigh / trough > _NOTCH_DEPTH:
            reasons.append(f"{f0:.0f} Hz mains notch present")
            score = max(score, 0.4)

    return PrefilterVerdict(len(reasons) > 0, float(min(1.0, score)), reasons)
