"""Interpretable classical-SQI breakdown for a single window (research3.md Rec.1/3: classical SQIs
are interpretable and best fused; Elgendi PPG SQIs; CMU Auton Lab "a combination of SQIs beats any
single index").

Read-only EXPLAINABILITY — shows WHY a window looks the way it does (and, per the workflow
experiment, surfaces the discrepancy when the raw model over-confidently grades a corrupt window
clean). It does NOT change the model's grade. Pure numpy (sub-ms/window); computed on demand for the
selected segment only. Reuses the filter-robust bSQI + rhythm regularity from :mod:`integrity`.
"""
from __future__ import annotations

import numpy as np

from biosqa.inference.integrity import autocorr_regularity, bsqi, tsqi

_EPS = 1e-9


def _band_power(x, fs, lo, hi):
    X = np.abs(np.fft.rfft(x - x.mean())) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    return float(X[(f >= lo) & (f < hi)].sum())


def _psqi(x, fs):
    """QRS-band spectral concentration P(5-15)/P(5-40) — higher = cleaner (energy sits in the QRS band)."""
    return _band_power(x, fs, 5.0, 15.0) / (_band_power(x, fs, 5.0, 40.0) + _EPS)


def _bassqi(x, fs):
    """Relative baseline power 1 - P(0-1)/P(0-40) — higher = cleaner (low → baseline wander dominates)."""
    return 1.0 - _band_power(x, fs, 0.0, 1.0) / (_band_power(x, fs, 0.0, 40.0) + _EPS)


def _skewness(x):
    z = x - x.mean()
    return float(np.mean(z ** 3) / ((x.std() + _EPS) ** 3))


def _kurtosis(x):
    z = x - x.mean()
    return float(np.mean(z ** 4) / ((x.std() + _EPS) ** 4))


def _amp_entropy(x, bins=32):
    h, _ = np.histogram(x, bins=bins)
    p = h / (h.sum() + _EPS)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(bins))   # normalized 0..1


def _spectral_entropy(x, fs, lo=1.0, hi=45.0):
    """Normalized Shannon entropy of the power spectrum over the EEG band — 0 = all power in one bin
    (a pure rhythm), 1 = perfectly flat (white/broadband noise). This is the quantity the EEG panel's
    row NAMES; it used to display :func:`_amp_entropy`, a 32-bin AMPLITUDE histogram that never touches
    the spectrum (measured: a 10 Hz sine and white noise scored 0.94 vs 0.84, i.e. the wrong way round
    for the "flat -> broadband noise" reading the row promises)."""
    X = np.abs(np.fft.rfft(x - x.mean())) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    X = X[(f >= lo) & (f <= min(hi, fs / 2.0 - 1.0))]
    if X.size < 4:
        return 0.0
    p = X / (X.sum() + _EPS)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(X.size))


def _hf_fraction(x, fs, split_hz):
    power = np.abs(np.fft.rfft(x - x.mean())) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    return float(power[f >= split_hz].sum() / (power.sum() + _EPS))


def _aperiodic_slope(x, fs):
    """Slope of log-power vs log-frequency over 2-40 Hz — the EEG aperiodic (1/f) exponent. Physiological
    EEG has a clear negative slope (~-1..-3); a flat slope (→0) signals broadband noise / EMG contamination."""
    X = np.abs(np.fft.rfft(x - x.mean())) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    band = (f >= 2.0) & (f <= min(40.0, fs / 2.0 - 1.0))
    if band.sum() < 4:
        return 0.0
    return float(np.polyfit(np.log(f[band] + _EPS), np.log(X[band] + _EPS), 1)[0])


def _hjorth_complexity(x):
    """Hjorth complexity: how far the waveform departs from a pure sinusoid. ~1 for a clean rhythm, rising
    with broadband/spiky contamination."""
    dx = np.diff(x); ddx = np.diff(dx)
    v0, v1, v2 = np.var(x) + _EPS, np.var(dx) + _EPS, np.var(ddx) + _EPS
    mob = np.sqrt(v1 / v0)
    return float(np.sqrt(v2 / v1) / (mob + _EPS))


def _line_noise_index(x, fs):
    """Fraction of spectral power within ±2 Hz of 50 or 60 Hz mains (high → powerline contamination)."""
    X = np.abs(np.fft.rfft(x - x.mean())) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    line = sum(X[(f >= f0 - 2.0) & (f <= f0 + 2.0)].sum() for f0 in (50.0, 60.0) if f0 < fs / 2.0)
    return float(line / (X.sum() + _EPS))


def _bar(value, good, bad):
    """Quality FILL 0..1 for the display bar: 1.0 when ``value`` is at/beyond the ``good`` anchor, 0.0 at/
    beyond the ``bad`` anchor, linear between (clamped). Higher = better quality, so the bar length tracks
    the reader's intuition — a clean bSQI≈1.0 renders a FULL (green) bar, and a corrupt one an (near-)empty
    (red) bar — regardless of whether the underlying index is "higher=cleaner" or "higher=noisier"."""
    return float(1.0 - min(1.0, max(0.0, (value - good) / ((bad - good) + _EPS))))


def sqi_breakdown(window, fs: float, modality: str) -> list[dict]:
    """Per-window named SQIs as ``[{name, value, hint, bar, desc}, ...]`` for the inspector panel.
    ``hint`` says which direction is cleaner; ``bar`` is an indicative 0(good)..1(bad) position."""
    x = np.asarray(window, dtype=np.float64).reshape(-1)
    fs = float(fs) or 1.0
    m = (modality or "").lower()
    if x.size < 8:
        return []
    sk, ku, ent = _skewness(x), _kurtosis(x), _amp_entropy(x)

    if m in ("ecg", "ppg"):
        # HF-noise split is modality-aware: ECG uses 40 Hz (EMG band); PPG's Nyquist can be ~32 Hz and its
        # pulse lives < ~8 Hz, so a 40 Hz split sits above Nyquist (structurally 0) — split at 8 Hz instead.
        hf_split = 40.0 if m == "ecg" else 8.0
        b = bsqi(x, fs); reg = autocorr_regularity(x, fs); ts = tsqi(x, fs, m); hf = _hf_fraction(x, fs, hf_split)
        rows = [
            {"name": "bSQI", "value": round(b, 2), "hint": "higher = cleaner",
             "bar": _bar(b, 1.0, 0.5), "desc": "Agreement of two independent beat detectors"},
            {"name": "tSQI", "value": round(ts, 2), "hint": "higher = cleaner",
             "bar": _bar(ts, 0.95, 0.5), "desc": "Beat-to-template correlation — morphology consistency (filter-robust)"},
            {"name": "Rhythm reg.", "value": round(reg, 2), "hint": "higher = cleaner",
             "bar": _bar(reg, 0.9, 0.3), "desc": "Autocorrelation regularity in the beat band"},
            {"name": "Kurtosis", "value": round(ku, 1), "hint": "higher = cleaner",
             "bar": _bar(ku, 12.0, 2.5), "desc": "Peakedness — sharp beats give high kurtosis"},
            {"name": "HF noise", "value": round(hf, 3), "hint": "higher = noisier",
             "bar": _bar(hf, 0.02, 0.15), "desc": f"Energy fraction above {hf_split:.0f} Hz (EMG/motion)"},
            {"name": "Skewness", "value": round(sk, 2), "hint": "|·| higher = cleaner",
             "bar": _bar(abs(sk), 3.0, 0.2), "desc": "Waveform asymmetry (beats are skewed)"},
        ]
        if m == "ecg":                                  # QRS-band spectral-ratio SQIs (Li/Clifford) — ECG only
            ps = _psqi(x, fs); bas = _bassqi(x, fs)
            rows.insert(4, {"name": "pSQI", "value": round(ps, 2), "hint": "higher = cleaner",
                            "bar": _bar(ps, 0.7, 0.2), "desc": "QRS-band spectral concentration P(5-15)/P(5-40)"})
            rows.insert(5, {"name": "basSQI", "value": round(bas, 2), "hint": "higher = cleaner",
                            "bar": _bar(bas, 0.95, 0.5), "desc": "Baseline-power SQI — low → baseline wander"})
        return rows
    if m == "eeg":
        hf = _hf_fraction(x, fs, 45.0)
        slope = _aperiodic_slope(x, fs); hjc = _hjorth_complexity(x); line = _line_noise_index(x, fs)
        # Spectral entropy is INFORMATIONAL: it now measures what its name says (it used to display the
        # amplitude-histogram entropy with an inverted bar), but on the reference EEG corpus it does not
        # separate the grade classes — 300 store_v8 test windows per class give 0.71/0.68/0.63/0.72 for
        # Q0..Q3, i.e. the cleanest class is not the most peaked. So it explains a window's spectrum
        # without voting in :func:`sqi_consensus`, which fires the discordance banner; the panel's
        # broadband evidence comes from the aperiodic slope and the HF fraction, which are directional.
        spec_ent = _spectral_entropy(x, fs)
        return [
            {"name": "Aperiodic 1/f", "value": round(slope, 2), "hint": "steeper (−) = cleaner",
             "bar": _bar(slope, -2.0, 0.0), "desc": "Log-power vs log-freq slope — flat → broadband noise/EMG"},
            {"name": "Spec. entropy", "value": round(spec_ent, 2), "hint": "informational", "informational": True,
             "bar": _bar(spec_ent, 0.45, 0.90), "desc": "Flatness of the 1-45 Hz power spectrum — flat → broadband noise"},
            {"name": "Hjorth comp.", "value": round(hjc, 2), "hint": "higher = noisier",
             "bar": _bar(hjc, 1.5, 6.0), "desc": "Departure from a pure rhythm — rises with spiky/broadband noise"},
            {"name": "Kurtosis", "value": round(ku, 1), "hint": "higher = noisier",
             "bar": _bar(ku, 3.0, 15.0), "desc": "Spikes / blinks give high kurtosis"},
            {"name": "Line 50/60", "value": round(line, 3), "hint": "higher = noisier",
             "bar": _bar(line, 0.02, 0.25), "desc": "Power within ±2 Hz of 50/60 Hz mains"},
            {"name": "HF fraction", "value": round(hf, 3), "hint": "higher = noisier",
             "bar": _bar(hf, 0.05, 0.30), "desc": "High-frequency energy fraction (muscle/EMG)"},
        ]
    if m == "eda":
        # power fraction above the phasic band (0.6 Hz) — motion/HF energy. Bounded 0..1, so it avoids the
        # heavy-tailed blow-up of a percentile/median diff ratio on the near-constant EDA tonic baseline.
        motion = _hf_fraction(x, fs, 0.6)
        return [
            {"name": "Motion", "value": round(motion, 3), "hint": "higher = noisier",
             "bar": _bar(motion, 0.05, 0.4), "desc": "Power fraction above 0.6 Hz (motion / high-freq)"},
            {"name": "Skewness", "value": round(sk, 2), "hint": "informational", "informational": True,
             "bar": _bar(abs(sk), 0.0, 3.0), "desc": "Waveform asymmetry (clean EDA is inherently skewed)"},
        ]
    return [{"name": "Entropy", "value": round(ent, 2), "hint": "higher = noisier",
             "bar": _bar(ent, 0.5, 0.9), "desc": "Amplitude-histogram entropy"}]


def sqi_consensus(rows) -> float:
    """Fuse a breakdown's indicative bars into ONE 0..1 quality consensus (higher = cleaner) — the
    "combination of SQIs beats any single index" finding (Li 2014). Mean quality-fill (``bar``, which is
    itself 0..1 cleanliness now) over the CLEANLINESS-directional rows only: rows tagged ``informational``
    (e.g. EDA skewness, which is inherently skewed on clean signals) are excluded so they can't drag a clean
    window into false discordance. Returns ``-1.0`` when no eligible row exists — a sentinel for "not
    computable", DISTINCT from ``0.0`` (a real, maximally-corrupt consensus that must still trigger the
    discordance banner)."""
    bars = [float(r.get("bar", 0.5)) for r in (rows or []) if not r.get("informational")]
    if not bars:
        return -1.0
    return float(max(0.0, min(1.0, sum(bars) / len(bars))))
