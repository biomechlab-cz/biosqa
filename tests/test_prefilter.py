"""Tests for the pre-filter detector (false-clean guard)."""
import numpy as np
from scipy.signal import butter, filtfilt

from biosqa.inference.prefilter import detect_prefiltering


def _synth_ecg(fs=500.0, secs=10.0, seed=0):
    """A crude broadband synthetic ECG-ish signal with real HF content (so 'raw' is not flagged)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * secs)) / fs
    beats = np.zeros_like(t)
    for k in range(int(secs)):
        c = k + 0.5
        beats += np.exp(-0.5 * ((t - c) / 0.02) ** 2)          # narrow QRS-like spikes (broadband)
    x = beats + 0.15 * np.sin(2 * np.pi * 0.3 * t)             # baseline wander
    x = x + 0.05 * rng.standard_normal(len(t))                # broadband noise (HF content)
    return x.astype(np.float32)


def _lowpass(x, fs, hz):
    b, a = butter(4, hz / (fs / 2), "low")
    return filtfilt(b, a, x.astype(np.float64)).astype(np.float32)


def test_raw_not_flagged():
    x = _synth_ecg()
    v = detect_prefiltering(x, 500.0, "ecg")
    assert not v.prefiltered, f"raw signal wrongly flagged: {v.reasons}"


def test_lowpass_flagged():
    x = _lowpass(_synth_ecg(), 500.0, 15.0)                    # aggressive low-pass (the false-clean case)
    v = detect_prefiltering(x, 500.0, "ecg")
    assert v.prefiltered and any("high-frequency" in r for r in v.reasons)
    assert 0.0 <= v.score <= 1.0


def test_multichannel_input():
    x = np.stack([_synth_ecg(seed=i) for i in range(3)])       # [C, L]
    v = detect_prefiltering(x, 500.0, "ecg")
    assert isinstance(v.as_dict(), dict) and "prefiltered" in v.as_dict()
