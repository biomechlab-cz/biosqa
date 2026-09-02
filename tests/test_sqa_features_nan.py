"""The host-side SQA feature bank must not abort a whole record on one bad sample.

A single NaN/Inf in a PPG/EEG/EDA window used to make ``np.clip(nan, 1, c).astype
(int64)`` yield INT64_MIN inside ``_dispersion_entropy``; the pattern index then
wrapped and ``np.bincount`` raised for the entire batch, so InferenceTask emitted
``failed`` with a raw numpy message and the user got ZERO graded windows for a
record that was 99.9 % fine (audit 2026-08).
"""
import numpy as np

from biosqa.inference.sqa_features import combined_vector


def _rows(modality, fs, n_s=10.0):
    t = np.arange(int(n_s * fs)) / fs
    return np.sin(2 * np.pi * 1.2 * t)[None, :] + 0.02 * np.cos(2 * np.pi * 9 * t)


def test_one_bad_sample_does_not_abort_the_batch():
    for modality, fs in (("ppg", 64.0), ("eeg", 256.0), ("eda", 8.0)):
        x = _rows(modality, fs, 10.0 if modality != "eda" else 60.0)
        ref, names = combined_vector(x, fs, modality)
        for bad in (np.nan, np.inf, -np.inf):
            dirty = x.copy()
            dirty[0, x.shape[1] // 3] = bad
            feat, names2 = combined_vector(dirty, fs, modality)
            assert feat.shape == ref.shape and names2 == names
            assert np.isfinite(feat).all()


def test_a_fully_non_finite_window_still_returns_finite_features():
    x = np.full((2, 640), np.nan)
    feat, names = combined_vector(x, 64.0, "ppg")
    assert feat.shape[0] == 2 and len(names) == feat.shape[1]
    assert np.isfinite(feat).all()
