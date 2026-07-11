"""Tests for the filter-robust integrity guard (false-clean override)."""
import numpy as np

from biosqa.inference.integrity import bsqi, integrity_guard
from biosqa.inference.prefilter import PrefilterVerdict

FS = 250.0


def _periodic_ecg(fs=FS, secs=10.0, hr=1.0, noise=0.02, seed=0):
    """A clean-ish periodic QRS spike train (two detectors agree -> high bSQI)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * secs)) / fs
    x = np.zeros_like(t)
    for c in np.arange(0.5, secs, 1.0 / hr):
        x += np.exp(-0.5 * ((t - c) / 0.02) ** 2)
    return (x + noise * rng.standard_normal(len(t))).astype(np.float32)


def _degraded(fs=FS, secs=10.0, seed=1):
    """A corrupted signal with strong broadband noise (detectors disagree -> low bSQI)."""
    rng = np.random.default_rng(seed)
    return (_periodic_ecg(fs, secs, seed=seed) + 0.8 * rng.standard_normal(int(fs * secs))).astype(np.float32)


def test_bsqi_clean_higher_than_degraded():
    assert bsqi(_periodic_ecg(), FS) > bsqi(_degraded(), FS)


def test_guard_overrides_prefiltered_degraded():
    x = np.stack([_degraded(seed=i) for i in range(6)])          # multi-lead degraded
    pf = PrefilterVerdict(True, 0.9, ["high-frequency band suppressed"])
    v = integrity_guard(x, FS, "ecg", model_p_unusable=0.1, prefilter_verdict=pf)  # model reads clean
    assert v.corrupt_override and v.reasons


def test_guard_no_override_on_raw_input():
    x = np.stack([_degraded(seed=i) for i in range(6)])
    pf = PrefilterVerdict(False, 0.0, [])                         # NOT pre-filtered
    v = integrity_guard(x, FS, "ecg", 0.1, pf)
    assert not v.corrupt_override                                 # raw path unchanged


def test_guard_no_override_on_clean():
    x = np.stack([_periodic_ecg(seed=i) for i in range(6)])       # clean, high bSQI
    pf = PrefilterVerdict(True, 0.9, ["high-frequency band suppressed"])
    v = integrity_guard(x, FS, "ecg", 0.1, pf)
    assert not v.corrupt_override
