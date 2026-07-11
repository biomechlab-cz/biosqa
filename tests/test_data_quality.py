"""Tests for the record-level data-quality report."""
import numpy as np

from biosqa.inference.data_quality import record_quality


def _clean(fs=250.0, secs=20.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * secs)) / fs
    return (np.sin(2 * np.pi * 1.2 * t) + 0.02 * rng.standard_normal(len(t))).astype(np.float32)


def test_clean_record_is_usable():
    q = record_quality(_clean(), 250.0)
    assert q.usable and q.completeness > 0.95 and not q.flags


def test_dropout_gap_detected():
    x = _clean(); x[1000:3000] = 0.0            # 8 s dropout at 250 Hz
    q = record_quality(x, 250.0)
    assert q.n_dropout_gaps >= 1 and q.longest_gap_s >= 7.0
    assert any("dropout" in f for f in q.flags)


def test_clipping_detected():
    x = _clean(); x = np.clip(x, -0.3, 0.3)     # saturate the rails
    q = record_quality(x, 250.0)
    assert q.clipping_frac > 0.05 and any("clipped" in f for f in q.flags)


def test_nan_missing_and_multichannel_worst_case():
    a = _clean(seed=1); b = _clean(seed=2); b[::2] = np.nan   # one broken lead
    q = record_quality(np.stack([a, b]), 250.0)
    assert q.missing_frac > 0.4 and not q.usable


def test_completeness_counts_all_dropouts_not_just_longest():
    """Regression: completeness must reflect TOTAL dropout duration, not only the longest gap."""
    import numpy as np
    from biosqa.inference.data_quality import record_quality
    fs = 250
    n = fs * 60
    x = np.sin(np.arange(n) / fs).astype(float)
    for k in range(20):                       # 20 separate 1 s zero-dropouts = 20 s of 60 s lost
        s = k * 3 * fs
        x[s:s + fs] = 0.0
    rq = record_quality(x, fs)
    assert rq.n_dropout_gaps >= 15
    assert rq.completeness < 0.75             # ~0.67, not the old ~0.98 longest-gap value
