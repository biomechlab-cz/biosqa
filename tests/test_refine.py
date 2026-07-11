"""Boundary refinement: localize a poor segment to the actual artifact via a fine amplitude score."""
import numpy as np

from biosqa.inference.refine import fine_badness, refine_intervals
from biosqa.inference.segmenter import QualityInterval as QI

FS = 250.0


def _burst_signal(secs=60, burst=(49.0, 50.0)):
    n = int(secs * FS)
    t = np.arange(n) / FS
    x = 0.1 * np.sin(2 * np.pi * 1.1 * t)                    # calm quasi-periodic base
    m = (t >= burst[0]) & (t < burst[1])
    x[m] += 3.0 * np.random.default_rng(0).standard_normal(int(m.sum()))  # big-amplitude burst
    return x.astype(np.float32)


def test_fine_badness_isolates_the_burst():
    bad, bs = fine_badness(_burst_signal(), FS, 1.0)
    assert bs == 250
    flagged = set(np.where(bad)[0].tolist())
    assert 49 in flagged                     # the burst second
    assert flagged <= {48, 49, 50}           # and essentially nothing else


def test_refine_shrinks_poor_segment_to_the_burst():
    sig = _burst_signal()
    # the model's coarse 10 s window flagged [40, 50] poor (burst is only [49, 50])
    ivs = [QI(0, 40, "Q3", 0.9, ()), QI(40, 50, "Q0", 0.4, ("motion",)), QI(50, 60, "Q3", 0.9, ())]
    ref = refine_intervals(ivs, sig, FS, "ecg")
    poor = [iv for iv in ref if iv.tier in ("Q0", "Q1")]
    assert poor, "the artifact should remain flagged"
    span = max(p.end_sec for p in poor) - min(p.start_sec for p in poor)
    assert span <= 4.0                       # localized to ~the burst (was 10 s)
    assert all(p.start_sec >= 46 and p.end_sec <= 52 for p in poor)
    # the reclaimed flank is handed back to the neighbouring good grade
    assert any(iv.tier == "Q3" and iv.start_sec < 48 for iv in ref)


def test_refine_does_not_fabricate_a_trailing_segment():
    """Regression: `fine_badness` bins the WHOLE model-rate signal, but the RLE intervals stop at the last
    full window (the trailing partial window is dropped). Those tail bins must NOT be emitted as a default
    'Q2 acceptable' segment over signal the model never scored."""
    sig = _burst_signal(secs=105, burst=(95.0, 98.0))          # 105 s, but only [0,100] was 'analyzed'
    ivs = [QI(i * 10, (i + 1) * 10, "Q3", 0.9, ()) for i in range(9)] + [QI(90, 100, "Q0", 0.4, ("motion",))]
    ref = refine_intervals(ivs, sig, FS, "ecg")
    assert max(iv.end_sec for iv in ref) <= 100.0 + 1e-6        # nothing past the analyzed span
    assert not any(iv.start_sec >= 100.0 for iv in ref)         # no fabricated [100,105] tail segment


def test_refine_leaves_clean_and_all_good_untouched():
    clean = (0.1 * np.sin(2 * np.pi * 1.1 * np.arange(int(60 * FS)) / FS)).astype(np.float32)
    all_good = [QI(0, 30, "Q3", 0.9, ()), QI(30, 60, "Q2", 0.8, ())]
    assert refine_intervals(all_good, clean, FS, "ecg") == all_good   # no poor → unchanged


def test_refine_keeps_poor_run_with_no_localizable_core():
    # a poor segment whose signal has NO amplitude burst (e.g. diffuse in-band corruption) is trusted
    calm = (0.1 * np.sin(2 * np.pi * 1.1 * np.arange(int(30 * FS)) / FS)).astype(np.float32)
    ivs = [QI(0, 10, "Q3", 0.9, ()), QI(10, 20, "Q1", 0.5, ()), QI(20, 30, "Q3", 0.9, ())]
    ref = refine_intervals(ivs, calm, FS, "ecg")
    assert any(iv.tier == "Q1" for iv in ref)   # not erased
