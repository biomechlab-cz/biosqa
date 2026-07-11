"""Tests for recoverability: the filtered-vs-raw second pass (biosqa.inference.recover) and its
run-length aggregation + segment-model surface."""
import numpy as np

from biosqa.inference import recover as rec
from biosqa.inference.segmenter import QualityInterval, filter_intervals, run_length_encode

ORDER = ["Q0", "Q1", "Q2", "Q3"]  # card class_order is worst -> best


def test_filter_is_length_preserving_and_bandlimits():
    fs = 8.0                                   # EDA rate; band is (0.05, 1.5) Hz
    t = np.arange(256) / fs
    low = np.sin(2 * np.pi * 0.3 * t)          # in-band
    high = np.sin(2 * np.pi * 3.0 * t)         # above the 1.5 Hz cutoff -> should be removed
    y = rec.filter_for_modality(low + high, fs, "eda")
    assert y.shape[0] == 256
    # after filtering, the signal is much closer to the in-band component than the raw sum was
    assert np.std(y - low) < np.std((low + high) - low)


def test_filter_unknown_modality_is_passthrough():
    x = np.random.default_rng(0).standard_normal(200).astype(np.float32)
    y = rec.filter_for_modality(x, 250.0, "xyz")
    assert np.allclose(y, x)


def test_recoverable_windows_ordinal_rule():
    raw = ["Q0", "Q1", "Q3", "Q0"]
    filt = ["Q2", "Q1", "Q3", "Q3"]
    r, rtier = rec.recoverable_windows(raw, filt, ORDER)
    assert list(r) == [True, False, False, True]   # poor->usable only for windows 0 and 3
    assert rtier[0] == "Q2" and rtier[3] == "Q3" and rtier[1] == "" and rtier[2] == ""


def test_recoverable_windows_verbose_labels():
    r, rtier = rec.recoverable_windows(
        ["Q0_unacceptable"], ["Q2_acceptable"], ["Q0_unacceptable", "Q1_poor", "Q2_acceptable", "Q3_excellent"])
    assert bool(r[0]) is True and rtier[0] == "Q2"


def test_rle_marks_recoverable_by_majority():
    tiers = np.array(["Q0", "Q0", "Q0"])
    conf = np.array([0.4, 0.4, 0.4])
    ivs = run_length_encode(tiers, conf, 10.0, 10.0,
                            recoverable_per_window=np.array([True, True, False]),
                            recovered_tier_per_window=["Q2", "Q2", ""])
    assert len(ivs) == 1 and ivs[0].recoverable is True and ivs[0].recovered_tier == "Q2"


def test_rle_minority_recoverable_is_not_flagged():
    tiers = np.array(["Q0", "Q0", "Q0"])
    conf = np.array([0.4, 0.4, 0.4])
    ivs = run_length_encode(tiers, conf, 10.0, 10.0,
                            recoverable_per_window=np.array([True, False, False]))
    assert ivs[0].recoverable is False and ivs[0].recovered_tier == ""


def test_rle_length_mismatch_raises():
    import pytest
    with pytest.raises(ValueError):
        run_length_encode(np.array(["Q0", "Q0"]), np.array([0.4, 0.4]), 10.0, 10.0,
                          recoverable_per_window=np.array([True]))


def test_recoverable_filter_role():
    ivs = [
        QualityInterval(0, 10, "Q0", 0.4, (), True, "Q2"),
        QualityInterval(10, 20, "Q1", 0.5, (), False, ""),
    ]
    assert [iv.start_sec for iv in filter_intervals(ivs, "recoverable")] == [0]


def test_segment_model_recoverable_stats():
    from biosqa.viewmodels.quality_segment_model import QualitySegmentModel

    m = QualitySegmentModel()
    m.load_intervals([
        QualityInterval(0, 10, "Q0", 0.4, (), True, "Q2"),   # poor + recoverable
        QualityInterval(10, 20, "Q1", 0.5, (), False, ""),   # poor, not recoverable
        QualityInterval(20, 30, "Q3", 0.9, ()),              # usable
    ])
    assert m.recoverableCount == 1
    assert abs(m.recoverableFraction - 0.5) < 1e-6           # 10s recoverable / 20s poor
