"""Unit tests for RLE segmentation (Plan 2 §7.3) -- pure numpy/stdlib, no Qt."""

from __future__ import annotations

import numpy as np
import pytest

from biosqa.inference.segmenter import (
    filter_intervals,
    next_interval_after,
    run_length_encode,
    threshold_artifact_labels,
)


def test_run_length_encode_collapses_contiguous_runs():
    tiers = np.array(["Q3", "Q3", "Q1", "Q1", "Q1", "Q3"])
    confidences = np.array([0.9, 0.8, 0.6, 0.7, 0.5, 0.95])
    intervals = run_length_encode(tiers, confidences, window_stride_sec=1.0, window_length_sec=1.0)

    assert [iv.tier for iv in intervals] == ["Q3", "Q1", "Q3"]
    assert intervals[0].start_sec == 0.0
    assert intervals[0].end_sec == 2.0  # 2 windows @ stride 1s + 1s window length
    assert intervals[1].confidence == pytest.approx((0.6 + 0.7 + 0.5) / 3)


def test_run_length_encode_empty_input():
    assert run_length_encode(np.array([]), np.array([]), 1.0, 1.0) == []


def test_run_length_encode_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        run_length_encode(np.array(["Q3"]), np.array([0.5, 0.6]), 1.0, 1.0)


def test_filter_intervals_roles():
    tiers = np.array(["Q0", "Q1", "Q2", "Q3"])
    confidences = np.ones(4)
    intervals = run_length_encode(tiers, confidences, 1.0, 1.0)

    assert [iv.tier for iv in filter_intervals(intervals, "all")] == ["Q0", "Q1", "Q2", "Q3"]
    assert [iv.tier for iv in filter_intervals(intervals, "q0")] == ["Q0"]
    assert [iv.tier for iv in filter_intervals(intervals, "poor")] == ["Q0", "Q1"]
    assert [iv.tier for iv in filter_intervals(intervals, "usable")] == ["Q2", "Q3"]

    with pytest.raises(ValueError):
        filter_intervals(intervals, "bogus")


def test_next_interval_after_finds_next_poor_segment():
    tiers = np.array(["Q3", "Q3", "Q0", "Q0", "Q3", "Q1"])
    confidences = np.ones(6)
    intervals = run_length_encode(tiers, confidences, 1.0, 1.0)

    hit = next_interval_after(intervals, from_sec=0.5, tiers=("Q0",))
    assert hit is not None
    assert hit.tier == "Q0"

    miss = next_interval_after(intervals, from_sec=1000.0, tiers=("Q0",))
    assert miss is None


# --- artifact-type plumbing (Plan 1 §12.1 multilabel head) -------------------

def test_run_length_encode_defaults_to_no_artifacts():
    intervals = run_length_encode(np.array(["Q3", "Q1"]), np.ones(2), 1.0, 1.0)
    assert all(iv.artifacts == () for iv in intervals)


def test_run_length_encode_unions_artifacts_over_run():
    tiers = np.array(["Q3", "Q1", "Q1", "Q3"])
    confidences = np.ones(4)
    per_window = [[], ["motion"], ["motion", "muscle"], []]
    intervals = run_length_encode(tiers, confidences, 1.0, 1.0, artifacts_per_window=per_window)

    assert [iv.tier for iv in intervals] == ["Q3", "Q1", "Q3"]
    assert intervals[0].artifacts == ()
    assert intervals[1].artifacts == ("motion", "muscle")  # order-preserving union
    assert intervals[2].artifacts == ()


def test_run_length_encode_rejects_mismatched_artifacts_length():
    with pytest.raises(ValueError):
        run_length_encode(np.array(["Q3", "Q1"]), np.ones(2), 1.0, 1.0, artifacts_per_window=[[]])


def test_threshold_artifact_labels_drops_clean_and_applies_threshold():
    class_order = ["clean", "motion", "muscle"]
    probs = np.array([[0.9, 0.1, 0.1], [0.1, 0.8, 0.6], [0.2, 0.2, 0.2]])
    labels = threshold_artifact_labels(probs, class_order, threshold=0.5)
    assert labels == [[], ["motion", "muscle"], []]


def test_threshold_artifact_labels_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        threshold_artifact_labels(np.zeros((2, 3)), ["clean", "motion"], threshold=0.5)
