"""Unit tests for RLE segmentation (Plan 2 §7.3) -- pure numpy/stdlib, no Qt."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from biosqa.inference.preprocess import window_starts
from biosqa.inference.segmenter import (
    filter_intervals,
    next_interval_after,
    run_length_encode,
    threshold_artifact_labels,
    window_intervals,
)
from biosqa.model.model_card import ModelCard, Normalization


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


# --- overlapping windows (stride < length, the shipped windowOverlap=0.5 default) -------------

def test_run_length_encode_overlapping_runs_do_not_overlap_in_time():
    # 5 windows @ stride 5s / length 10s -> analyzed span (5-1)*5 + 10 = 30s. Naive bounds would
    # emit Q3 [0,15] / Q1 [10,25] / Q3 [20,30]: 10-15s claimed by two tiers, 40s of duration over a
    # 30s span. Boundaries belong at the midpoint of the shared zone instead.
    tiers = np.array(["Q3", "Q3", "Q1", "Q1", "Q3"])
    confidences = np.ones(5)
    intervals = run_length_encode(tiers, confidences, window_stride_sec=5.0, window_length_sec=10.0)

    assert [iv.tier for iv in intervals] == ["Q3", "Q1", "Q3"]
    for k in range(len(intervals) - 1):
        assert intervals[k].end_sec <= intervals[k + 1].start_sec

    bounds = [(iv.start_sec, iv.end_sec) for iv in intervals]
    assert bounds == [(0.0, 12.5), (12.5, 22.5), (22.5, 30.0)]

    # No double counting and no gaps: the durations tile the analyzed span exactly.
    analyzed_span = (len(tiers) - 1) * 5.0 + 10.0
    assert sum(iv.duration_sec for iv in intervals) == pytest.approx(analyzed_span)
    assert intervals[0].start_sec == 0.0
    assert intervals[-1].end_sec == pytest.approx(analyzed_span)


@pytest.mark.parametrize("stride_sec", [10.0, 7.5, 5.0, 2.5, 1.0])
def test_run_length_encode_tiles_analyzed_span_at_any_overlap(stride_sec):
    tiers = np.array(["Q3", "Q2", "Q2", "Q0", "Q0", "Q0", "Q3"])
    confidences = np.ones(len(tiers))
    intervals = run_length_encode(tiers, confidences, stride_sec, window_length_sec=10.0)

    for k in range(len(intervals) - 1):
        assert intervals[k].end_sec <= intervals[k + 1].start_sec
        assert intervals[k].duration_sec > 0.0

    analyzed_span = (len(tiers) - 1) * stride_sec + 10.0
    assert sum(iv.duration_sec for iv in intervals) == pytest.approx(analyzed_span)
    assert intervals[0].start_sec == 0.0
    assert intervals[-1].end_sec == pytest.approx(analyzed_span)


def test_run_length_encode_single_run_with_overlap_spans_whole_track():
    # Nothing to split when there is no interior boundary: the run keeps the full analyzed span.
    intervals = run_length_encode(np.array(["Q3"] * 5), np.ones(5), 5.0, 10.0)
    assert len(intervals) == 1
    assert (intervals[0].start_sec, intervals[0].end_sec) == (0.0, 30.0)


def test_run_length_encode_zero_overlap_is_unaffected_by_midpoint_bounding():
    # stride == length -> the shared zone is empty, so bounds are the plain window edges.
    tiers = np.array(["Q3", "Q3", "Q1", "Q3"])
    intervals = run_length_encode(tiers, np.ones(4), 10.0, 10.0)
    assert [(iv.start_sec, iv.end_sec) for iv in intervals] == [(0.0, 20.0), (20.0, 30.0), (30.0, 40.0)]


# --- the REAL (non-uniform) window grid: make_windows end-anchors the final window ------------

ECG = ModelCard(                       # the shipped ECG card: L_m=2500 @ 250 Hz == a 10 s window
    modality="ecg", l_m=2500, fs_hz=250.0, class_order=("Q0", "Q1", "Q2", "Q3"),
    normalization=Normalization(method="none"), training_data_hash="sha256:test",
    model_version="test", source_path=Path("ecg.model_card.json"),
)


def _real_starts_sec(n_samples: int, overlap: float) -> np.ndarray:
    """The start times the REAL windower produces -- not a hand-typed grid."""
    return window_starts(n_samples, ECG, overlap).astype(np.float64) / float(ECG.fs_hz)


@pytest.mark.parametrize("overlap", [0.0, 0.25, 0.5])
def test_no_interval_ends_past_the_record_when_the_tail_window_is_end_anchored(overlap):
    """A 61.2 s record is 6.12 windows. `make_windows` END-ANCHORS the final window at [51.2, 61.2] so
    the tail is graded -- which puts that window OFF the `i * stride` grid. Bounding the runs on the
    grid attributes its grade to [60, 70] (overlap 0) / [55, 65] (overlap 0.5): graded signal claimed
    past the end of a recording that stops at 61.2 s. The mis-shift is up to one full stride."""
    n = 15300                                             # 61.2 s @ 250 Hz
    starts = _real_starts_sec(n, overlap)
    assert starts[-1] == pytest.approx(51.2)              # end-anchored: NOT on the i*stride grid
    tiers = np.array(["Q3"] * (len(starts) - 1) + ["Q0"])  # only the tail window is poor
    ivs = run_length_encode(tiers, np.ones(len(starts)), 10.0 * (1.0 - overlap), 10.0,
                            window_starts_sec=starts)

    assert max(iv.end_sec for iv in ivs) == pytest.approx(61.2)   # the record end, exactly
    assert ivs[0].start_sec == 0.0
    for a, b in zip(ivs, ivs[1:]):                                # contiguous + non-overlapping
        assert a.end_sec == pytest.approx(b.start_sec)
    assert sum(iv.duration_sec for iv in ivs) == pytest.approx(61.2)
    poor = [iv for iv in ivs if iv.tier == "Q0"]
    assert len(poor) == 1 and poor[0].end_sec == pytest.approx(61.2)   # not 70.0 / 65.0


@pytest.mark.parametrize("overlap", [0.0, 0.25, 0.5])
def test_explicit_uniform_starts_reduce_exactly_to_the_stride_formula(overlap):
    """The general midpoint rule must be a strict generalization: on a grid that IS uniform (a record
    that tiles evenly) passing the real starts must reproduce the scalar-stride bounds bit for bit --
    so the non-overlapping guarantee and the overlap-0 no-op both survive the change."""
    n = 2500 + 7500 * 2                    # 70 s: tiles evenly at all three strides (2500/1875/1250)
    starts = _real_starts_sec(n, overlap)
    stride = 10.0 * (1.0 - overlap)
    assert starts.tolist() == [i * stride for i in range(len(starts))]   # no end-anchored window

    cycle = ["Q3", "Q0", "Q0", "Q2", "Q3", "Q3"]
    tiers = np.array([cycle[i % len(cycle)] for i in range(len(starts))])
    conf = np.linspace(0.5, 0.9, len(starts))
    with_starts = run_length_encode(tiers, conf, stride, 10.0, window_starts_sec=starts)
    scalar = run_length_encode(tiers, conf, stride, 10.0)
    assert with_starts == scalar


def test_run_length_encode_rejects_starts_that_do_not_describe_these_grades():
    with pytest.raises(ValueError):
        run_length_encode(np.array(["Q3", "Q0"]), np.ones(2), 10.0, 10.0,
                          window_starts_sec=[0.0, 10.0, 20.0])


# --- the model's PER-WINDOW statements (what RLE collapses away, and refine needs) -------------

def test_window_intervals_keeps_the_overlap_rle_resolves_away():
    """`run_length_encode` hands every time to exactly ONE run, so its output can never show that the
    model graded a second twice. `window_intervals` keeps that: one interval per window, overlapping."""
    n = 2500 * 6
    starts = _real_starts_sec(n, 0.5)
    tiers = np.array(["Q3", "Q0", "Q0", "Q3", "Q3", "Q3", "Q3", "Q3", "Q3", "Q3", "Q3"])
    conf = np.full(len(starts), 0.8)
    assert len(starts) == len(tiers)

    wins = window_intervals(tiers, conf, starts, 10.0)
    assert [(w.start_sec, w.end_sec) for w in wins[:3]] == [(0.0, 10.0), (5.0, 10.0 + 5.0), (10.0, 20.0)]
    assert wins[0].end_sec > wins[1].start_sec           # they OVERLAP -- the whole point
    assert [w.tier for w in wins] == list(tiers)
    assert all(w.confidence == 0.8 for w in wins)

    rle = run_length_encode(tiers, conf, 5.0, 10.0, window_starts_sec=starts)
    for a, b in zip(rle, rle[1:]):                       # ... whereas RLE output does NOT overlap
        assert a.end_sec <= b.start_sec
    # t=7.0 s sits in BOTH window 0 (Q3) and window 1 (Q0): the model said two things, RLE shows one
    assert [w.tier for w in wins if w.start_sec <= 7.0 < w.end_sec] == ["Q3", "Q0"]
    assert len([iv for iv in rle if iv.start_sec <= 7.0 < iv.end_sec]) == 1


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
