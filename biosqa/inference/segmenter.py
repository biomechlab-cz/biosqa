"""Threshold per-window quality predictions and run-length-encode into intervals (Plan 2 §7.3).

Turns a dense per-window ``Q0..Q3`` prediction track into the sparse
``(t_start, t_end, q_level, confidence)`` interval table that
``QualitySegmentModel`` renders and ``export.exporters`` writes out. This is
pure numpy/stdlib so it is directly unit-testable (see
``tests/test_segmenter.py``) without any ONNX Runtime or Qt dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QualityInterval:
    """One run-length-encoded quality segment.

    ``artifacts`` are the artifact-TYPE tags (Plan 1 §12.1 multilabel head)
    present anywhere in the run — e.g. ``("motion", "muscle")`` — surfaced in the
    segment table's *artifacts* column. Empty for legacy single-head models.
    """

    start_sec: float
    end_sec: float
    tier: str  # one of the card's class_order, e.g. "Q0".."Q3"
    confidence: float  # mean class-probability over the run
    artifacts: tuple[str, ...] = ()
    #: advisory recoverability (see :mod:`inference.recover`): a poor run whose windows become
    #: usable after a standard per-modality filter. ``recovered_tier`` is the filtered grade
    #: ("Q2"/"Q3"). Never changes ``tier`` — the raw grade stays the source of truth.
    recoverable: bool = False
    recovered_tier: str = ""
    #: predictive uncertainty — normalized entropy of the model softmax, 0 (certain) .. 1 (uniform);
    #: mean over the run (research3 Rec.6: ship uncertainty with the score).
    uncertainty: float = 0.0
    #: task-relative usability (see :mod:`inference.task_usability`): a poor-morphology run whose
    #: RATE is still recoverable (beats reliably detected). ``hr_bpm`` = estimated rate.
    rate_usable: bool = False
    hr_bpm: float = 0.0
    #: the model's ORIGINAL grade, preserved when a reviewer reclassifies a segment in place
    #: (empty = never reclassified → same as ``tier``). Keeps export/training-queue provenance honest.
    model_tier: str = ""
    #: split-conformal (APS) prediction set of grade labels at the card's calibrated coverage
    #: (:mod:`inference.conformal`). Size 1 = confident; ≥ 2 (e.g. ("Q2","Q3")) = ambiguous/abstain.
    #: Empty when the model card ships no conformal threshold (feature gracefully absent).
    conformal_set: tuple[str, ...] = ()

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec

    @property
    def ambiguous(self) -> bool:
        """The grade is conformally ambiguous when the prediction set holds more than one tier."""
        return len(self.conformal_set) >= 2


def _union_run_artifacts(
    artifacts_per_window: list[list[str]], start_idx: int, end_idx: int
) -> tuple[str, ...]:
    """Order-preserving union of artifact tags over windows ``[start_idx, end_idx)``."""
    seen: list[str] = []
    for k in range(start_idx, end_idx):
        for tag in artifacts_per_window[k]:
            if tag not in seen:
                seen.append(tag)
    return tuple(seen)


def run_length_encode(
    tiers: np.ndarray,
    confidences: np.ndarray,
    window_stride_sec: float,
    window_length_sec: float,
    artifacts_per_window: list[list[str]] | None = None,
    recoverable_per_window: "np.ndarray | list[bool] | None" = None,
    recovered_tier_per_window: list[str] | None = None,
    uncertainty_per_window: "np.ndarray | list[float] | None" = None,
    rate_usable_per_window: "np.ndarray | list[bool] | None" = None,
    hr_bpm_per_window: "np.ndarray | list[float] | None" = None,
    grade_probs_per_window: "np.ndarray | None" = None,
    class_order: "tuple[str, ...] | list[str] | None" = None,
    conformal_threshold: float | None = None,
) -> list[QualityInterval]:
    """Collapse a per-window tier/confidence track into contiguous intervals.

    Args:
        tiers: 1-D array of per-window predicted tier labels (strings or any
            hashable, e.g. ``["Q3", "Q3", "Q1", "Q1", "Q1", "Q3"]``).
        confidences: 1-D array of per-window confidence in ``[0, 1]``, same
            length as ``tiers``.
        window_stride_sec: time between consecutive window starts.
        window_length_sec: duration covered by one window (used only for the
            final interval's end time).
        artifacts_per_window: optional per-window artifact-type tag lists (from
            the multilabel head via :func:`threshold_artifact_labels`), same
            length as ``tiers``. When given, each interval's ``artifacts`` is the
            order-preserving union of tags over its run; when ``None`` (legacy
            single-head model) every interval gets no artifact tags.

    Returns:
        A list of ``QualityInterval`` covering the whole track, ordered by
        time, with confidence averaged over each run.
    """
    tiers = np.asarray(tiers)
    confidences = np.asarray(confidences, dtype=np.float64)
    if tiers.shape[0] != confidences.shape[0]:
        raise ValueError("tiers and confidences must have the same length")
    n = tiers.shape[0]
    if artifacts_per_window is not None and len(artifacts_per_window) != n:
        raise ValueError("artifacts_per_window must have the same length as tiers")
    if recoverable_per_window is not None and len(recoverable_per_window) != n:
        raise ValueError("recoverable_per_window must have the same length as tiers")
    if recovered_tier_per_window is not None and len(recovered_tier_per_window) != n:
        raise ValueError("recovered_tier_per_window must have the same length as tiers")
    if n == 0:
        return []

    # Indices where the tier changes from the previous window (run boundaries).
    change_points = np.flatnonzero(tiers[1:] != tiers[:-1]) + 1
    run_starts = np.concatenate(([0], change_points))
    run_ends = np.concatenate((change_points, [n]))

    intervals: list[QualityInterval] = []
    for start_idx, end_idx in zip(run_starts, run_ends):
        start_sec = start_idx * window_stride_sec
        end_sec = (end_idx - 1) * window_stride_sec + window_length_sec
        artifacts = (
            _union_run_artifacts(artifacts_per_window, int(start_idx), int(end_idx))
            if artifacts_per_window is not None
            else ()
        )
        # A run is recoverable when the MAJORITY of its windows are (advisory only); the reported
        # recovered tier is the most common filtered grade among those recoverable windows.
        recoverable, recovered_tier = False, ""
        if recoverable_per_window is not None:
            run_rec = np.asarray(recoverable_per_window[int(start_idx):int(end_idx)], dtype=bool)
            if run_rec.size and run_rec.mean() >= 0.5:
                recoverable = True
                if recovered_tier_per_window is not None:
                    cand = [recovered_tier_per_window[k] for k in range(int(start_idx), int(end_idx))
                            if recoverable_per_window[k] and recovered_tier_per_window[k]]
                    if cand:
                        recovered_tier = max(set(cand), key=cand.count)
        uncertainty = 0.0
        if uncertainty_per_window is not None:
            u = np.asarray(uncertainty_per_window[int(start_idx):int(end_idx)], dtype=np.float64)
            uncertainty = float(u.mean()) if u.size else 0.0
        rate_usable, hr_bpm = False, 0.0
        if rate_usable_per_window is not None:
            ru = np.asarray(rate_usable_per_window[int(start_idx):int(end_idx)], dtype=bool)
            if ru.size and ru.mean() >= 0.5:
                rate_usable = True
                if hr_bpm_per_window is not None:
                    hrs = [hr_bpm_per_window[k] for k in range(int(start_idx), int(end_idx))
                           if rate_usable_per_window[k] and hr_bpm_per_window[k] > 0]
                    hr_bpm = float(np.median(hrs)) if hrs else 0.0
        # APS prediction set from the run's MEAN (temperature-scaled) grade distribution — the segment's
        # aggregate confidence. Empty when the card ships no conformal threshold.
        conformal_set: tuple[str, ...] = ()
        if grade_probs_per_window is not None and conformal_threshold is not None and class_order is not None:
            from biosqa.inference.conformal import aps_prediction_set
            gp = np.asarray(grade_probs_per_window[int(start_idx):int(end_idx)], dtype=np.float64)
            if gp.size:
                conformal_set = aps_prediction_set(gp.mean(axis=0), class_order, conformal_threshold)
        intervals.append(
            QualityInterval(
                start_sec=float(start_sec),
                end_sec=float(end_sec),
                tier=str(tiers[start_idx]),
                confidence=float(confidences[start_idx:end_idx].mean()),
                artifacts=artifacts,
                recoverable=recoverable,
                recovered_tier=recovered_tier,
                uncertainty=uncertainty,
                rate_usable=rate_usable,
                hr_bpm=hr_bpm,
                conformal_set=conformal_set,
            )
        )
    return intervals


def threshold_artifact_labels(
    type_probs: np.ndarray,
    class_order: "tuple[str, ...] | list[str]",
    threshold: float = 0.5,
    clean_label: str = "clean",
) -> list[list[str]]:
    """Turn a multilabel artifact head's per-window probabilities into tag lists.

    Args:
        type_probs: ``[n_windows, K]`` sigmoid probabilities from the artifact head.
        class_order: the artifact head's ``class_order`` (length ``K``).
        threshold: a label is emitted when its probability is ``>= threshold``.
        clean_label: the "no artifact" class is never emitted as a tag (a clean
            window simply has an empty tag list).

    Returns:
        One list of artifact-type labels per window (each in ``class_order`` order).
    """
    probs = np.asarray(type_probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != len(class_order):
        raise ValueError(
            f"type_probs shape {probs.shape} does not match class_order length {len(class_order)}"
        )
    order = list(class_order)
    out: list[list[str]] = []
    for row in probs:
        out.append(
            [order[j] for j, p in enumerate(row) if p >= threshold and order[j] != clean_label]
        )
    return out


def filter_intervals(intervals: list[QualityInterval], filter_id: str) -> list[QualityInterval]:
    """Apply a `QualitySegmentModel` filter role: "all" | "q0" | "poor" | "usable" | "recoverable".

    "poor" == Q0 or Q1; "usable" == Q2 or Q3 (design spec quality bands table);
    "recoverable" == poor runs a standard filter would likely lift to usable (advisory overlay).
    """
    if filter_id == "all":
        return list(intervals)
    if filter_id == "q0":
        return [iv for iv in intervals if iv.tier == "Q0"]
    if filter_id == "poor":
        return [iv for iv in intervals if iv.tier in ("Q0", "Q1")]
    if filter_id == "usable":
        return [iv for iv in intervals if iv.tier in ("Q2", "Q3")]
    if filter_id == "recoverable":
        return [iv for iv in intervals if iv.recoverable]
    raise ValueError(f"unknown filter_id {filter_id!r} (expected all|q0|poor|usable|recoverable)")


def next_interval_after(
    intervals: list[QualityInterval], from_sec: float, tiers: tuple[str, ...] = ("Q0",)
) -> QualityInterval | None:
    """Find the next interval whose tier is in ``tiers`` starting after ``from_sec``.

    Backs ``segments.jumpToNextPoor(fromSec)`` (design spec (d)).
    """
    for interval in sorted(intervals, key=lambda iv: iv.start_sec):
        if interval.start_sec > from_sec and interval.tier in tiers:
            return interval
    return None
