"""Threshold per-window quality predictions and run-length-encode into intervals (Plan 2 §7.3).

Turns a dense per-window ``Q0..Q3`` prediction track into the sparse
``(t_start, t_end, q_level, confidence)`` interval table that
``QualitySegmentModel`` renders and ``export.exporters`` writes out. This is
pure numpy/stdlib so it is directly unit-testable (see
``tests/test_segmenter.py``) without any ONNX Runtime or Qt dependency.
"""

from __future__ import annotations

from collections.abc import Sequence
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


def _start_times(
    n: int, window_stride_sec: float, window_starts_sec: Sequence[float] | None
) -> np.ndarray:
    """Per-window START TIMES: the REAL ones when the caller has them, else the uniform grid.

    ``preprocess.make_windows`` END-ANCHORS its final window when the record is not a whole number of
    windows (so the tail is graded), which puts that window OFF the ``i * stride`` grid. Assuming the
    grid then attributes the tail window's grade to a span that runs past the end of the recording --
    by up to one full stride (60 s for EDA at overlap 0). Callers that know the true starts pass them.
    """
    if window_starts_sec is None:
        return np.arange(n, dtype=np.float64) * float(window_stride_sec)
    starts = np.asarray(window_starts_sec, dtype=np.float64).reshape(-1)
    if starts.shape[0] != n:
        raise ValueError("window_starts_sec must have the same length as tiers")
    return starts


def _run_bounds(
    starts: np.ndarray, window_length_sec: float, start_idx: int, end_idx: int, n: int
) -> tuple[float, float]:
    """``(start_sec, end_sec)`` of the run over windows ``[start_idx, end_idx)`` on an ARBITRARY
    (possibly non-uniform) start grid.

    The boundary between the run ending at window ``e-1`` and the run starting at window ``e`` belongs
    at the MIDPOINT OF THE AMBIGUOUS ZONE ``[starts[e], starts[e-1] + L]`` -- the zone both windows
    cover and disagree about. That makes adjacent intervals contiguous and non-overlapping (no span is
    claimed by two tiers, no duration double-counted). When the windows do NOT overlap the zone is
    empty and the plain window edges are used, so overlap 0 is a no-op. On a uniform grid
    (``starts[i] = i * stride``) the midpoint reduces exactly to ``e * stride + (L - stride) / 2``.
    The track's outer edges are never trimmed: the first run starts at ``starts[0]``, the last ends at
    ``starts[-1] + L`` -- i.e. at the true end of the analyzed signal, never past it.
    """
    length = float(window_length_sec)

    def _boundary(e: int) -> tuple[float, float]:
        prev_end = float(starts[e - 1]) + length  # end of the last window of the earlier run
        next_start = float(starts[e])             # start of the first window of the later run
        if next_start >= prev_end:                # windows do not overlap -> no ambiguous zone
            return prev_end, next_start
        mid = 0.5 * (next_start + prev_end)
        return mid, mid

    start_sec = float(starts[0]) if start_idx == 0 else _boundary(int(start_idx))[1]
    end_sec = (
        float(starts[n - 1]) + length if end_idx == n else _boundary(int(end_idx))[0]
    )
    return start_sec, end_sec


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
    window_starts_sec: Sequence[float] | None = None,
) -> list[QualityInterval]:
    """Collapse a per-window tier/confidence track into contiguous intervals.

    Args:
        tiers: 1-D array of per-window predicted tier labels (strings or any
            hashable, e.g. ``["Q3", "Q3", "Q1", "Q1", "Q1", "Q3"]``).
        confidences: 1-D array of per-window confidence in ``[0, 1]``, same
            length as ``tiers``.
        window_stride_sec: time between consecutive window starts. Used ONLY to
            synthesize the uniform start grid ``i * stride`` when
            ``window_starts_sec`` is not given.
        window_length_sec: duration covered by one window. When consecutive
            windows overlap, run boundaries are placed at the midpoint of the
            shared (ambiguous) zone so the returned intervals stay contiguous
            and non-overlapping.
        artifacts_per_window: optional per-window artifact-type tag lists (from
            the multilabel head via :func:`threshold_artifact_labels`), same
            length as ``tiers``. When given, each interval's ``artifacts`` is the
            order-preserving union of tags over its run; when ``None`` (legacy
            single-head model) every interval gets no artifact tags.
        window_starts_sec: the ACTUAL start time of every window (from
            :func:`preprocess.window_starts`), same length as ``tiers``. The
            window grid is NOT uniform when the record does not tile evenly --
            ``make_windows`` end-anchors the final window so the tail is graded --
            so a caller that has the real starts must pass them, or the tail
            window's grade is attributed to a span running past the end of the
            recording. Defaults to the uniform ``i * window_stride_sec`` grid.

    Returns:
        A list of ``QualityInterval`` covering the whole analyzed span, ordered by
        time, with confidence averaged over each run. The last interval ends at
        ``starts[-1] + window_length_sec`` -- never past the analyzed signal.
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

    # Runs are bounded at DECISION MIDPOINTS on the window grid the model ACTUALLY used (see
    # `_run_bounds` / `_start_times`): overlapping windows share an ambiguous zone that naive bounds
    # would claim TWICE, and the end-anchored tail window is not on the uniform grid at all.
    starts = _start_times(n, window_stride_sec, window_starts_sec)

    intervals: list[QualityInterval] = []
    for start_idx, end_idx in zip(run_starts, run_ends):
        start_sec, end_sec = _run_bounds(
            starts, window_length_sec, int(start_idx), int(end_idx), n
        )
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


def window_intervals(
    tiers: np.ndarray,
    confidences: np.ndarray,
    window_starts_sec: Sequence[float],
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
    """One ``QualityInterval`` PER MODEL WINDOW -- the model's raw, still-OVERLAPPING statements.

    :func:`run_length_encode` deliberately collapses these: it resolves a time that several windows
    cover into ONE displayed grade (the run boundary sits at the midpoint of the ambiguous zone), so
    its output is non-overlapping and the fact that the model graded a given second more than once is
    gone. :mod:`inference.refine` needs exactly that discarded fact -- the SET of grades the model gave
    the windows covering a time -- to know how far a poor boundary may honestly be tightened, so it
    reads these, not the RLE output.

    Each window's interval spans ``[start, start + window_length_sec]`` and carries only that window's
    own numbers (its confidence, uncertainty, artifact tags, ...), never a run aggregate. Windows
    overlap whenever the caller used ``overlap > 0``; the list is in window order.
    """
    tiers = np.asarray(tiers)
    confidences = np.asarray(confidences, dtype=np.float64)
    n = tiers.shape[0]
    if confidences.shape[0] != n:
        raise ValueError("tiers and confidences must have the same length")
    starts = np.asarray(window_starts_sec, dtype=np.float64).reshape(-1)
    if starts.shape[0] != n:
        raise ValueError("window_starts_sec must have the same length as tiers")

    out: list[QualityInterval] = []
    for i in range(n):
        conformal_set: tuple[str, ...] = ()
        if grade_probs_per_window is not None and conformal_threshold is not None and class_order is not None:
            from biosqa.inference.conformal import aps_prediction_set
            conformal_set = aps_prediction_set(
                np.asarray(grade_probs_per_window[i], dtype=np.float64), class_order, conformal_threshold
            )
        out.append(
            QualityInterval(
                start_sec=float(starts[i]),
                end_sec=float(starts[i]) + float(window_length_sec),
                tier=str(tiers[i]),
                confidence=float(confidences[i]),
                artifacts=tuple(artifacts_per_window[i]) if artifacts_per_window is not None else (),
                recoverable=bool(recoverable_per_window[i]) if recoverable_per_window is not None else False,
                recovered_tier=(
                    str(recovered_tier_per_window[i]) if recovered_tier_per_window is not None else ""
                ),
                uncertainty=float(uncertainty_per_window[i]) if uncertainty_per_window is not None else 0.0,
                rate_usable=bool(rate_usable_per_window[i]) if rate_usable_per_window is not None else False,
                hr_bpm=float(hr_bpm_per_window[i]) if hr_bpm_per_window is not None else 0.0,
                conformal_set=conformal_set,
            )
        )
    return out


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
