"""`QualitySegmentationView.qml` table/track backing model (design spec (b), Plan 2 §7.3).

Wraps the RLE intervals produced by `inference.segmenter` with a filter role
(`all`/`q0`/`poor`/`usable`) consumed by `SegmentTable.qml` and `RunLengthTrack.qml`.
"""

from __future__ import annotations

import bisect
from typing import Any

from PySide6.QtCore import Property, QAbstractTableModel, QModelIndex, Qt, Signal, Slot

from biosqa.inference.segmenter import (
    QualityInterval,
    filter_intervals,
    next_interval_after,
)

COLUMNS = ("startSec", "endSec", "tier", "confidence", "artifacts")


class QualitySegmentModel(QAbstractTableModel):
    """`QAbstractTableModel` over quality intervals, with a live filter role."""

    StartSecRole = Qt.UserRole + 1
    EndSecRole = Qt.UserRole + 2
    TierRole = Qt.UserRole + 3
    ConfidenceRole = Qt.UserRole + 4
    ArtifactsRole = Qt.UserRole + 5
    RecoverableRole = Qt.UserRole + 6
    RecoveredTierRole = Qt.UserRole + 7
    UncertaintyRole = Qt.UserRole + 8
    RateUsableRole = Qt.UserRole + 9
    HrBpmRole = Qt.UserRole + 10

    filterChanged = Signal()
    statsChanged = Signal()  # per-tier fractions / artifact counts changed (overview)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._all_intervals: list[QualityInterval] = []
        self._filter_id = "all"
        self._filtered: list[QualityInterval] = []
        # role -> getter, built ONCE (not a fresh 10-entry dict per data() call, which fires for every
        # (visible row × role) on every model refresh).
        self._role_getters = {
            self.StartSecRole: lambda iv: iv.start_sec,
            self.EndSecRole: lambda iv: iv.end_sec,
            self.TierRole: lambda iv: iv.tier,
            self.ConfidenceRole: lambda iv: iv.confidence,
            self.ArtifactsRole: lambda iv: list(iv.artifacts),   # multilabel head tags; () for single-head
            self.RecoverableRole: lambda iv: iv.recoverable,
            self.RecoveredTierRole: lambda iv: iv.recovered_tier,
            self.UncertaintyRole: lambda iv: iv.uncertainty,
            self.RateUsableRole: lambda iv: iv.rate_usable,
            self.HrBpmRole: lambda iv: iv.hr_bpm,
        }

    # -- aggregate stats for the Overview dashboard (bound in OverviewView.qml) ----
    def _tier_fractions(self):
        """Fraction of total recording duration in each tier Q0..Q3 (for the donut)."""
        total = sum(iv.duration_sec for iv in self._all_intervals)
        if total <= 0:
            return {}
        acc: dict[str, float] = {}
        for iv in self._all_intervals:
            acc[iv.tier] = acc.get(iv.tier, 0.0) + iv.duration_sec
        return {t: acc[t] / total for t in ("Q3", "Q2", "Q1", "Q0") if acc.get(t, 0.0) > 0}

    tierFractions = Property("QVariant", _tier_fractions, notify=statsChanged)

    def _artifact_bars(self):
        """Artifact-type occurrence counts across all intervals (for ArtifactBars)."""
        counts: dict[str, int] = {}
        for iv in self._all_intervals:
            for tag in iv.artifacts:
                counts[tag] = counts.get(tag, 0) + 1
        return [{"label": k, "value": v} for k, v in sorted(counts.items(), key=lambda kv: -kv[1])]

    artifactBars = Property("QVariant", _artifact_bars, notify=statsChanged)

    def _segment_bands(self):
        """Normalized (0..1) quality bands over the recording, for QualitySparkline
        (channel-row sparkline, minimap ribbon, per-modality overview ribbon)."""
        total = self._all_intervals[-1].end_sec if self._all_intervals else 0.0
        if total <= 0:
            return []
        return [{"start": iv.start_sec / total,
                 "width": (iv.end_sec - iv.start_sec) / total,
                 "tier": iv.tier} for iv in self._all_intervals]

    segmentBands = Property("QVariant", _segment_bands, notify=statsChanged)

    def _total_duration(self):
        return self._all_intervals[-1].end_sec if self._all_intervals else 0.0

    totalDurationSec = Property(float, _total_duration, notify=statsChanged)

    def _total_count(self):
        """Unfiltered segment count (the Overview KPI; ``rowCount`` reflects the filter)."""
        return len(self._all_intervals)

    totalCount = Property(int, _total_count, notify=statsChanged)

    def _filter_id_get(self):
        return self._filter_id

    #: the ACTIVE filter role — the toolbar binds its pill highlight to this so it can't desync from
    #: the (persistent) model when the segmentation view is recreated by the Loader.
    filterId = Property(str, _filter_id_get, notify=filterChanged)

    def _recoverable_count(self):
        """Number of poor segments a standard filter would likely make usable (advisory)."""
        return sum(1 for iv in self._all_intervals if iv.recoverable)

    recoverableCount = Property(int, _recoverable_count, notify=statsChanged)

    def _recoverable_fraction(self):
        """Of the POOR (Q0/Q1) recording duration, the fraction flagged recoverable-by-filtering.
        0.0 when there is no poor signal. Drives the Overview 'X% recoverable' stat."""
        poor = sum(iv.duration_sec for iv in self._all_intervals if iv.tier in ("Q0", "Q1"))
        if poor <= 0:
            return 0.0
        rec = sum(iv.duration_sec for iv in self._all_intervals if iv.recoverable)
        return min(1.0, rec / poor)

    recoverableFraction = Property(float, _recoverable_fraction, notify=statsChanged)

    # -- QAbstractTableModel plumbing ---------------------------------------
    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if parent.isValid():
            return 0
        return len(self._filtered)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if parent.isValid():
            return 0
        return len(COLUMNS)

    def roleNames(self) -> dict[int, bytes]:  # noqa: N802
        return {
            self.StartSecRole: b"startSec",
            self.EndSecRole: b"endSec",
            self.TierRole: b"tier",
            self.ConfidenceRole: b"confidence",
            self.ArtifactsRole: b"artifacts",
            self.RecoverableRole: b"recoverable",
            self.RecoveredTierRole: b"recoveredTier",
            self.UncertaintyRole: b"uncertainty",
            self.RateUsableRole: b"rateUsable",
            self.HrBpmRole: b"hrBpm",
        }

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:  # noqa: N802
        if not index.isValid() or not (0 <= index.row() < len(self._filtered)):
            return None
        getter = self._role_getters.get(role)
        return getter(self._filtered[index.row()]) if getter is not None else None

    # -- QML-invokable API (design spec (d): "segments.setFilter(filterId)",
    # "segments.jumpToNextPoor(fromSec)") ------------------------------------
    @Slot(str)
    def setFilter(self, filter_id: str) -> None:  # noqa: N802
        """Apply a filter role: "all" | "q0" | "poor" | "usable" (Plan 2 §8.2)."""
        self.beginResetModel()
        self._filter_id = filter_id
        self._filtered = filter_intervals(self._all_intervals, filter_id)
        self.endResetModel()
        self.filterChanged.emit()

    @Slot(float, result="QVariant")
    def jumpToNextPoor(self, from_sec: float) -> float | None:  # noqa: N802
        """Return the start time of the next Q0/Q1 interval after `from_sec`, or None."""
        interval = next_interval_after(self._all_intervals, from_sec, tiers=("Q0", "Q1"))
        return interval.start_sec if interval is not None else None

    @Slot(float, result=int)
    def nextPoorIndex(self, from_sec: float) -> int:  # noqa: N802
        """FULL-list index of the earliest Q0/Q1 interval starting after ``from_sec`` (-1 if none) —
        so the toolbar can SELECT it (highlighting it in the table/track/plot), not just pan."""
        best_i, best_start = -1, None
        for i, iv in enumerate(self._all_intervals):
            if iv.start_sec > from_sec and iv.tier in ("Q0", "Q1"):
                if best_start is None or iv.start_sec < best_start:
                    best_i, best_start = i, iv.start_sec
        return best_i

    @Slot(float, result="QVariant")
    def segmentAt(self, sec: float) -> Any:  # noqa: N802
        """The interval covering time ``sec`` as ``{tier, confidence, startSec, endSec}``
        (for the plot hover tooltip), or ``None`` if outside all intervals. Intervals are sorted +
        contiguous, so bisect on start_sec (O(log N)) instead of a linear scan on every mouse-move."""
        ivs = self._all_intervals
        if not ivs:
            return None
        pos = bisect.bisect_right(ivs, sec, key=lambda iv: iv.start_sec)   # last interval with start<=sec
        if pos == 0:
            return None                                    # before the first segment
        iv = ivs[pos - 1]
        if iv.start_sec <= sec < iv.end_sec:
            return {"tier": iv.tier, "confidence": iv.confidence,
                    "startSec": iv.start_sec, "endSec": iv.end_sec}
        return None                                        # in a gap / past the last segment

    @Slot(float, float, result="QVariant")
    def segmentsInRange(self, start: float, end: float) -> Any:  # noqa: N802
        """All segments overlapping the time window ``[start, end)`` as a list of
        ``{index, tier, confidence, startSec, endSec, artifacts}`` (``index`` is the
        position in the FULL, unfiltered interval list — feed it to
        ``selection.selectByAllIndex``). Drives the viewport segment cards in the
        AI Quality Inspector, refreshed as the plot pans/zooms."""
        out = []
        for i, iv in enumerate(self._all_intervals):
            if iv.start_sec < end and iv.end_sec > start:
                out.append({
                    "index": i, "tier": iv.tier, "confidence": iv.confidence,
                    "startSec": iv.start_sec, "endSec": iv.end_sec,
                    "artifacts": list(iv.artifacts),
                    "recoverable": iv.recoverable, "recoveredTier": iv.recovered_tier,
                    "uncertainty": iv.uncertainty, "rateUsable": iv.rate_usable, "hrBpm": iv.hr_bpm,
                    "conformalSet": list(iv.conformal_set), "ambiguous": len(iv.conformal_set) >= 2,
                })
        return out

    def interval_at(self, row: int) -> QualityInterval | None:
        """The (filtered) interval at ``row``, for the selection controller."""
        if 0 <= row < len(self._filtered):
            return self._filtered[row]
        return None

    def interval_at_all(self, index: int) -> QualityInterval | None:
        """The interval at ``index`` in the FULL (unfiltered) list — for card selection."""
        if 0 <= index < len(self._all_intervals):
            return self._all_intervals[index]
        return None

    def all_index_of(self, interval: QualityInterval) -> int:
        """Position of ``interval`` in the full list (-1 if absent) — keeps card/table highlight in sync."""
        try:
            return self._all_intervals.index(interval)
        except ValueError:
            return -1

    def filtered_row_of_all(self, index: int) -> int:
        """Filtered-table row for full-list ``index`` (-1 if the interval is filtered out)."""
        iv = self.interval_at_all(index)
        if iv is None:
            return -1
        try:
            return self._filtered.index(iv)
        except ValueError:
            return -1

    def load_intervals(self, intervals: list[QualityInterval]) -> None:
        """Replace the full (unfiltered) interval set, e.g. once inference completes.

        Not QML-facing directly -- called from the `InferenceWorkerSignals.
        intervalsReady` handler (main thread).
        """
        self.beginResetModel()
        self._all_intervals = list(intervals)
        self._filtered = filter_intervals(self._all_intervals, self._filter_id)
        self.endResetModel()
        self.statsChanged.emit()

    # -- manual segment edits (boundary editor; SelectionController drives these) ----------
    #    Each mutates the FULL interval list in place, keeping it contiguous & sorted, then
    #    refilters + resets the model so plot bands, the table, the overview, and exports all
    #    update live. A subsequent re-inference (load_intervals) discards manual edits, exactly
    #    like relabel — edits are a human overlay on the current segmentation.
    MIN_SEGMENT_SEC = 0.1

    def _commit(self) -> None:
        self.beginResetModel()
        self._filtered = filter_intervals(self._all_intervals, self._filter_id)
        self.endResetModel()
        self.statsChanged.emit()

    def move_boundary(self, index: int, new_boundary_sec: float) -> bool:
        """Move the boundary between segment ``index`` and ``index+1`` to ``new_boundary_sec``
        (a drag of their shared edge). Reassigns the straddling sliver from one to the other;
        tiers are unchanged. Clamped so neither side shrinks below ``MIN_SEGMENT_SEC``."""
        import dataclasses
        if not (0 <= index < len(self._all_intervals) - 1):
            return False
        left, right = self._all_intervals[index], self._all_intervals[index + 1]
        lo = left.start_sec + self.MIN_SEGMENT_SEC
        hi = right.end_sec - self.MIN_SEGMENT_SEC
        t = max(lo, min(hi, float(new_boundary_sec)))
        if hi < lo:                                   # both too short to move a boundary between
            return False
        self._all_intervals[index] = dataclasses.replace(left, end_sec=t)
        self._all_intervals[index + 1] = dataclasses.replace(right, start_sec=t)
        self._commit()
        return True

    def split_interval(self, index: int, at_sec: float) -> int:
        """Split segment ``index`` at ``at_sec`` into two same-grade pieces. Returns the index of
        the new RIGHT piece (``index+1``), or -1 if ``at_sec`` isn't strictly inside the segment."""
        import dataclasses
        if not (0 <= index < len(self._all_intervals)):
            return -1
        iv = self._all_intervals[index]
        t = float(at_sec)
        if not (iv.start_sec + self.MIN_SEGMENT_SEC <= t <= iv.end_sec - self.MIN_SEGMENT_SEC):
            return -1
        left = dataclasses.replace(iv, end_sec=t)
        right = dataclasses.replace(iv, start_sec=t)
        self._all_intervals[index:index + 1] = [left, right]
        self._commit()
        return index + 1

    def merge_with_next(self, index: int) -> bool:
        """Merge segment ``index`` with ``index+1`` into one spanning both; the merged grade +
        confidence + artifacts come from the LONGER of the two (so absorbing a small sliver keeps
        the dominant grade). Directly answers 'shorten so only the Q0 stays'."""
        import dataclasses
        if not (0 <= index < len(self._all_intervals) - 1):
            return False
        a, b = self._all_intervals[index], self._all_intervals[index + 1]
        keep = a if a.duration_sec >= b.duration_sec else b
        merged = dataclasses.replace(
            keep, start_sec=a.start_sec, end_sec=b.end_sec,
            artifacts=tuple(dict.fromkeys(a.artifacts + b.artifacts)),
        )
        self._all_intervals[index:index + 2] = [merged]
        self._commit()
        return True

    def set_tier(self, index: int, tier: str) -> bool:
        """Reclassify segment ``index`` to ``tier`` (updates the band/table/export immediately).
        Preserves the model's ORIGINAL grade in ``model_tier`` the first time, so a reclassify never
        erases provenance (exports/training queue still report what the model said)."""
        import dataclasses
        if not (0 <= index < len(self._all_intervals)) or not tier:
            return False
        iv = self._all_intervals[index]
        self._all_intervals[index] = dataclasses.replace(
            iv, tier=str(tier), model_tier=(iv.model_tier or iv.tier))
        self._commit()
        return True
