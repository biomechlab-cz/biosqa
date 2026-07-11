"""Selected-segment state -> `QualityInspectorPanel`/`SegmentInspectorView` (design spec (b)/(d)).

Single source of truth for both panels (design spec (b)): whichever
interval the user has clicked in the plot, minimap, or segmentation table
becomes `selectedSegment` here, and both QML surfaces bind to the same
object.
"""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Property, QObject, QStandardPaths, Signal, Slot

from biosqa.inference.segmenter import QualityInterval


def _training_queue_path() -> Path:
    """Durable app-data JSONL sink for reviewer corrections (the active-learning reverse channel)."""
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
    return Path(base or (Path.home() / ".biosqa")) / "training_queue.jsonl"


class SelectedSegment(QObject):
    """QML-facing wrapper around one `QualityInterval` plus human-in-the-loop state.

    Exposed as `selection.selectedSegment` (design spec (d)): "a QObject with
    tier, color, confidence, rationale, artifacts, overridden".
    """

    #: Design spec (c): quality-band colors, kept here so `ConfidenceGauge.qml`/
    #: `QualityInspectorPanel.qml` don't need to duplicate the palette lookup.
    TIER_COLORS = {
        "Q3": "#2FBF71",
        "Q2": "#86C440",
        "Q1": "#E0A32E",
        "Q0": "#E5484D",
    }

    changed = Signal()

    def __init__(self, interval: QualityInterval, rationale: str = "", parent: QObject | None = None):
        super().__init__(parent)
        self._interval = interval
        self._rationale = rationale
        self._artifacts: list[str] = list(interval.artifacts)
        self._overridden = False
        self._corrected_tier = ""   # set by relabel() — the tier the reviewer chose
        self._note = ""
        self._flagged = False       # set by flagForReview()

    def _start(self) -> float:
        return self._interval.start_sec

    startSec = Property(float, _start, notify=changed)

    def _end(self) -> float:
        return self._interval.end_sec

    endSec = Property(float, _end, notify=changed)

    def _note_text(self) -> str:
        return self._note

    note = Property(str, _note_text, notify=changed)

    def _tier(self) -> str:
        return self._interval.tier

    tier = Property(str, _tier, notify=changed)

    def _color(self) -> str:
        return self.TIER_COLORS.get(self._interval.tier, "#9AA4B6")

    color = Property(str, _color, notify=changed)

    def _confidence(self) -> float:
        return self._interval.confidence

    confidence = Property(float, _confidence, notify=changed)

    def _rationale_text(self) -> str:
        return self._rationale

    rationale = Property(str, _rationale_text, notify=changed)

    def _artifacts_list(self) -> list:
        return self._artifacts

    artifacts = Property(list, _artifacts_list, notify=changed)

    def _recoverable_flag(self) -> bool:
        return bool(self._interval.recoverable)

    #: advisory: a standard per-modality filter would likely lift this poor segment to
    #: ``recoveredTier`` (Q2/Q3). Never changes ``tier`` (see :mod:`inference.recover`).
    recoverable = Property(bool, _recoverable_flag, notify=changed)

    def _recovered_tier(self) -> str:
        return self._interval.recovered_tier

    recoveredTier = Property(str, _recovered_tier, notify=changed)

    def _uncertainty(self) -> float:
        return float(getattr(self._interval, "uncertainty", 0.0))

    #: predictive uncertainty (normalized softmax entropy, 0..1); pairs with confidence.
    uncertainty = Property(float, _uncertainty, notify=changed)

    def _conformal_set(self) -> list:
        return [str(t) for t in getattr(self._interval, "conformal_set", ())]

    #: conformal (APS) grade prediction set at the card's calibrated coverage.
    conformalSet = Property("QVariant", _conformal_set, notify=changed)

    def _ambiguous(self) -> bool:
        return len(getattr(self._interval, "conformal_set", ())) >= 2

    #: True when the prediction set holds ≥2 tiers — the model can't commit to one grade at the coverage.
    ambiguous = Property(bool, _ambiguous, notify=changed)

    def _rate_usable(self) -> bool:
        return bool(getattr(self._interval, "rate_usable", False))

    #: task-relative: the RATE is recoverable despite poor morphology (ECG/PPG). ``hrBpm`` = rate.
    rateUsable = Property(bool, _rate_usable, notify=changed)

    def _hr_bpm(self) -> float:
        return float(getattr(self._interval, "hr_bpm", 0.0))

    hrBpm = Property(float, _hr_bpm, notify=changed)

    def _overridden_flag(self) -> bool:
        return self._overridden

    overridden = Property(bool, _overridden_flag, notify=changed)

    def _flagged_flag(self) -> bool:
        return self._flagged

    flagged = Property(bool, _flagged_flag, notify=changed)


_TIER_LABEL = {"Q0": "unacceptable", "Q1": "poor", "Q2": "acceptable", "Q3": "excellent"}


def _rationale_for(interval: QualityInterval) -> str:
    """A short human-readable rationale for a segment (shown in the inspector)."""
    label = _TIER_LABEL.get(interval.tier, interval.tier)
    parts = [f"Graded {interval.tier} ({label}) at {interval.confidence:.0%} confidence over "
             f"{interval.duration_sec:.1f}s."]
    if interval.artifacts:
        parts.append("Artifact types: " + ", ".join(interval.artifacts) + ".")
    return " ".join(parts)


class SelectionController(QObject):
    """Owns the currently-selected segment and human-in-the-loop write-through slots."""

    selectedSegmentChanged = Signal()
    #: (start_sec, end_sec, original_tier, new_tier_or_'', note) — the active-learning
    #: "reverse channel" (Plan 1 §weaksup); ExportController collects these for round-trip.
    overrideRecorded = Signal(float, float, str, str, str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._selected: SelectedSegment | None = None
        self._selected_index = -1  # filtered-table row of the selection (for in-place highlight)
        self._selected_all_index = -1  # index in the FULL interval list (for viewport-card highlight)
        self._segments = None  # QualitySegmentModel, attached in build_engine
        # One human-review record per segment, keyed by (round(start,3), round(end,3)) so accept /
        # relabel / addNote for the SAME segment merge (order-independent) instead of appending
        # duplicates or clobbering each other's fields. value = {"orig", "new_tier", "note"}.
        self._overrides: dict[tuple, dict] = {}

    def attach_segments(self, segments) -> None:
        """Give the controller the segment model so QML can select by row index."""
        self._segments = segments

    def _get_selected_segment(self) -> SelectedSegment | None:
        return self._selected

    selectedSegment = Property(QObject, _get_selected_segment, notify=selectedSegmentChanged)

    def _get_selected_index(self) -> int:
        return self._selected_index

    #: row index of the selected segment (-1 = none) — lets the table/track paint the
    #: currently-selected row/block so a click gives clear in-place feedback.
    selectedIndex = Property(int, _get_selected_index, notify=selectedSegmentChanged)

    def _get_selected_all_index(self) -> int:
        return self._selected_all_index

    #: index of the selection in the FULL interval list (-1 = none) — the viewport segment
    #: cards highlight the active one by comparing against this.
    selectedAllIndex = Property(int, _get_selected_all_index, notify=selectedSegmentChanged)

    def select(self, interval: QualityInterval, rationale: str = "", index: int = -1,
               all_index: int = -1) -> None:
        """Set the current selection (from plot/table/track/card click handlers)."""
        self._selected = SelectedSegment(interval, rationale or _rationale_for(interval), parent=self)
        self._selected_index = index
        self._selected_all_index = all_index
        self.selectedSegmentChanged.emit()

    @Slot(int)
    def selectByIndex(self, row: int) -> None:  # noqa: N802
        """QML entry point: select the interval at ``row`` of the (filtered) segment table."""
        if self._segments is None:
            return
        interval = self._segments.interval_at(row)
        if interval is not None:
            self.select(interval, index=row, all_index=self._segments.all_index_of(interval))

    @Slot(int)
    def selectByAllIndex(self, index: int) -> None:  # noqa: N802
        """QML entry point (viewport cards): select the interval at ``index`` of the FULL list."""
        if self._segments is None:
            return
        interval = self._segments.interval_at_all(index)
        if interval is not None:
            self.select(interval, index=self._segments.filtered_row_of_all(index), all_index=index)

    @Slot()
    def clear(self) -> None:
        """Clear the current selection (e.g. when a new recording is opened)."""
        if self._selected is not None or self._selected_index != -1:
            self._selected = None
            self._selected_index = -1
            self._selected_all_index = -1
            self.selectedSegmentChanged.emit()

    # -- human-in-the-loop write-through (Plan 1 active-learning reverse channel) ----
    def _upsert_override(self, s: "SelectedSegment", *, new_tier: str | None = None,
                         note: str | None = None) -> None:
        """Merge a review action into the single record for ``s`` (created on first touch), then
        re-emit ``overrideRecorded`` with the record's CURRENT full state. ``new_tier``/``note`` are
        applied only when not ``None`` so accept/relabel/addNote don't wipe each other's fields."""
        key = (round(s.startSec, 3), round(s.endSec, 3))
        rec = self._overrides.get(key) or {"orig": s._interval.tier, "new_tier": "", "note": ""}
        if new_tier is not None:
            rec["new_tier"] = new_tier
        if note is not None:
            rec["note"] = note
        self._overrides[key] = rec
        self.overrideRecorded.emit(s.startSec, s.endSec, rec["orig"], rec["new_tier"], rec["note"])

    @Slot()
    def acceptLabel(self) -> None:  # noqa: N802
        """Confirm the model's predicted tier as correct (records a positive review; keeps any note)."""
        if self._selected is None:
            return
        self._upsert_override(self._selected, new_tier="")

    @Slot(str)
    def relabel(self, tier: str) -> None:
        """Manually override the selected segment's tier and record the correction."""
        if self._selected is None:
            return
        s = self._selected
        s._overridden = True
        s._corrected_tier = tier   # remember the reviewer's choice for the training queue
        self._upsert_override(s, new_tier=tier)
        s.changed.emit()

    @Slot(str)
    def addNote(self, text: str) -> None:  # noqa: N802
        """Attach a free-text note to the selected segment. Persisted into the segment's override
        record too, so the note survives to export regardless of whether the tier was relabeled or
        in what order (previously a note added without/after a relabel was silently dropped)."""
        if self._selected is None:
            return
        self._selected._note = text
        self._upsert_override(self._selected, note=text)
        self._selected.changed.emit()

    # -- boundary editor: structural edits to the segmentation (mutate the model live) ----
    #    These call into QualitySegmentModel and re-select the edited piece so the inspector,
    #    plot bands, table and exports all reflect the change. `all_index` is stable across a
    #    move/split/merge for the piece we keep selected.
    def _edited_reselect(self, all_index: int) -> None:
        if self._segments is not None and 0 <= all_index < self._segments._total_count():
            self.selectByAllIndex(all_index)

    @Slot(float)
    def nudgeSelectedStart(self, delta_sec: float) -> None:  # noqa: N802
        """Move the selected segment's START by ``delta_sec`` (reassigns the sliver to/from the
        previous segment). No-op on the first segment."""
        if self._selected is None or self._segments is None:
            return
        i = self._selected_all_index
        new_start = self._selected._interval.start_sec + float(delta_sec)
        if self._segments.move_boundary(i - 1, new_start):
            self._edited_reselect(i)

    @Slot(float)
    def nudgeSelectedEnd(self, delta_sec: float) -> None:  # noqa: N802
        """Move the selected segment's END by ``delta_sec`` (reassigns the sliver to/from the next
        segment). No-op on the last segment."""
        if self._selected is None or self._segments is None:
            return
        i = self._selected_all_index
        new_end = self._selected._interval.end_sec + float(delta_sec)
        if self._segments.move_boundary(i, new_end):
            self._edited_reselect(i)

    @Slot()
    @Slot(float)
    def splitSelected(self, at_sec: float = -1.0) -> None:  # noqa: N802
        """Split the selected segment at ``at_sec`` (default: its midpoint) into two same-grade
        pieces; keeps the LEFT piece selected so it can be reclassified."""
        if self._selected is None or self._segments is None:
            return
        iv = self._selected._interval
        at = float(at_sec) if at_sec and at_sec > 0 else (iv.start_sec + iv.end_sec) / 2.0
        if self._segments.split_interval(self._selected_all_index, at) >= 0:
            self._edited_reselect(self._selected_all_index)

    @Slot()
    def mergeSelectedNext(self) -> None:  # noqa: N802
        """Merge the selected segment with the one after it (absorb the next sliver)."""
        if self._selected is None or self._segments is None:
            return
        if self._segments.merge_with_next(self._selected_all_index):
            self._edited_reselect(self._selected_all_index)

    @Slot(str)
    def reclassifySelected(self, tier: str) -> None:  # noqa: N802
        """Reclassify the selected segment to ``tier`` — updates the band/table/export AND records
        the correction for the training queue (unlike ``relabel``, which is a training-only override
        that leaves the displayed band on the model's grade)."""
        if self._selected is None or self._segments is None:
            return
        original = self._selected._interval.tier
        i = self._selected_all_index
        if self._segments.set_tier(i, tier):
            # record the correction against the ORIGINAL interval bounds before re-selecting
            s = self._selected
            self._upsert_override(s, new_tier=tier)
            self._edited_reselect(i)

    def _write_training_row(self, flagged: bool) -> str:
        """Append the selected segment (current grade + note, corrected tier if relabeled) as one
        JSONL row to the durable app-data training queue; return the file path. ``flagged`` marks a
        'flag for review' entry. This is the real sink for the active-learning reverse channel."""
        if self._selected is None:
            return ""
        s = self._selected
        # model's original grade: survives an in-place reclassify (which mutates iv.tier); "" → the
        # interval was never reclassified, so it equals the current tier.
        model_tier = getattr(s._interval, "model_tier", "") or s._interval.tier
        tier = s._corrected_tier if (s._overridden and s._corrected_tier) else s._interval.tier
        corrected = bool(s._overridden) or (model_tier != s._interval.tier)
        rec = {
            "startSec": float(s.startSec), "endSec": float(s.endSec),
            "tier": tier, "modelTier": model_tier, "corrected": corrected,
            "flagged": bool(flagged),
            "confidence": float(s.confidence), "note": s._note, "artifacts": list(s._artifacts),
        }
        path = _training_queue_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        return str(path)

    @Slot(result=str)
    def saveToTrainingQueue(self) -> str:  # noqa: N802
        """Persist the selected segment's correction to the training queue; return the file path."""
        return self._write_training_row(flagged=False)

    @Slot(str, result=str)
    def flagForReview(self, note: str) -> str:  # noqa: N802
        """Flag the selected segment for review: attach ``note`` and durably append a ``flagged``
        row to the training queue (so the button actually records something), returning the path."""
        if self._selected is None:
            return ""
        if note:
            self.addNote(note)
        self._selected._flagged = True
        return self._write_training_row(flagged=True)

    def collected_overrides(self) -> list[tuple]:
        """All recorded reviews/overrides as ``(start, end, orig_tier, new_tier, note)`` tuples
        (for ExportController). ``new_tier`` is "" for accept-only reviews."""
        return [(k[0], k[1], v["orig"], v["new_tier"], v["note"])
                for k, v in self._overrides.items()]
