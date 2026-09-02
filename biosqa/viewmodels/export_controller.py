"""Export button triggers -> `export.exporters` (design spec (b)/(d)).

Sources the intervals from ``QualitySegmentModel`` (+ human-in-the-loop overrides/notes from
``SelectionController``), plus model provenance/fs from the ``ModelCardModel`` (for the JSON report
and WFDB annotations), and routes a chosen path + format through the ``exporters.EXPORTERS`` registry.
QML export menus call ``exportSelection(fmt)`` -> ``saveRequested`` -> a shared ``FileDialog`` prompts
for a path -> ``exportToPath(path, fmt)``.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Property, QObject, QUrl, Signal, Slot

from biosqa.export.exporters import EXPORTERS, FORMAT_LABELS


def _local_path(path_or_url: str) -> str:
    """Accept a plain path or a QML ``file://`` URL and return a local filesystem path.

    Qt's own converter, not string surgery: the hand-rolled version DROPPED the host of a UNC URL
    (``file://nas01/quality/run.csv`` -> ``quality/run.csv``), so an export aimed at a network share
    silently landed on the local disk. Anything that isn't a ``file:`` URL (a plain path) passes
    through untouched.
    """
    url = QUrl(path_or_url)
    return url.toLocalFile() if url.isLocalFile() else path_or_url


def _provenance(model_card, context=None) -> dict:
    """Provenance from the ModelCardModel's parsed card + the analysis identity (``context``).

    ``analyzed_channel`` is not decoration: the grades in this file describe ONE channel of the
    recording, so an export that doesn't name it cannot be checked against the signal it grades.
    """
    prov: dict = {}
    card = getattr(model_card, "_card", None) if model_card is not None else None
    if card is not None:
        prov.update({
            "modality": card.modality,
            "model_version": card.model_version,
            "fs_hz": float(card.fs_hz),
            "L_m": int(card.l_m),
            "normalization": card.normalization.method,
            "training_data_hash": card.training_data_hash,
        })
    if context is not None and getattr(context, "recording", ""):
        prov["recording"] = context.recording
        prov["analyzed_channel"] = context.channel
        prov["analyzed_channel_index"] = int(context.channel_index)
        prov["analysis_revision"] = int(context.revision)
    return prov


class ExportController(QObject):
    """Owns the Export button's format dispatch; delegates real work to `export.exporters`."""

    exportSucceeded = Signal(str)   # (output_path)
    exportFailed = Signal(str)      # (error_message)
    saveRequested = Signal(str)     # (fmt) -> a shared QML FileDialog opens for this format
    formatsChanged = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._segments = None
        self._selection = None
        self._model_card = None
        self._recordings = None
        self._guard = None

    def attach(self, segments, selection, model_card=None, recordings=None, guard=None) -> None:
        """Give the controller its data sources (called once from build_engine).

        ``model_card`` (the ModelCardModel) feeds JSON provenance; ``recordings`` provides the
        recording's NATIVE sample rate for WFDB annotation sample indices (the model rate would
        misplace them on non-250 Hz records); ``guard`` provides the record-level Domain-Shift Index.
        """
        self._segments = segments
        self._selection = selection
        self._model_card = model_card
        self._recordings = recordings
        self._guard = guard

    # ---- formats exposed to QML -------------------------------------------
    def _formats(self) -> list:
        return [{"fmt": f, "label": FORMAT_LABELS.get(f, f.upper()), "ext": ext}
                for f, (_writer, ext) in EXPORTERS.items()]

    availableFormats = Property("QVariant", _formats, notify=formatsChanged)

    @Slot(result="QVariant")
    def formats(self):  # noqa: N802 - convenience for QML that prefers a call
        return self._formats()

    @Slot(str, result=str)
    def extensionFor(self, fmt: str) -> str:  # noqa: N802
        return EXPORTERS.get(fmt, EXPORTERS["csv"])[1]

    # ---- export flow ------------------------------------------------------
    @Slot(str)
    def exportSelection(self, fmt: str) -> None:  # noqa: N802
        """Ask the UI for a destination path for ``fmt`` (a key of EXPORTERS)."""
        if self._segments is None or not self._segments._all_intervals:
            self.exportFailed.emit("Nothing to export — run inference on a recording first.")
            return
        self.saveRequested.emit(fmt)

    @Slot(str, str)
    def exportToPath(self, path: str, fmt: str) -> None:  # noqa: N802
        """Write the current quality intervals (+ recorded overrides) to ``path`` in ``fmt``."""
        if self._segments is None:
            self.exportFailed.emit("No segments to export.")
            return
        intervals = list(self._segments._all_intervals)
        overridden, notes, corrected = self._collect_overrides(intervals)

        writer, ext = EXPORTERS.get(fmt, EXPORTERS["csv"])
        out = _local_path(path)
        if not out.lower().endswith(ext):
            out += ext
        context = self._analysis_context()
        prov = _provenance(self._model_card, context)
        if self._guard is not None:                  # record-level acquisition-regime / domain-shift
            prov["domain_shift_index"] = round(float(self._guard.domainShiftIndex), 3)
            prov["regime_flags"] = list(self._guard.regimeFlags)
            prov["novelty_fraction"] = round(float(self._guard.noveltyFraction), 3)
        # WFDB annotations are placed at native-rate sample indices, so prefer the recording's real
        # fs; provenance (JSON) keeps the model fs. Fall back to the model fs if no recording fs.
        native_fs = float(getattr(self._recordings, "currentFsHz", 0.0) or 0.0) if self._recordings else 0.0
        fs = native_fs or prov.get("fs_hz")
        # WFDB annotations name a signal INDEX; grade the analyzed channel, not channel 0 (they are
        # not the same channel on e.g. ["RESP", "II"] or any 12-lead ECG). The flat tables
        # (csv/tsv/parquet/mat) name it by NAME, per row — they carry no provenance block, so without
        # it the grades in the file that feeds downstream analysis don't say which signal they grade.
        chan = int(getattr(context, "channel_index", -1)) if context is not None else -1
        chan_name = str(getattr(context, "channel", "") or "") if context is not None else ""
        try:
            written = writer(intervals, out, overridden=overridden, notes=notes,
                             corrected=corrected, provenance=prov, fs=fs,
                             inference_channel=max(0, chan), channel=chan_name)
        except Exception as exc:  # noqa: BLE001
            self.exportFailed.emit(str(exc))
            return
        self.exportSucceeded.emit(str(written))

    def _analysis_context(self):
        """The (recording, graded channel, model, revision) identity of what is currently on screen.
        Owned by the SelectionController (which scopes the human reviews to it); ``None`` when
        nothing has been analyzed."""
        getter = getattr(self._selection, "analysis_context", None) if self._selection else None
        try:
            return getter() if callable(getter) else None
        except Exception:  # noqa: BLE001
            return None

    def _collect_overrides(self, intervals) -> tuple[dict, dict, dict]:
        """Map the human reviews onto interval positions: ``overridden`` (bool), ``notes`` (text),
        and ``corrected`` (the reviewer's NEW tier). Keeping ``corrected`` separate is the fix for
        the relabel-loses-the-tier bug — the writer applies it as the effective exported grade.

        Matched on the FULL (start, end) span, not the start alone: a start time is not an identity
        (every recording has an interval at 0.0), and ``collected_overrides`` is already scoped to
        the current recording/channel/model/revision — a review can only ever land on the exact
        interval it was made against."""
        overridden: dict[int, bool] = {}
        notes: dict[int, str] = {}
        corrected: dict[int, str] = {}
        if self._selection is not None:
            by_span = {(round(iv.start_sec, 3), round(iv.end_sec, 3)): i
                       for i, iv in enumerate(intervals)}
            for start, end, _orig, new_tier, note in self._selection.collected_overrides():
                i = by_span.get((round(start, 3), round(end, 3)))
                if i is None:
                    continue
                if new_tier:
                    overridden[i] = True
                    corrected[i] = new_tier
                if note:
                    notes[i] = note
        return overridden, notes, corrected

    @Slot(str)
    def exportFigure(self, fmt: str) -> None:  # noqa: N802
        """Export the current plot view as `fmt` ("png" | "svg") -- needs a live QQuickWindow (Phase 4)."""
        self.exportFailed.emit("Figure export (PNG/SVG) is not yet available.")
