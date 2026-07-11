"""Export button triggers -> `export.exporters` (design spec (b)/(d)).

Sources the intervals from ``QualitySegmentModel`` (+ human-in-the-loop overrides/notes from
``SelectionController``), plus model provenance/fs from the ``ModelCardModel`` (for the JSON report
and WFDB annotations), and routes a chosen path + format through the ``exporters.EXPORTERS`` registry.
QML export menus call ``exportSelection(fmt)`` -> ``saveRequested`` -> a shared ``FileDialog`` prompts
for a path -> ``exportToPath(path, fmt)``.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlparse

from PySide6.QtCore import Property, QObject, Signal, Slot

from biosqa.export.exporters import EXPORTERS, FORMAT_LABELS


def _local_path(path_or_url: str) -> str:
    """Accept a plain path or a QML ``file://`` URL and return a local filesystem path."""
    if path_or_url.startswith("file:"):
        return unquote(urlparse(path_or_url).path).lstrip("/") if ":" in path_or_url[7:10] \
            else unquote(urlparse(path_or_url).path)
    return path_or_url


def _provenance(model_card) -> dict:
    """Provenance dict from the ModelCardModel's parsed card (empty if none loaded)."""
    card = getattr(model_card, "_card", None) if model_card is not None else None
    if card is None:
        return {}
    return {
        "modality": card.modality,
        "model_version": card.model_version,
        "fs_hz": float(card.fs_hz),
        "L_m": int(card.l_m),
        "normalization": card.normalization.method,
        "training_data_hash": card.training_data_hash,
    }


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
        prov = _provenance(self._model_card)
        if self._guard is not None:                  # record-level acquisition-regime / domain-shift
            prov["domain_shift_index"] = round(float(self._guard.domainShiftIndex), 3)
            prov["regime_flags"] = list(self._guard.regimeFlags)
            prov["novelty_fraction"] = round(float(self._guard.noveltyFraction), 3)
        # WFDB annotations are placed at native-rate sample indices, so prefer the recording's real
        # fs; provenance (JSON) keeps the model fs. Fall back to the model fs if no recording fs.
        native_fs = float(getattr(self._recordings, "currentFsHz", 0.0) or 0.0) if self._recordings else 0.0
        fs = native_fs or prov.get("fs_hz")
        try:
            written = writer(intervals, out, overridden=overridden, notes=notes,
                             corrected=corrected, provenance=prov, fs=fs)
        except Exception as exc:  # noqa: BLE001
            self.exportFailed.emit(str(exc))
            return
        self.exportSucceeded.emit(str(written))

    def _collect_overrides(self, intervals) -> tuple[dict, dict, dict]:
        """Map the human reviews onto interval positions: ``overridden`` (bool), ``notes`` (text),
        and ``corrected`` (the reviewer's NEW tier). Keeping ``corrected`` separate is the fix for
        the relabel-loses-the-tier bug — the writer applies it as the effective exported grade."""
        overridden: dict[int, bool] = {}
        notes: dict[int, str] = {}
        corrected: dict[int, str] = {}
        if self._selection is not None:
            by_start = {round(iv.start_sec, 3): i for i, iv in enumerate(intervals)}
            for start, _end, _orig, new_tier, note in self._selection.collected_overrides():
                i = by_start.get(round(start, 3))
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
