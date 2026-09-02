"""`FileTreePanel`/`FileTree.qml` backing model: open/transcode files, recent recordings.

Design spec (b): "open/transcode files, list recent recordings".
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PySide6.QtCore import Property, QAbstractListModel, QModelIndex, Qt, Signal, Slot

from biosqa.io.loaders import RecordingHandle, modality_vote, open_recording

#: below this, an AUTO-detected modality vote is treated as a guess (modality_vote caps the fs-tie-break
#: confidence at ~0.35), and the user is prompted to verify rather than shown authoritative grades.
_MODALITY_CONF_FLOOR = 0.4

#: LRU bound on cached open handles. MNE (read_raw(preload=False)) keeps a file descriptor open for lazy
#: reads, so an unbounded cache leaks fds over a long session; evicted MNE/zarr backends are closed.
_MAX_OPEN_HANDLES = 8


@dataclass
class RecordingEntry:
    """One row: a recording available in the file tree."""

    name: str
    path: str
    modality: str = ""
    duration_sec: float = 0.0
    transcoded: bool = False


class RecordingListModel(QAbstractListModel):
    """`QAbstractListModel` over recently-opened / discovered recordings.

    Roles: ``name``, ``path``, ``modality``, ``durationSec``, ``transcoded``.
    """

    NameRole = Qt.UserRole + 1
    PathRole = Qt.UserRole + 2
    ModalityRole = Qt.UserRole + 3
    DurationRole = Qt.UserRole + 4
    TranscodedRole = Qt.UserRole + 5

    recordingOpened = Signal(str)  # emitted once a recording finishes opening
    openFailed = Signal(str, str)  # (path, error_message)
    #: (used_modality, detected_modality, confidence) — the auto-detect vote ALWAYS runs in the
    #: background; this fires when the user forced a modality the header confidently disagrees with.
    modalityMismatch = Signal(str, str, float)
    #: (used_modality, confidence) — on the AUTO path (no forced modality) the vote was low-confidence
    #: (effectively a guess); fires so the UI can prompt the user to verify the signal type.
    modalityUncertain = Signal(str, float)
    countChanged = Signal()
    currentChanged = Signal()      # the "active" recording (for the top-bar chip)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._entries: list[RecordingEntry] = []
        # path -> (lazily-opened handle, detected modality); the Coordinator reads
        # this on `recordingOpened` to run inference without re-opening the file.
        self._handles: "OrderedDict[str, tuple[RecordingHandle, str]]" = OrderedDict()
        self._current_name = ""
        self._current_duration = 0.0
        self._current_modality = ""
        self._current_path = ""
        self._current_fs = 0.0

    # -- scalar "current recording" (the top-bar chip; QAbstractListModel has no
    #    single-selection notion, so we track the most-recently-opened row here) --
    def _get_current_name(self) -> str:
        return self._current_name

    def _get_current_duration(self) -> float:
        return self._current_duration

    def _get_current_modality(self) -> str:
        return self._current_modality

    def _get_current_path(self) -> str:
        return self._current_path

    def _get_current_fs(self) -> float:
        return self._current_fs

    currentName = Property(str, _get_current_name, notify=currentChanged)
    currentDurationSec = Property(float, _get_current_duration, notify=currentChanged)
    currentModality = Property(str, _get_current_modality, notify=currentChanged)
    currentPath = Property(str, _get_current_path, notify=currentChanged)
    #: the recording's NATIVE sample rate (not the model rate) — WFDB annotation export needs it to
    #: place annotations at the right sample indices on non-250 Hz records.
    currentFsHz = Property(float, _get_current_fs, notify=currentChanged)

    def _set_current(self, path: str, handle: RecordingHandle, modality: str) -> None:
        fs = float(next(iter(handle.fs_hz.values()), 0.0)) or 1.0
        n = max(handle.n_samples.values()) if handle.n_samples else 0
        self._current_name = Path(path).name
        self._current_duration = float(n) / fs
        self._current_modality = modality
        self._current_path = path
        self._current_fs = fs
        self.currentChanged.emit()

    def _get_count(self) -> int:
        return len(self._entries)

    # Convenience property for QML (`recordings.count`) -- QAbstractListModel
    # does not expose `rowCount()` as a bindable QML property by default.
    count = Property(int, _get_count, notify=countChanged)

    # -- QAbstractListModel plumbing ---------------------------------------
    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802 - Qt override name
        if parent.isValid():
            return 0
        return len(self._entries)

    def roleNames(self) -> dict[int, bytes]:  # noqa: N802
        return {
            self.NameRole: b"name",
            self.PathRole: b"path",
            self.ModalityRole: b"modality",
            self.DurationRole: b"durationSec",
            self.TranscodedRole: b"transcoded",
        }

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:  # noqa: N802
        if not index.isValid() or not (0 <= index.row() < len(self._entries)):
            return None
        entry = self._entries[index.row()]
        return {
            self.NameRole: entry.name,
            self.PathRole: entry.path,
            self.ModalityRole: entry.modality,
            self.DurationRole: entry.duration_sec,
            self.TranscodedRole: entry.transcoded,
        }.get(role)

    @staticmethod
    def _close_backend(handle) -> None:
        """Best-effort release of an evicted handle's file descriptor (MNE keeps one open; zarr may).
        Never raises — a close failure must not break opening a new recording."""
        be = getattr(handle, "backend", None)
        for closer in (getattr(be, "close", None),
                       getattr(getattr(be, "store", None), "close", None)):
            if callable(closer):
                try:
                    closer()
                except Exception:  # noqa: BLE001
                    pass

    def _remember_handle(self, path: str, handle: RecordingHandle, modality: str) -> None:
        """Cache a handle as most-recently-used, evicting (and closing) the LRU beyond
        ``_MAX_OPEN_HANDLES``. The current recording is always the newest entry, so it is never evicted."""
        self._handles[path] = (handle, modality)
        self._handles.move_to_end(path)
        while len(self._handles) > _MAX_OPEN_HANDLES:
            _old_path, (old_handle, _m) = self._handles.popitem(last=False)
            self._close_backend(old_handle)

    # -- QML-invokable API (design spec (d)) -------------------------------
    @Slot(str)
    @Slot(str, str)
    def open(self, path: str, modality: str = "") -> None:
        """Open a recording header (lazily), pick its modality, add a row, and emit
        ``recordingOpened`` so the Coordinator runs inference.

        ``modality`` FORCES the model (from the "Open ▾" menu: ecg|eeg|ppg|eda); empty
        string = auto-detect from the header. The open is header-only (``<2s``), so it runs
        inline; the heavy sliding-window inference is dispatched by the Coordinator.
        """
        path = str(Path(path))
        forced = (modality or "").strip().lower()
        # Already open: re-select it (drive inference again) instead of appending a
        # duplicate row every click. A new forced modality re-runs against that model.
        if path in self._handles:
            handle, cur = self._handles[path]
            mod = forced or cur
            self._remember_handle(path, handle, mod)   # re-tag (if changed) + mark most-recently-used
            if mod != cur:
                self._update_entry_modality(path, mod)
            self._set_current(path, handle, mod)
            self.recordingOpened.emit(path)
            return
        try:
            handle = open_recording(path)
            voted, conf = modality_vote(handle)   # the vote ALWAYS runs (background verification)
            mod = forced or voted
        except Exception as exc:  # noqa: BLE001 - surface to the UI, don't crash
            self.openFailed.emit(path, str(exc))
            return
        fs = float(next(iter(handle.fs_hz.values()), 0.0)) or 1.0
        n = max(handle.n_samples.values()) if handle.n_samples else 0
        duration = float(n) / fs
        self._remember_handle(path, handle, mod)
        self._append_entry(
            RecordingEntry(
                name=Path(path).name, path=path, modality=mod,
                duration_sec=duration, transcoded=False,
            )
        )
        self._set_current(path, handle, mod)
        self.recordingOpened.emit(path)
        # Background sanity check: warn if the user forced a modality the header confidently
        # disagrees with (e.g. opened as ECG but the channel units/names say EEG).
        if forced and voted and voted != forced and conf >= 0.6:
            self.modalityMismatch.emit(forced, voted, float(conf))
        # On the AUTO path a low-confidence vote (only the fs tie-break fired, conf capped ~0.35) is
        # effectively a guess — surface it so the grades aren't presented as authoritative for an
        # arbitrarily-chosen model. modality_vote's own contract says to ask the user here.
        elif not forced and voted and conf < _MODALITY_CONF_FLOOR:
            self.modalityUncertain.emit(voted, float(conf))

    @Slot(str, str)
    def setModality(self, path: str, modality: str) -> None:  # noqa: N802
        """Post-open correction: re-tag an already-open recording's modality and re-emit
        ``recordingOpened`` so the Coordinator reloads the matching model + re-runs."""
        path = str(Path(path))
        mod = (modality or "").strip().lower()
        if path not in self._handles or not mod:
            return
        handle, _cur = self._handles[path]
        self._remember_handle(path, handle, mod)
        self._update_entry_modality(path, mod)
        self._set_current(path, handle, mod)
        self.recordingOpened.emit(path)

    def _update_entry_modality(self, path: str, modality: str) -> None:
        for i, e in enumerate(self._entries):
            if e.path == path:
                e.modality = modality
                idx = self.index(i, 0)
                self.dataChanged.emit(idx, idx, [self.ModalityRole])
                break

    def handle_for(self, path: str) -> tuple[RecordingHandle, str] | None:
        """Return ``(handle, modality)`` for an opened recording, or ``None``."""
        key = str(Path(path))
        hit = self._handles.get(key)
        if hit is not None:
            self._handles.move_to_end(key)              # mark as recently used (protect from LRU eviction)
        return hit

    @Slot(str)
    def transcodeIfNeeded(self, path: str) -> None:  # noqa: N802 - QML-facing camelCase slot name
        """Offer to transcode a foreign-format file into the Zarr store + pyramid.

        TODO(Plan2 §6.1): dispatch a `workers.qt_threads.TranscodeTask`; this
        is a one-time, cancellable cost per recording (Plan 2 §14).
        """
        raise NotImplementedError("RecordingListModel.transcodeIfNeeded: TODO Plan2 §6.1")

    def _append_entry(self, entry: RecordingEntry) -> None:
        """Internal helper workers/handlers use to add a row (not QML-facing)."""
        self.beginInsertRows(QModelIndex(), len(self._entries), len(self._entries))
        self._entries.append(entry)
        self.endInsertRows()
        self.countChanged.emit()
