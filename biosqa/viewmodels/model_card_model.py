"""`ModelCardPanel.qml` backing model: parsed model_card.json -> QML k/v rows (design spec (b))."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Property, QAbstractListModel, QModelIndex, Qt, Signal, Slot

from biosqa.model.model_card import ModelCard, load_model_card


class ModelCardModel(QAbstractListModel):
    """Flattens a validated `ModelCard` into (key, value) rows for a simple `ListView`."""

    KeyRole = Qt.UserRole + 1
    ValueRole = Qt.UserRole + 2

    cardChanged = Signal()   # a new card was loaded (windowSec / modality changed)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._rows: list[tuple[str, str]] = []
        self._card: ModelCard | None = None

    def _window_sec(self) -> float:
        c = self._card
        return float(c.l_m) / float(c.fs_hz) if c is not None and c.fs_hz else 0.0

    #: the model's fixed analysis-window length in seconds (L_m / fs_hz) — the real per-modality
    #: window (ECG/PPG 10 s, EEG 5 s, EDA 60 s), so the UI stops showing a hardcoded value.
    windowSec = Property(float, _window_sec, notify=cardChanged)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if parent.isValid():
            return 0
        return len(self._rows)

    def roleNames(self) -> dict[int, bytes]:  # noqa: N802
        return {self.KeyRole: b"key", self.ValueRole: b"value"}

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:  # noqa: N802
        if not index.isValid() or not (0 <= index.row() < len(self._rows)):
            return None
        key, value = self._rows[index.row()]
        return {self.KeyRole: key, self.ValueRole: value}.get(role)

    @Slot(str)
    def load(self, path: str) -> None:
        """Parse `path` as a `model_card.json` and populate the k/v rows.

        Raises via `ModelCardError` (surfaced to QML as a thrown exception --
        TODO(Plan2 §7.1): catch this in a worker and route to a QML error
        dialog instead of letting it propagate raw).
        """
        card = load_model_card(path)
        self.beginResetModel()
        self._card = card
        self.cardChanged.emit()
        self._rows = [
            ("modality", card.modality),
            ("L_m", str(card.l_m)),
            ("fs_hz", str(card.fs_hz)),
            ("class_order", ", ".join(card.class_order)),
            ("normalization", card.normalization.method),
            ("training_data_hash", card.training_data_hash),
            ("model_version", card.model_version),
        ]
        self.endResetModel()
