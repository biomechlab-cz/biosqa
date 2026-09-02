"""`ChannelListPanel.qml` backing model: per-channel visibility, color, unit, sparkline.

Design spec (b): "per-channel visibility, color, unit, sparkline data".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PySide6.QtCore import Property, QAbstractListModel, QModelIndex, Qt, Signal, Slot

#: Modality identity colors (design spec (c) "Modality colors"), keyed by the
#: channel's modality tag -- distinct from the Q0-Q3 quality-tier palette.
MODALITY_COLORS = {
    "ecg": "#FF7A85",
    "ppg": "#35D0BA",
    "eeg": "#6E8BFF",
    "eda": "#C08CF2",
}


@dataclass
class ChannelEntry:
    """One row: a single signal channel within the open recording."""

    name: str
    modality: str = ""
    visible: bool = True
    unit: str = ""
    mod_color: str = field(default="")
    sparkline: list[float] = field(default_factory=list)
    #: this is the channel inference actually GRADED — the quality bands/segments/exports describe
    #: it and no other channel of the recording. At most one row carries it (none if nothing ran).
    analyzed: bool = False

    def __post_init__(self) -> None:
        if not self.mod_color:
            self.mod_color = MODALITY_COLORS.get(self.modality.lower(), "#9AA4B6")


class ChannelListModel(QAbstractListModel):
    """`QAbstractListModel` over the open recording's channels.

    Roles: ``name``, ``visible``, ``modColor``, ``unit``, ``sparkline``, ``analyzed``.
    """

    NameRole = Qt.UserRole + 1
    VisibleRole = Qt.UserRole + 2
    ModColorRole = Qt.UserRole + 3
    UnitRole = Qt.UserRole + 4
    SparklineRole = Qt.UserRole + 5
    AnalyzedRole = Qt.UserRole + 6

    channelVisibilityChanged = Signal(int, bool)  # (index, visible)
    channelOrderChanged = Signal()
    countChanged = Signal()
    visibleCountChanged = Signal()   # number of VISIBLE channels changed (toggle or set)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._entries: list[ChannelEntry] = []
        self._role_getters = {                          # built once, not per data() call
            self.NameRole: lambda e: e.name,
            self.VisibleRole: lambda e: e.visible,
            self.ModColorRole: lambda e: e.mod_color,
            self.UnitRole: lambda e: e.unit,
            self.SparklineRole: lambda e: e.sparkline,
            self.AnalyzedRole: lambda e: e.analyzed,
        }

    def _get_count(self) -> int:
        return len(self._entries)

    # Convenience property for QML (`channels.count`), see RecordingListModel.
    count = Property(int, _get_count, notify=countChanged)

    def _get_visible_count(self) -> int:
        return sum(1 for e in self._entries if e.visible)

    #: number of currently-VISIBLE channels (the header shows `visibleCount / count`, not count/count).
    visibleCount = Property(int, _get_visible_count, notify=visibleCountChanged)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if parent.isValid():
            return 0
        return len(self._entries)

    def roleNames(self) -> dict[int, bytes]:  # noqa: N802
        # NOTE: the visibility role is exposed as "channelVisible", not
        # "visible" -- QML `Item`/delegate types already have a built-in
        # `visible` property, and a same-named model role would silently
        # shadow it inside delegates (a common QML footgun).
        return {
            self.NameRole: b"name",
            self.VisibleRole: b"channelVisible",
            self.ModColorRole: b"modColor",
            self.UnitRole: b"unit",
            self.SparklineRole: b"sparkline",
            self.AnalyzedRole: b"analyzed",
        }

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:  # noqa: N802
        if not index.isValid() or not (0 <= index.row() < len(self._entries)):
            return None
        getter = self._role_getters.get(role)
        return getter(self._entries[index.row()]) if getter is not None else None

    # -- QML-invokable API (design spec (d): "channels.toggle(index)",
    # "channels.setChannelOrder(...)") --------------------------------------
    @Slot(result=list)
    def visibleNames(self) -> list:  # noqa: N802
        """Ordered names of the currently-visible channels — the lanes the plot should draw
        (QML calls ``signalView.setLaneChannels(channels.visibleNames())`` on toggle)."""
        return [e.name for e in self._entries if e.visible]

    @Slot(int)
    def toggle(self, index: int) -> None:
        """Flip a channel's visibility and notify both QML and the plot canvas."""
        if not 0 <= index < len(self._entries):
            return
        entry = self._entries[index]
        entry.visible = not entry.visible
        model_index = self.index(index, 0)
        self.dataChanged.emit(model_index, model_index, [self.VisibleRole])
        self.channelVisibilityChanged.emit(index, entry.visible)
        self.visibleCountChanged.emit()

    @Slot(list)
    def setChannelOrder(self, order: list[int]) -> None:  # noqa: N802
        """Reorder channels (e.g. drag-to-reorder in `ChannelListPanel.qml`).

        TODO(Plan2 §8.2 "stacked vs overlaid channels"): reordering affects
        lane stacking order in `SignalView.qml`'s `Repeater`; implement once
        that layout is real.
        """
        if sorted(order) != list(range(len(self._entries))):
            raise ValueError("order must be a permutation of the current channel indices")
        self.beginResetModel()
        self._entries = [self._entries[i] for i in order]
        self.endResetModel()
        self.channelOrderChanged.emit()

    def _append_entry(self, entry: ChannelEntry) -> None:
        """Internal helper (e.g. called once a recording is opened) -- not QML-facing."""
        self.beginInsertRows(QModelIndex(), len(self._entries), len(self._entries))
        self._entries.append(entry)
        self.endInsertRows()
        self.countChanged.emit()
        self.visibleCountChanged.emit()

    def set_channels(self, entries: list[ChannelEntry]) -> None:
        """Replace the whole channel list (e.g. when a new recording is opened).

        Not QML-facing -- called from the Coordinator on the GUI thread.
        """
        self.beginResetModel()
        self._entries = list(entries)
        self.endResetModel()
        self.countChanged.emit()
        self.visibleCountChanged.emit()
