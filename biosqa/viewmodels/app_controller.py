"""Global navigation state: which of the four full-bleed views is showing (design spec (a)/(b)).

``Main.qml``'s ``Loader { sourceComponent: viewFor(AppController.currentView) }``
binds to this controller's ``currentView`` property; the activity rail calls
``AppController.go("workspace")`` etc. on click.
"""

from __future__ import annotations

from PySide6.QtCore import Property, QObject, Signal, Slot

#: The four full-bleed views (design spec §0): the icon rail swaps between
#: these, it does not dock them simultaneously.
VIEWS = ("workspace", "overview", "inspector", "segmentation")


class AppController(QObject):
    """Owns/wires the other singleton controllers and tracks the current view."""

    currentViewChanged = Signal()
    settingsOpenChanged = Signal()
    notify = Signal(str)  # global toast message (Main.qml shows it)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._current_view = "workspace"
        self._settings_open = False
        # Plain attribute references to the sibling controllers (design spec
        # (b): "AppController ... owns other controllers"). These are NOT
        # re-exposed as Q_PROPERTYs here -- QML accesses them directly via
        # their own top-level context properties (`recordings`, `channels`,
        # ...) set up in main.build_engine(). This dict exists so future
        # cross-controller wiring (e.g. "opening a recording resets
        # selection") has one place to live.
        self._controllers: dict[str, QObject] = {}

    def attach_controllers(self, **controllers: QObject) -> None:
        """Register sibling controllers by name (called once from ``main.build_engine``)."""
        self._controllers.update(controllers)

    def _get_current_view(self) -> str:
        return self._current_view

    def _set_current_view(self, view: str) -> None:
        if view not in VIEWS:
            raise ValueError(f"unknown view {view!r}, expected one of {VIEWS!r}")
        if view != self._current_view:
            self._current_view = view
            self.currentViewChanged.emit()

    currentView = Property(str, _get_current_view, _set_current_view, notify=currentViewChanged)

    @Slot(str)
    def go(self, view: str) -> None:
        """QML-invokable navigation entry point (``ActivityRail.qml`` onClicked)."""
        self._set_current_view(view)

    @Slot(str)
    def toast(self, message: str) -> None:
        """Show a transient toast message (QML calls ``AppController.toast('…')``)."""
        self.notify.emit(message)

    # -- settings panel state ------------------------------------------------
    def _get_settings_open(self) -> bool:
        return self._settings_open

    def _set_settings_open(self, value: bool) -> None:
        if value != self._settings_open:
            self._settings_open = value
            self.settingsOpenChanged.emit()

    settingsOpen = Property(bool, _get_settings_open, _set_settings_open,
                            notify=settingsOpenChanged)

    @Slot()
    def openSettings(self) -> None:
        self._set_settings_open(True)

    @Slot()
    def closeSettings(self) -> None:
        self._set_settings_open(False)
