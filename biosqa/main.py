"""BioSQA Studio — application entry point.

Bootstraps a ``QApplication`` + ``QQmlApplicationEngine``, wires the
Python-backed controllers/models as QML context properties (design spec
section (b)/(d)), registers the custom scene-graph plot item, and loads
``biosqa/ui/Main.qml``.

Run with::

    python -m biosqa.main

TODO(Plan2 §12 Phase 0): accept a ``--open <path>`` CLI arg once
``io.loaders`` + ``RecordingListModel.open`` are implemented, so the app can
be launched directly onto a recording instead of the empty-state UI.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Force the customizable "Basic" Controls style app-wide. The native Windows style
# silently ignores background/contentItem customization, so the dark design's custom
# buttons/pills would render as light native chrome. Set as an env var (read when the
# Controls plugin first loads) so it applies to the app, tests, and the render driver
# alike, and is idempotent across repeated build_engine() calls (unlike QQuickStyle.setStyle).
os.environ.setdefault("QT_QUICK_CONTROLS_STYLE", "Basic")

from PySide6.QtCore import QUrl
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtWidgets import QApplication

from biosqa.viewmodels.app_controller import AppController
from biosqa.viewmodels.coordinator import Coordinator
from biosqa.viewmodels.channel_model import ChannelListModel
from biosqa.viewmodels.export_controller import ExportController
from biosqa.viewmodels.guard_controller import GuardController
from biosqa.viewmodels.inference_status import InferenceStatusController
from biosqa.viewmodels.model_card_model import ModelCardModel
from biosqa.viewmodels.quality_segment_model import QualitySegmentModel
from biosqa.viewmodels.recording_controller import RecordingListModel
from biosqa.viewmodels.selection_controller import SelectionController
from biosqa.viewmodels.settings_controller import SettingsController
from biosqa.viewmodels.signal_view_controller import SignalViewController

# Resolve data paths from source AND from a PyInstaller freeze. When frozen, the bundled
# data lives under sys._MEIPASS (the one-folder `_internal` dir) at the same relative layout
# the spec ships it: `biosqa/ui/*` and `models/*` (NOT next to this module's __file__, which
# PyInstaller places at the bundle root).
if getattr(sys, "frozen", False):
    _BASE = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    UI_DIR = _BASE / "biosqa" / "ui"
    # build/biosqa.spec ships the models at the bundle root; accept the package-relative
    # layout too so the spec can move to it without breaking this resolver.
    _MODEL_CANDIDATES = (_BASE / "biosqa" / "models", _BASE / "models")
else:
    _PKG = Path(__file__).resolve().parent
    UI_DIR = _PKG / "ui"
    # A wheel install ships the models as package data (biosqa/models, force-included by
    # pyproject.toml); a dev checkout keeps them at the repo root (app/models) and has no
    # package-relative copy. Package-relative wins so an installed app can never read a
    # stale sibling directory that happens to sit next to site-packages.
    _MODEL_CANDIDATES = (_PKG / "models", _PKG.parent / "models")


def _resolve_models_dir(candidates: tuple[Path, ...]) -> Path:
    """First candidate that actually holds model cards; else the last (for an honest error).

    Picking a directory that merely *exists* would silently select an empty models/ over a
    populated one, leaving the app modelless with no explanation.
    """
    for path in candidates:
        if path.is_dir() and any(path.glob("*.model_card.json")):
            return path
    return candidates[-1]


MODELS_DIR = _resolve_models_dir(_MODEL_CANDIDATES)
MAIN_QML = UI_DIR / "Main.qml"
FONTS_DIR = UI_DIR / "fonts"

_FONTS_LOADED = False


def _load_fonts() -> None:
    """Register the bundled Geist (UI) + JetBrains Mono (data) fonts so the app's
    typography matches the design regardless of what the host machine has installed.

    Without this, Qt silently falls back to a system face (and headless renders as
    tofu). Called once at engine construction so tests / the screenshot driver /
    ``main`` all pick it up. Family names: "Geist" and "JetBrains Mono" (see Theme.qml).
    """
    global _FONTS_LOADED
    if _FONTS_LOADED:
        return
    from PySide6.QtGui import QFont, QFontDatabase, QGuiApplication

    for fname in ("Geist.ttf", "JetBrainsMono.ttf"):
        path = FONTS_DIR / fname
        if path.exists():
            QFontDatabase.addApplicationFont(str(path))
    app = QGuiApplication.instance()
    if app is not None:
        base = QFont("Geist")
        base.setPixelSize(13)  # design base font-size
        app.setFont(base)
    _FONTS_LOADED = True


def build_engine() -> QQmlApplicationEngine:
    """Create the engine, register singleton context properties, load Main.qml.

    All of the "singleton" Python-backed objects (design spec table (b)) are
    exposed once via ``rootContext().setContextProperty`` rather than
    ``qmlRegisterType`` -- QML binds to them directly (``recordings.open(...)``,
    ``channels.toggle(0)``, ...) without ever instantiating them from QML.
    """
    _load_fonts()
    # Org/app identity — set before any QSettings use so SettingsController persists to a
    # stable location (build_engine may run in tests where main() didn't set these).
    from PySide6.QtCore import QCoreApplication
    if not QCoreApplication.organizationName():
        QCoreApplication.setOrganizationName("BioSQA")
    if not QCoreApplication.applicationName():
        QCoreApplication.setApplicationName("BioSQA Studio")
    # The native Windows Controls style forbids customizing `background`/`contentItem`,
    # so our dark AccentButton/OutlineButton/etc. would render as light native chrome.
    # "Basic" is the customizable style the dark design requires. Must be set before the
    # engine instantiates any Controls.
    from PySide6.QtQuickControls2 import QQuickStyle
    if not QQuickStyle.name():
        QQuickStyle.setStyle("Basic")
    engine = QQmlApplicationEngine()

    # --- construct singleton controllers/models -----------------------------
    recordings = RecordingListModel()
    channels = ChannelListModel()
    signal_view = SignalViewController()
    segments = QualitySegmentModel()
    selection = SelectionController()
    inference = InferenceStatusController()
    model_card = ModelCardModel()
    exporter = ExportController()
    guard = GuardController()
    settings = SettingsController()

    app_controller = AppController()
    app_controller.attach_controllers(
        recordings=recordings,
        channels=channels,
        signal_view=signal_view,
        segments=segments,
        selection=selection,
        inference=inference,
        model_card=model_card,
        exporter=exporter,
    )

    exporter.attach(segments, selection, model_card, recordings, guard)  # +guard=domain-shift index for provenance

    # --- the Coordinator: the connective tissue that runs inference on open and
    # routes results back into the (already-bound) viewmodels (Plan 2 §7/§9). ---
    coordinator = Coordinator(
        models_dir=MODELS_DIR,
        recordings=recordings,
        channels=channels,
        segments=segments,
        inference=inference,
        model_card=model_card,
        signal_view=signal_view,
        selection=selection,
        guard=guard,
        settings=settings,
    )

    # --- expose to QML (design spec (b): "recordings, channels, signalView,
    # selection, segments, inference, modelCard, exporter") ------------------
    ctx = engine.rootContext()
    ctx.setContextProperty("AppController", app_controller)
    ctx.setContextProperty("recordings", recordings)
    ctx.setContextProperty("channels", channels)
    ctx.setContextProperty("signalView", signal_view)
    ctx.setContextProperty("segments", segments)
    ctx.setContextProperty("selection", selection)
    ctx.setContextProperty("inference", inference)
    ctx.setContextProperty("modelCard", model_card)
    ctx.setContextProperty("exporter", exporter)
    ctx.setContextProperty("guard", guard)
    ctx.setContextProperty("settings", settings)

    # Keep strong Python references alive for the lifetime of the engine
    # (setContextProperty does not take ownership); stash them on the engine
    # instance itself so they aren't garbage-collected once this function
    # returns.
    engine._biosqa_controllers = (  # type: ignore[attr-defined]
        app_controller,
        recordings,
        channels,
        signal_view,
        segments,
        selection,
        inference,
        model_card,
        exporter,
        guard,
        settings,
        coordinator,
    )

    engine.addImportPath(str(UI_DIR))
    engine.load(QUrl.fromLocalFile(str(MAIN_QML)))
    return engine


def _prewarm_loaders() -> None:
    """Import the heavy loader deps (wfdb/mne, which pull in pandas/pyarrow) ONCE on a background
    thread at startup. The loaders import these lazily inside their open functions; if the first
    open happens from the deep native-file-dialog callback stack, that import chain overflows the
    stack in the frozen build. Pre-warming caches the modules at a shallow stack — and Python's
    per-module import lock makes any later in-function ``import`` a safe cache hit, so the deep
    import never runs on the GUI callback stack.
    """
    try:
        import mne  # noqa: F401
        import wfdb  # noqa: F401
    except Exception:  # noqa: BLE001 - best-effort; a real open() surfaces any import error properly
        pass


def install_teardown_guard(app, engine) -> None:
    """Destroy the QML scene at quit WHILE the Python context objects are still alive.

    Otherwise interpreter shutdown can garbage-collect the context controllers (``settings`` /
    ``AppController`` / ``recordings`` / ``inference`` / ...) before the QML engine, and every binding that
    reads a context property re-evaluates against a now-destroyed object — spamming dozens of
    ``TypeError: Cannot read property '<x>' of null`` warnings on close. Deleting the engine's root objects
    on ``aboutToQuit`` (which fires while the event loop can still process the deferred deletes) guarantees
    the QML tree is gone before the controllers are collected.
    """
    def _teardown_qml() -> None:
        for obj in engine.rootObjects():
            obj.deleteLater()

    app.aboutToQuit.connect(_teardown_qml)


def main(argv: list[str] | None = None) -> int:
    """Application entry point; returns the process exit code."""
    import threading

    threading.Thread(target=_prewarm_loaders, name="loader-prewarm", daemon=True).start()
    app = QApplication(argv if argv is not None else sys.argv)
    app.setOrganizationName("BioSQA")
    app.setApplicationName("BioSQA Studio")

    from pathlib import Path

    from PySide6.QtGui import QIcon
    _icon = UI_DIR / "icon.png"
    if _icon.exists():
        app.setWindowIcon(QIcon(str(_icon)))

    engine = build_engine()
    if not engine.rootObjects():
        # QML failed to load (syntax error, missing type, ...) -- fail loudly
        # rather than showing a blank window.
        return 1

    install_teardown_guard(app, engine)

    # Debug/CI helper: BIOSQA_AUTOLOAD=<path> opens a recording ~1s after startup so the
    # load path can be smoke-tested headlessly. No effect unless the env var is set.
    _autoload = os.environ.get("BIOSQA_AUTOLOAD")
    if _autoload:
        from PySide6.QtCore import QTimer

        def _do_autoload() -> None:
            for o in getattr(engine, "_biosqa_controllers", ()):
                if type(o).__name__ == "RecordingListModel":
                    try:
                        o.open(_autoload)
                    except Exception:  # noqa: BLE001
                        import traceback
                        traceback.print_exc()
                    break

        QTimer.singleShot(1000, _do_autoload)

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
