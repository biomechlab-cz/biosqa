"""Regression: closing the app must not spam 'TypeError: Cannot read property <x> of null'.

Root cause (verified): on quit, Python can garbage-collect the QML context controllers (settings /
AppController / recordings / ...) before the QML engine, so every binding that reads a context property
re-evaluates against a destroyed object. ``install_teardown_guard`` deletes the QML root objects on
aboutToQuit — while the controllers are still alive — so the scene is gone before they are collected.

This test drives the REAL failure order (drop the controller refs + gc while the engine is alive) and
asserts the shipped guard reduces the null-binding warnings to zero. Without the guard this same scenario
emits ~90 warnings, so the assertion is not vacuous.
"""
from __future__ import annotations

import gc
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QEvent, qInstallMessageHandler

from biosqa.main import build_engine, install_teardown_guard


def test_teardown_guard_silences_context_null_bindings():
    app = QApplication.instance() or QApplication([])
    warnings: list[str] = []
    prev = qInstallMessageHandler(
        lambda mode, ctx, msg: warnings.append(msg) if "of null" in msg else None)
    try:
        engine = build_engine()
        assert engine.rootObjects(), "engine failed to load QML"
        install_teardown_guard(app, engine)

        # aboutToQuit fires while the loop can still process the deferred deletes -> QML torn down first.
        # (A real quit drains DeferredDelete as exec() returns; replicate that flush explicitly here.)
        app.aboutToQuit.emit()
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()

        # now simulate main() returning + interpreter GC collecting the controllers.
        engine._biosqa_controllers = None  # type: ignore[attr-defined]
        gc.collect()
        app.processEvents()

        assert warnings == [], f"teardown produced {len(warnings)} null-binding warnings: {warnings[:3]}"
    finally:
        qInstallMessageHandler(prev)
