"""Pytest bootstrap: make `biosqa` importable without an editable install.

These tests exercise only the framework-agnostic pieces (`io`, `inference`,
`model`) -- see `app/README.md` for why that subset happens to be runnable
even without the app's own venv (PySide6/onnxruntime/etc.) fully set up.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

APP_ROOT = Path(__file__).resolve().parent.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


@pytest.fixture(autouse=True, scope="session")
def _isolate_qsettings(tmp_path_factory):
    """Redirect QSettings to a throwaway dir so tests never read or clobber the real user config
    (SettingsController persists to QSettings). No-op if PySide6 isn't importable."""
    try:
        from PySide6.QtCore import QSettings
    except Exception:
        yield
        return
    d = str(tmp_path_factory.mktemp("qsettings"))
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, d)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.SystemScope, d)
    yield
