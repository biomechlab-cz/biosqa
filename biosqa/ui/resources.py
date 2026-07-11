"""QML resource resolution: disk (dev) vs. compiled Qt resource (frozen builds).

`biosqa.main` loads `Main.qml` straight off disk at dev time (see
`UI_DIR` there) -- fast iteration, no compile step. This module is the
"Python resource-registration approach" alternative mentioned in the task
brief, for use once a release build wants to ship the QML tree as a single
compiled resource instead of loose files (Plan 2 §13 packaging note).

Two supported strategies, either is valid -- pick one per build target:

1. **Qt resource file (`qml.qrc`)**: compile with
   ``pyside6-rcc qml.qrc -o qml_rc.py``, then ``import qml_rc`` (this
   registers the compiled resource with Qt as an import side effect) before
   loading ``QUrl("qrc:/qml/Main.qml")``.
2. **Plain search path registration** (no compilation step, still works in a
   frozen one-folder build as long as the `ui/` directory ships alongside
   the executable): just point `QQmlApplicationEngine.addImportPath` /
   `QUrl.fromLocalFile` at the frozen app's own directory, which is what
   `resolve_ui_dir()` below does, resolving relative to `sys.executable`
   when frozen (PyInstaller/Nuitka set `sys.frozen`) and relative to this
   file otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path


def resolve_ui_dir() -> Path:
    """Return the directory containing `Main.qml`, in dev or frozen mode.

    TODO(Plan2 §13): verify this against an actual PyInstaller/Nuitka build
    -- frozen path resolution for bundled QML data files is explicitly
    flagged in Plan 2 §13 as a common pitfall, alongside the `models/` path
    issue.
    """
    if getattr(sys, "frozen", False):
        # PyInstaller/Nuitka one-folder builds place bundled data next to
        # the executable; see build/biosqa.spec's `datas` and
        # build/NUITKA_BUILD_NOTES.md.
        return Path(sys.executable).resolve().parent / "biosqa" / "ui"
    return Path(__file__).resolve().parent


def try_load_compiled_resource() -> bool:
    """Attempt to import the `pyside6-rcc`-compiled `qml_rc` module, if present.

    Returns True if the compiled resource module was importable (so callers
    should load from `qrc:/qml/...` instead of a local file path).
    """
    try:
        import qml_rc  # noqa: F401  (import registers the Qt resource as a side effect)
    except ImportError:
        return False
    return True
