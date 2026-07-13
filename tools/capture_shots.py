"""Capture the docs screenshots by driving the REAL app (tools/capture_shots.py).

Not a mock and not a mockup: this boots the actual QML engine, opens a real recording from
``dummy_data/``, waits for the ACTUAL inference to land, and grabs each view. What ships on the
site is therefore what the app really renders -- which is the whole point of the section these
shots live in ("A working instrument beats any illustration").

    python tools/capture_shots.py                # all four, into docs/public/shots/

Must run on the real platform (QT_QPA_PLATFORM=windows), never offscreen: the app has a history
of GPU-only rendering faults that offscreen cannot see.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "windows")
os.environ.setdefault("PYTHONUTF8", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from biosqa.main import build_engine  # noqa: E402

OUT = ROOT / "docs" / "public" / "shots"
#: the WFDB header, not the bare basename -- open_recording() sniffs on the EXTENSION and rejects
#: an extensionless path as format 'unknown'.
RECORD = ROOT / "dummy_data" / "test_ecg_3min.hea"

#: (view, filename, width, height) -- the aspect ratios the docs layout expects.
SHOTS = [
    ("workspace",    "workspace.png",    1920, 1080),   # 16:9 hero
    ("overview",     "overview.png",     1280, 960),    # 4:3 cards
    ("segmentation", "segmentation.png", 1280, 960),
    ("inspector",    "inspector.png",    1280, 960),
]


def pump(ms: int) -> None:
    """Run the event loop for ms -- lets Qt render and lets worker results actually land."""
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def wait_for(predicate, timeout_ms: int = 60_000, label: str = "") -> bool:
    """Pump the loop until predicate() is true. Inference runs on a QThreadPool, so its results
    arrive as QUEUED signals -- they can only be delivered while the event loop spins."""
    waited = 0
    while waited < timeout_ms:
        if predicate():
            return True
        pump(100)
        waited += 100
    print(f"  !! timed out after {timeout_ms/1000:.0f}s waiting for {label}")
    return False


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    app = QApplication(sys.argv)
    app.setOrganizationName("BioSQA")
    app.setApplicationName("BioSQA Studio")

    engine = build_engine()
    if not engine.rootObjects():
        print("QML failed to load")
        return 1
    window = engine.rootObjects()[0]

    ctrl = {type(o).__name__: o for o in getattr(engine, "_biosqa_controllers", ())}
    app_ctrl = ctrl["AppController"]
    recordings = ctrl["RecordingListModel"]
    segments = ctrl["QualitySegmentModel"]
    selection = ctrl["SelectionController"]
    settings = ctrl["SettingsController"]

    # SettingsController persists to real QSettings, so whatever theme this machine last used would
    # leak into the shots. Pin the app's DEFAULT (dark) -- the site is dark-by-default too, and a new
    # user sees dark. Restore afterwards so running this doesn't silently re-theme the dev's app.
    was_dark = settings.themeDark
    settings.setThemeDark(True)

    pump(1200)                                     # let the scene settle before touching anything

    # open_recording() failures arrive on a signal, not as an exception -- without this a bad path
    # just produces an empty screenshot and a 60 s timeout with no reason given.
    recordings.openFailed.connect(lambda p, e: print(f"  !! open failed: {p}: {e}"))

    print(f"opening {RECORD.name} ...")
    recordings.open(str(RECORD))
    if not wait_for(lambda: segments.totalCount > 0, label="inference to produce segments"):
        return 1
    print(f"  inference landed: {segments.totalCount} segments")

    # The inspector renders a SELECTED segment; with none selected it is (correctly) an empty state,
    # which would make a misleading screenshot. Pick the WORST segment -- a clean Q3 shows none of the
    # artifact tags, SQI bars or recoverability the panel exists for.
    tiers = [iv.tier for iv in segments._all_intervals]        # dev tool: the model has no tier getter
    poor = next((i for i, t in enumerate(tiers) if t == "Q0"),
                next((i for i, t in enumerate(tiers) if t == "Q1"), 0))
    print(f"  inspecting segment #{poor + 1} ({tiers[poor]})")
    selection.selectByAllIndex(poor)
    pump(400)

    # Zoom the trace out to the WHOLE record. At the default 30 s viewport the visible span is a
    # single Q3 segment, so the workspace shot shows no quality bands at all -- which is precisely
    # the thing the section exists to show. Full-record view puts the real Q0-Q3 run-length bands
    # over the waveform.
    signal_view = ctrl["SignalViewController"]
    duration = float(recordings.currentDurationSec or 0.0)
    if duration > 0:
        signal_view.setView(0.0, duration)
        pump(600)

    for view, name, w, h in SHOTS:
        app_ctrl.go(view)
        window.setWidth(w)
        window.setHeight(h)
        pump(1400)                                 # relayout + repaint at the new size
        img = window.grabWindow()
        path = OUT / name
        if not img.save(str(path)):
            print(f"  !! failed to save {path}")
            return 1
        print(f"  {view:<13} -> {path.relative_to(ROOT)}  ({img.width()}x{img.height()})")

    settings.setThemeDark(was_dark)                # leave the dev's own preference as we found it
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
