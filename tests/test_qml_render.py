"""The QML actually RENDERS what it claims to (offscreen, real pipeline, real pixels).

This file exists because the suite had ZERO coverage of QML rendering, and a real bug shipped
through that hole: the quality bands stopped painting over the waveform entirely.

The cause is worth remembering. ``Theme.tierInfo(tier).color`` is a JS *string* ("#2FBF71").
Everywhere that needs the components binds it to a ``property color`` and QML converts it for you --
but ``WaveformChart``'s band layer is raw Canvas JS, where ``"#2FBF71".r`` is ``undefined``, so
``Qt.rgba(undefined, undefined, undefined, 0.15)`` produced an invalid brush and filled NOTHING.
Every one of the 379 other tests passed. The app silently stopped showing the one thing it exists
to show, and only a screenshot caught it.

So: assert on PIXELS, from the REAL pipeline. Anything less cannot see this class of bug.
"""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import QEvent, QEventLoop, QObject, QPointF, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from biosqa.main import build_engine  # noqa: E402

RECORD = Path(__file__).resolve().parent.parent / "dummy_data" / "test_ecg_3min.hea"


def _pump(ms: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def _wait_for(pred, timeout_ms: int = 90_000) -> bool:
    waited = 0
    while waited < timeout_ms:
        if pred():
            return True
        _pump(100)
        waited += 100
    return False


@pytest.fixture(scope="module")
def rendered():
    """Boot the real app offscreen, grade the real dummy ECG, view the whole record, grab the frame."""
    app = QApplication.instance() or QApplication([])
    app.setOrganizationName("BioSQA")
    app.setApplicationName("BioSQA Studio")

    engine = build_engine()
    assert engine.rootObjects(), "QML engine failed to load Main.qml"
    window = engine.rootObjects()[0]

    ctrl = {type(o).__name__: o for o in getattr(engine, "_biosqa_controllers", ())}
    recordings, segments = ctrl["RecordingListModel"], ctrl["QualitySegmentModel"]
    signal_view = ctrl["SignalViewController"]

    _pump(600)
    recordings.open(str(RECORD))
    assert _wait_for(lambda: segments.totalCount > 0), "inference never produced segments"

    duration = float(recordings.currentDurationSec)
    signal_view.setView(0.0, duration)          # whole record, so several tiers are on screen at once
    _pump(1200)

    canvas = window.findChild(QObject, "qualityBands")
    assert canvas is not None, "WaveformChart's quality-band Canvas is gone (objectName 'qualityBands')"

    # Locate the band layer in window coordinates rather than hardcoding pixels, so a layout change
    # relocates the probe instead of silently making this test meaningless.
    origin = canvas.mapToScene(QPointF(0.0, 0.0))
    rect = (origin.x(), origin.y(), canvas.property("width"), canvas.property("height"))
    assert rect[2] > 0 and rect[3] > 0, "the band Canvas has no size"

    img = window.grabWindow()
    assert not img.isNull(), "grabWindow() returned a null image"

    intervals = list(segments._all_intervals)
    yield img, rect, intervals, duration

    # Tear the QML scene down WHILE the context controllers are still alive -- exactly what
    # install_teardown_guard() does at quit. A lazy engine.deleteLater() leaves this engine's bindings
    # to re-evaluate against garbage-collected controllers during a LATER test's gc.collect(), which
    # spams "Cannot read property of null" and gets blamed on test_teardown_guard (it installs a
    # message handler and asserts there are none). Clean up after ourselves.
    for obj in engine.rootObjects():
        obj.deleteLater()
    app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()
    engine._biosqa_controllers = None
    engine.deleteLater()
    app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()
    gc.collect()


def _sample(img, rect, duration, t_sec, frac_down=0.12):
    """The rendered RGB at time ``t_sec``, taken high in the band layer to stay clear of the trace."""
    x0, y0, w, h = rect
    x = int(x0 + w * (t_sec / duration))
    y = int(y0 + h * frac_down)
    c = img.pixelColor(x, y)
    return c.red(), c.green(), c.blue()


def test_quality_bands_are_actually_painted_over_the_waveform(rendered):
    """The bands must be VISIBLE, not merely computed.

    Before the fix every sampled pixel was bit-identical to the empty background: the Canvas ran, the
    band geometry was right, and it filled with an invalid brush. Sampling real pixels is the only
    thing that can tell those apart.
    """
    img, rect, intervals, duration = rendered

    by_tier: dict[str, list[tuple[int, int, int]]] = {}
    for iv in intervals:
        mid = 0.5 * (iv.start_sec + iv.end_sec)
        by_tier.setdefault(iv.tier, []).append(_sample(img, rect, duration, mid))

    assert len(by_tier) >= 2, f"fixture is too bland to test banding (tiers: {sorted(by_tier)})"

    # A tinted band cannot be the same colour as an untinted one. If the brush is invalid, EVERY
    # sample collapses onto the background and this fails.
    flat = [c for cs in by_tier.values() for c in cs]
    assert len(set(flat)) > 1, (
        f"every band sampled to the same colour {flat[0]} -- the bands are not being painted "
        f"(this is exactly the Qt.rgba(undefined,...) regression)"
    )


def test_band_tint_matches_the_tier_it_represents(rendered):
    """Q0 (red) and Q3 (green) must not render as the same colour -- colour IS the encoding here."""
    img, rect, intervals, duration = rendered

    def mean_for(tier):
        px = [_sample(img, rect, duration, 0.5 * (iv.start_sec + iv.end_sec))
              for iv in intervals if iv.tier == tier]
        if not px:
            pytest.skip(f"no {tier} segment in the dummy recording")
        return tuple(sum(c[i] for c in px) / len(px) for i in range(3))

    q0, q3 = mean_for("Q0"), mean_for("Q3")

    # Not "is it exactly #E5484D" -- the band is a 15% wash over a dark plot, and pinning the exact
    # composite would break on any palette or opacity tweak. The INVARIANT is the ordering: the bad
    # tier reads warm, the good tier reads cool.
    assert q0[0] > q0[1], f"Q0 band is not red-dominant: RGB{q0}"
    assert q3[1] > q3[0], f"Q3 band is not green-dominant: RGB{q3}"
