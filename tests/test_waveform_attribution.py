"""Hiding the GRADED channel must not leave the plot attributing its grades to another trace.

The model grades ONE channel. The quality bands and the hover amplitude both describe that channel —
``signalView.valueAt()`` reads its cache, never the lane under the cursor. So with a two-channel
record (["RESP", "II"] — the analyzed lead is not channel 0) the user could hide the analyzed lead
and be left with tier-coloured bands painted full height over a completely ungraded RESP trace, and a
tooltip quoting lead II's millivolts while a ±0.5 sine was on screen. Same misleading-attribution
class the export provenance already had to fix.

These assertions are on the REAL QML scene (the state lives in ``WaveformChart.qml``), booted
offscreen against the real controllers.
"""

from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

wfdb = pytest.importorskip("wfdb")
pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QEvent, QObject  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from biosqa.main import build_engine  # noqa: E402
from biosqa.viewmodels.channel_model import ChannelListModel  # noqa: E402

FS, DUR = 250, 120


def _write_two_channel(tmp: Path) -> str:
    """RESP first, the graded ECG lead second — and RESP is a placid ±0.5 sine while II carries a
    noise burst, so a value taken from the wrong channel is unmistakable."""
    n = FS * DUR
    t = np.arange(n) / FS
    rng = np.random.default_rng(1)
    ii = np.exp(-((t % (1 / 1.2)) - 0.1) ** 2 / 8e-4) + 0.02 * rng.standard_normal(n)
    ii[FS * 40:FS * 60] += 3.0 * rng.standard_normal(FS * 20)
    resp = 0.5 * np.sin(2 * np.pi * 0.25 * t)
    wfdb.wrsamp("attrrec", fs=FS, units=["mV", "mV"], sig_name=["RESP", "II"],
                p_signal=np.stack([resp, ii], axis=1), fmt=["16", "16"], write_dir=str(tmp))
    return str(tmp / "attrrec.hea")


def _pump(app, seconds: float) -> None:
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.02)


@pytest.fixture(scope="module")
def charted(tmp_path_factory):
    app = QApplication.instance() or QApplication([])
    app.setOrganizationName("BioSQA")
    app.setApplicationName("BioSQA Studio")
    rec = _write_two_channel(tmp_path_factory.mktemp("attr"))

    engine = build_engine()
    assert engine.rootObjects(), "QML engine failed to load Main.qml"
    window = engine.rootObjects()[0]
    ctrl = {type(o).__name__: o for o in engine._biosqa_controllers}

    _pump(app, 0.5)
    ctrl["RecordingListModel"].open(rec)
    deadline = time.time() + 120.0
    while time.time() < deadline and not ctrl["QualitySegmentModel"].totalCount:
        _pump(app, 0.5)
    assert ctrl["QualitySegmentModel"].totalCount, "inference never produced segments"

    yield app, window, ctrl

    for obj in engine.rootObjects():        # tear the scene down while the controllers are alive
        obj.deleteLater()
    app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()
    engine._biosqa_controllers = None
    engine.deleteLater()
    app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()
    gc.collect()


def _names(channels) -> list:
    return [channels.data(channels.index(i, 0), ChannelListModel.NameRole)
            for i in range(channels.rowCount())]


def test_bands_are_tier_coloured_while_the_graded_channel_is_drawn(charted):
    """The baseline: nothing about this fix may dull the normal case."""
    _app, window, ctrl = charted
    chart = window.findChild(QObject, "waveformChart")
    bands = window.findChild(QObject, "qualityBands")
    notice = window.findChild(QObject, "bandsOrphanedNotice")
    assert chart is not None and bands is not None and notice is not None

    assert chart.property("analyzedChannel") == "II"     # the graded lead, not channel 0
    assert chart.property("analyzedDrawn") is True
    assert bands.property("muted") is False, "the bands were dulled while their channel is on screen"
    assert notice.property("visible") is False
    assert chart.property("valueSuffix") == "", "the tooltip is annotated when it need not be"


def test_hiding_the_graded_channel_disowns_the_bands_and_names_the_tooltip(charted):
    """Hide it and the plot must stop pretending: the bands lose their tier colours and say whose
    grades they are, and the hover amplitude names the (hidden) channel it was read from."""
    app, window, ctrl = charted
    chart = window.findChild(QObject, "waveformChart")
    bands = window.findChild(QObject, "qualityBands")
    notice = window.findChild(QObject, "bandsOrphanedNotice")
    channels, signal_view = ctrl["ChannelListModel"], ctrl["SignalViewController"]

    names = _names(channels)
    channels.toggle(names.index("RESP"))                 # show the ungraded lane
    channels.toggle(names.index("II"))                   # ...and hide the graded one
    _pump(app, 0.6)
    assert signal_view.laneChannels == ["RESP"], "the analyzed lane is still being drawn"

    assert chart.property("analyzedDrawn") is False
    assert bands.property("muted") is True, "tier-coloured bands still painted over an ungraded trace"
    assert notice.property("visible") is True, "nothing on screen says whose grades those bands are"
    suffix = chart.property("valueSuffix")
    assert "II" in suffix and "hidden" in suffix, (
        f"the hover amplitude still reads as the visible trace's ({suffix!r})")

    # and it recovers: show the graded channel again and the bands go back to their tier colours
    channels.toggle(names.index("II"))
    _pump(app, 0.6)
    assert chart.property("analyzedDrawn") is True
    assert bands.property("muted") is False
    assert notice.property("visible") is False
    # two lanes now, so the tooltip still names the channel its number belongs to
    assert "II" in chart.property("valueSuffix")
