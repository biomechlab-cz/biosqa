"""Regression: the Overview must not print a defaulted number as a measured one.

Two residual fabrications survived the first UI-truthfulness pass, both reachable with NO
recording open (the activity rail has no enabled gate on its nav buttons):

  * ``overview/ModalityRibbon.qml`` -- the per-tier breakdown printed "0%" for Q3/Q2/Q1/Q0
    whenever ``segments.tierFractions`` was empty, i.e. it asserted a MEASURED zero share for
    every tier on a recording nothing had graded yet.
  * ``overview/OverviewView.qml``   -- a "Windows" KPI computed ``duration / windowSec``, which
    ignores the window OVERLAP. The app ships 50% overlap by default, so the model really runs
    ~2x that many windows: a plausible-looking number that is simply wrong. No context property
    exposes the true count (the coordinator keeps it in a private ``_pending`` map), so the stat
    was removed rather than guessed at.
    ``Duration``/``Segments`` likewise defaulted to "00:00:00" / "0" with nothing loaded.

These drive the REAL QML scene (``build_engine``) and feed it through the REAL segmenter +
the REAL model setter the coordinator calls (``QualitySegmentModel.load_intervals``), so a
re-introduced default fails here rather than in front of a clinician.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QObject
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from biosqa.inference.segmenter import run_length_encode
from biosqa.main import build_engine
from biosqa.viewmodels.quality_segment_model import QualitySegmentModel

DASH = "—"  # em-dash: the "not measured" readout the overview settled on


@pytest.fixture(scope="module")
def scene():
    """The real QML scene with no recording open -- the state a user lands in on launch."""
    app = QApplication.instance() or QApplication([])
    engine = build_engine()
    assert engine.rootObjects(), "engine failed to load QML"
    QTest.qWait(300)          # let the scene lay out: items are 0-height until the first polish
    yield app, engine, engine.rootObjects()[0]
    for obj in engine.rootObjects():
        obj.deleteLater()
    app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()


def _ctl(engine, cls):
    for c in engine._biosqa_controllers:
        if isinstance(c, cls):
            return c
    raise AssertionError(f"no {cls.__name__} on the engine")


def _find(root, name):
    obj = root.findChild(QObject, name)
    assert obj is not None, f"no object named {name!r} in the scene"
    return obj


def _visual_children(item):
    """Recursive VISUAL children. Repeater-created delegates (the per-tier breakdown rows) are
    not QObject-children of the layout, so `findChild` cannot see them -- but they are the items
    actually painted, which is exactly what this test needs to read."""
    for child in item.childItems():
        yield child
        yield from _visual_children(child)


def _find_visual(root, name):
    for item in _visual_children(root):
        if item.objectName() == name:
            return item
    raise AssertionError(f"no visual item named {name!r} in the scene")


def _tier_text(overview, tier):
    """The string the ribbon's per-tier percentage Text actually renders."""
    return _find_visual(overview, "ribbonPct" + tier).property("text")


def _overview(app, engine, root):
    """Navigate to the overview exactly as the activity rail does, and return its root item."""
    from biosqa.viewmodels.app_controller import AppController

    _ctl(engine, AppController).go("overview")
    QTest.qWait(50)           # the Loader swaps the view; let the new one lay out
    return _find(root, "overviewView")


@pytest.fixture
def segments(scene):
    """The real `QualitySegmentModel` bound into QML; restored to 'no inference' afterwards so
    the module-scoped scene stays clean for the other tests."""
    _, engine, _ = scene
    model = _ctl(engine, QualitySegmentModel)
    yield model
    model.load_intervals([])


# -- no inference => no tier numbers -------------------------------------------------


def test_ribbon_tier_breakdown_is_not_measured_before_inference(scene):
    """Was "0%" for all four tiers: a measured-zero claim about an ungraded recording."""
    app, engine, root = scene
    ov = _overview(app, engine, root)
    ribbon = _find(ov, "modalityRibbon")

    assert ribbon.property("hasData") is False
    for tier in ("Q3", "Q2", "Q1", "Q0"):
        assert _tier_text(ov, tier) == DASH, f"{tier} reports a measured share with no inference"
    assert ribbon.property("usableText") == DASH
    assert _find(ov, "ribbonUsable").property("text") == DASH + " usable"


def test_overview_kpis_are_not_measured_before_a_recording_is_open(scene):
    """`Duration` read "00:00:00" and `Segments` read "0" with nothing loaded -- both are
    claims about a signal that was never read."""
    app, engine, root = scene
    ov = _overview(app, engine, root)

    assert ov.property("hasRecording") is False
    assert ov.property("hasStats") is False
    assert _find(ov, "kpiDuration").property("value") == DASH
    assert _find(ov, "kpiSegments").property("value") == DASH
    assert _find(ov, "kpiUsable").property("value") == DASH
    assert _find(ov, "kpiFlaggedQ0").property("value") == DASH


# -- the overlap-blind window count is gone ------------------------------------------


def test_overview_shows_no_window_count_kpi(scene):
    """The KPI labelled "Windows" was `duration / windowSec` -- it ignored the 50% overlap the
    app ships, so it under-reported the windows the model actually ran by ~2x. Nothing in the
    QML context exposes the true count, so the stat must not exist at all."""
    app, engine, root = scene
    ov = _overview(app, engine, root)

    labels = [c.property("label") for c in ov.findChildren(QObject)]
    assert "Windows" not in labels, "the overlap-blind window-count KPI is back"
    assert ov.findChild(QObject, "kpiWindows") is None


# -- real inference => real numbers (the fix must not just print em-dashes forever) ----


def test_ribbon_reports_the_real_tier_shares_after_inference(scene, segments):
    """Feed the REAL pipeline: per-window grades -> `run_length_encode` (the segmenter the
    inference worker uses) -> `load_intervals` (the exact call `Coordinator._on_intervals`
    makes). The ribbon must then print measured shares -- including a true "0.0%" for a tier
    the model never predicted, which is a measurement, not a default."""
    app, engine, root = scene
    ov = _overview(app, engine, root)

    # 10 non-overlapping 10 s windows: 5x Q3, 3x Q2, 2x Q1, and NO Q0 anywhere.
    tiers = np.array(["Q3"] * 5 + ["Q2"] * 3 + ["Q1"] * 2)
    conf = np.full(tiers.shape[0], 0.9)
    intervals = run_length_encode(tiers, conf, window_stride_sec=10.0, window_length_sec=10.0)
    segments.load_intervals(intervals)
    app.processEvents()

    ribbon = _find(ov, "modalityRibbon")
    assert ribbon.property("hasData") is True
    assert _tier_text(ov, "Q3") == "50.0%"
    assert _tier_text(ov, "Q2") == "30.0%"
    assert _tier_text(ov, "Q1") == "20.0%"
    # Q0 never occurred: the segmenter omits it from tierFractions entirely, and 0.0% of the
    # recording really was Q0. That is a measured zero -- it must NOT read as an em-dash.
    assert _tier_text(ov, "Q0") == "0.0%"
    assert ribbon.property("usableText") == "80.0%"

    # and the KPI row agrees, off the same real intervals
    assert _find(ov, "kpiUsable").property("value") == "80.0%"
    assert _find(ov, "kpiFlaggedQ0").property("value") == "0.0%"
    assert _find(ov, "kpiSegments").property("value") == "3"   # Q3 | Q2 | Q1 runs
