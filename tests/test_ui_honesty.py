"""Regression: the UI must never display an invented quality/latency/artifact value.

Five surfaces used to show fabricated data as if it were measured (all reachable with NO
recording open, since the activity rail has no enabled gate on its nav buttons):

  * ``overview/DonutChart.qml``   -- a hardcoded ``_static`` tier distribution (48/35.5/10/6.5%)
    was drawn whenever no inference had run, "so the tile is never blank".
  * ``overview/ArtifactBars.qml`` -- likewise a hardcoded artifact table (Baseline wander 412, ...).
  * ``workspace/HoverTooltip.qml`` -- defaulted to "Q1 / 71% conf", and ``WaveformChart`` never wrote
    the else-branch, so the tooltip also carried the LAST segment's grade onto ungraded time.
  * ``inspector/SegmentInspectorView.qml`` -- labelled the model's own prediction "YOU SET" on
    segments the user had never reviewed.
  * ``TopBar.qml`` -- fell back to a hardcoded "2.1 ms/win" latency that nothing had measured.

These drive the REAL QML scene (``build_engine``), so a re-introduced literal fails here rather
than in front of a clinician.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QObject, QPointF
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from biosqa.inference.segmenter import QualityInterval
from biosqa.main import build_engine
from biosqa.viewmodels.inference_status import InferenceStatusController
from biosqa.viewmodels.selection_controller import SelectionController


@pytest.fixture(scope="module")
def scene():
    """The real QML scene with no recording open -- the state a user lands in on launch."""
    app = QApplication.instance() or QApplication([])
    engine = build_engine()
    assert engine.rootObjects(), "engine failed to load QML"
    QTest.qWait(300)          # let the scene lay out: items are 0-height until the first polish
    yield app, engine, engine.rootObjects()[0]
    # Tear the QML scene down WHILE the context controllers are still alive and flush the
    # deferred deletes (same order as the shipped `install_teardown_guard`); dropping the
    # engine without this segfaults on interpreter GC.
    for obj in engine.rootObjects():
        obj.deleteLater()
    app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()


def _ctl(engine, cls):
    """The context controller of type `cls` (looked up by type, not tuple position)."""
    for c in engine._biosqa_controllers:
        if isinstance(c, cls):
            return c
    raise AssertionError(f"no {cls.__name__} on the engine")


def _find(root, name):
    obj = root.findChild(QObject, name)
    assert obj is not None, f"no object named {name!r} in the scene"
    return obj


def _goto(app, engine, view):
    from biosqa.viewmodels.app_controller import AppController
    _ctl(engine, AppController).go(view)
    QTest.qWait(50)           # the Loader swaps the view; let the new one lay out


# -- Overview: no inference => no numbers ------------------------------------------


def test_donut_paints_no_distribution_before_inference(scene):
    app, engine, root = scene
    _goto(app, engine, "overview")
    donut = _find(root, "qualityDonut")

    assert donut.property("hasData") is False
    # the centre readout is an em-dash, never a plausible "83.5%" usable share
    assert donut.property("usableText") == "—"


def test_artifact_bars_are_empty_before_inference(scene):
    app, engine, root = scene
    _goto(app, engine, "overview")
    bars = _find(root, "artifactBars")

    assert bars.property("hasData") is False
    assert bars.property("count") == 0        # no rows painted at all, not a mock table


# -- Workspace: ungraded time carries no grade --------------------------------------


def test_hover_tooltip_starts_with_no_quality(scene):
    app, engine, root = scene
    _goto(app, engine, "workspace")
    tip = _find(root, "hoverTooltip")

    assert tip.property("hasQuality") is False
    assert tip.property("qualityTier") == ""


def test_hovering_ungraded_time_clears_a_stale_grade(scene):
    """The real defect: WaveformChart never wrote the else-branch, so the tooltip kept the LAST
    segment's tier/confidence and showed it over time that has no grade at all. Hover the chart
    with no segments loaded -- `segments.segmentAt()` returns None -> the grade must be cleared."""
    app, engine, root = scene
    _goto(app, engine, "workspace")
    chart = _find(root, "waveformChart")
    tip = _find(root, "hoverTooltip")

    tip.setProperty("hasQuality", True)       # as if the cursor had just left a Q0 segment
    tip.setProperty("qualityTier", "Q0")
    tip.setProperty("confidence", 0.93)

    centre = chart.mapToScene(QPointF(chart.width() / 2, chart.height() / 2)).toPoint()
    QTest.mouseMove(root, centre)
    app.processEvents()

    assert tip.property("hasQuality") is False
    assert tip.property("qualityTier") == ""
    assert tip.property("confidence") == 0


# -- Inspector: an unreviewed model prediction is not a human verdict -----------------


def test_unreviewed_segment_is_not_labelled_you_set(scene):
    app, engine, root = scene
    selection = _ctl(engine, SelectionController)
    selection.select(QualityInterval(0.0, 10.0, "Q2", 0.81))
    _goto(app, engine, "inspector")

    assert _find(root, "verdictLabel").property("text") == "NOT REVIEWED"
    assert _find(root, "verdictTier").property("text") == "—"
    selection.clear()


def test_reviewed_segment_shows_the_tier_the_reviewer_chose(scene):
    """Clicking a tier row sets `userTier`; only then is the verdict a human one."""
    app, engine, root = scene
    selection = _ctl(engine, SelectionController)
    selection.select(QualityInterval(0.0, 10.0, "Q2", 0.81))
    _goto(app, engine, "inspector")

    view = _find(root, "segmentInspector")
    view.setProperty("userTier", "Q0")        # what the tier-override row's TapHandler does
    app.processEvents()

    assert _find(root, "verdictLabel").property("text") == "YOU SET"
    assert _find(root, "verdictTier").property("text") == "Q0"
    selection.clear()


def test_verdict_never_echoes_the_models_own_grade(scene):
    """A relabel recorded outside this view (e.g. the workspace panel) sets `overridden` but
    carries no readable tier -- the verdict must stay "—" rather than reprint the model's Q2."""
    app, engine, root = scene
    selection = _ctl(engine, SelectionController)
    selection.select(QualityInterval(0.0, 10.0, "Q2", 0.81))
    _goto(app, engine, "inspector")

    selection.relabel("Q0")
    app.processEvents()

    assert _find(root, "verdictTier").property("text") != "Q2"
    selection.clear()


# -- Top bar: nothing measured => no claim about the model ----------------------------


def test_status_pill_omits_unmeasured_latency_and_precision(scene):
    app, engine, root = scene
    text = _find(root, "modelStatusText").property("text")

    assert "ms/win" not in text, f"pill invents a latency: {text!r}"
    assert "FP32" not in text and "INT8" not in text, f"pill invents a precision: {text!r}"


def test_status_pill_shows_a_measured_latency(scene):
    app, engine, root = scene
    inference = _ctl(engine, InferenceStatusController)
    inference.report("ECG · 12 segments", "v1", 3.4, "INT8")
    app.processEvents()

    text = _find(root, "modelStatusText").property("text")
    assert "3.4 ms/win" in text
    assert "INT8" in text

    inference.report("No model loaded", "", 0.0, "")
    app.processEvents()


# -- InferenceStatusController: the "not measured" contract ---------------------------


def test_inference_status_starts_unmeasured():
    st = InferenceStatusController()
    assert st.latencyMs == 0.0                # 0.0 == never measured (the pill drops the clause)
    assert st.precision == ""                 # "" == unknown quantization


def test_report_keeps_precision_when_caller_omits_it():
    """A 3-arg report() (every existing coordinator call) must neither clear a known precision
    nor invent one."""
    st = InferenceStatusController()
    st.setPrecision("FP32")
    st.report("Running ECG…", "v1", 0.0)

    assert st.precision == "FP32"
    assert st.latencyMs == 0.0
