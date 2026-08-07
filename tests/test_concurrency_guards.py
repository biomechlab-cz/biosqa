"""Adversarial concurrency regressions: the generation guards must FAIL CLOSED, and the heavy
per-selection work must not run on the GUI thread.

Every guard in this app is of the form "is the emitting carrier still the live generation?". The
carriers are QObjects owned by Python lists that the Coordinator/SignalViewController PRUNE on every
open — so the superseded carrier is collected while its queued signal is still in the event queue,
``sender()`` comes back None, and a ``getattr(sender, attr, current)`` default turns "unidentifiable"
into "live". The tests below construct that exact condition (a DIRECT slot call has ``sender() is
None`` for the same reason a collected carrier does) and assert the result is dropped.
"""

from __future__ import annotations

import os
import threading

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

wfdb = pytest.importorskip("wfdb")
pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtWidgets import QApplication  # noqa: E402

from biosqa.main import build_engine  # noqa: E402
from tests.test_coordinator import _drain, _write_ecg  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _payload(modality: str, n: int, gen=None) -> dict:
    p = {"modality": modality, "sig": np.zeros(n, dtype=np.float32), "handle": None,
         "plot_channels": ["II"], "plot_fs": 250.0,
         "full_t": np.arange(10, dtype=np.float64), "full_y": np.zeros(10, dtype=np.float32),
         "trace_lo": -1.0, "trace_hi": 1.0, "n_samples_primary": n}
    if gen is not None:
        p["_rgen"] = gen
    return p


# --- the critical one: a result whose carrier is gone must be REJECTED, not accepted -------------

def test_load_result_from_a_collected_carrier_is_rejected(qapp):
    """THE fail-open bug. ``_stale`` read ``getattr(self.sender(), "_rgen", current)``: with the
    carrier collected (QThreadPool auto-delete + ``_invalidate``'s prune) ``sender()`` is None, the
    default WAS the live generation, and the check became ``current != current`` -> not stale. So the
    superseded recording's signal was installed — and re-graded — under the live recording's identity.

    A direct call reproduces it exactly: ``sender()`` is None there for the same reason."""
    engine = build_engine()
    coordinator = engine._biosqa_controllers[11]
    started: list = []
    coordinator._start_inference = lambda m, s: started.append((m, int(getattr(s, "size", 0))))

    coordinator._recording_gen = 2
    coordinator._current = ("ecg", np.zeros(90, dtype=np.float32))   # B, the live recording

    coordinator._on_loaded(_payload("eeg", 30))          # sender() is None -> unidentifiable

    assert coordinator._current[0] == "ecg" and coordinator._current[1].size == 90
    assert started == []                                  # and it did NOT re-enter inference


def test_load_result_generation_travels_in_the_payload(qapp):
    """The robust half of the fix: LoadResampleTask stamps the payload, so the check does not depend
    on the carrier being alive at delivery time at all."""
    engine = build_engine()
    coordinator = engine._biosqa_controllers[11]
    started: list = []
    coordinator._start_inference = lambda m, s: started.append((m, int(getattr(s, "size", 0))))
    coordinator._recording_gen = 2

    coordinator._on_loaded(_payload("ecg", 90, gen=2))    # live generation -> applies
    assert coordinator._current[1].size == 90
    coordinator._on_loaded(_payload("eeg", 30, gen=1))    # superseded generation -> dropped
    assert coordinator._current[0] == "ecg" and coordinator._current[1].size == 90
    assert started == [("ecg", 90)]


def test_saliency_result_from_a_collected_carrier_is_rejected(qapp):
    """The saliency slot's ``_rgen`` half was fail-open too; only its second check (``_sgen``, whose
    default happened to be ``-1`` rather than the live value) saved it. Pin the behaviour: an
    unidentifiable saliency result must never paint one segment's heatmap over another's trace."""
    engine = build_engine()
    ctl = engine._biosqa_controllers
    guard, coordinator = ctl[9], ctl[11]
    coordinator._on_saliency_ready({"map": [0.5, 0.5], "n": 2, "attribution": None})
    assert guard.saliencyMap == []


def test_channel_cache_from_a_collected_carrier_is_rejected(qapp):
    """SignalViewController had its own inline copy of the idiom — fixing the Coordinator alone left
    recording A's samples landing in recording B's same-named lane (["MLII","V5"] & friends)."""
    from biosqa.viewmodels.signal_view_controller import SignalViewController
    from biosqa.workers.signals import ChannelCacheWorkerSignals

    sv = SignalViewController()
    sv._lanes = ["V5"]
    t = np.arange(10, dtype=np.float64)
    y = np.full(10, 111.0, dtype=np.float32)          # "recording A"

    sv._on_channel_cache("V5", t, y, -1.0, 1.0)       # sender() is None -> must be rejected
    assert "V5" not in sv._caches

    carrier = ChannelCacheWorkerSignals()             # positive control: a live, current carrier lands
    carrier._gen = sv._load_gen
    carrier.ready.connect(sv._on_channel_cache)
    carrier.ready.emit("V5", t, y, -1.0, 1.0)
    assert "V5" in sv._caches and float(sv._caches["V5"][1][0]) == 111.0

    carrier2 = ChannelCacheWorkerSignals()            # ...and a superseded one still does not
    carrier2._gen = sv._load_gen - 1
    carrier2.ready.connect(sv._on_channel_cache)
    carrier2.ready.emit("V5", t, np.full(10, 222.0, dtype=np.float32), -1.0, 1.0)
    assert float(sv._caches["V5"][1][0]) == 111.0


# --- cancellation / shutdown ----------------------------------------------------------------------

def test_load_resample_task_honours_its_cancel_token(qapp, tmp_path):
    """A superseded load used to read + resample the WHOLE channel and emit anyway (it was the only
    task dispatched without a token)."""
    from biosqa.io.loaders import open_recording
    from biosqa.workers.qt_threads import LoadResampleTask
    from biosqa.workers.signals import LoadResampleWorkerSignals

    handle = open_recording(_write_ecg(tmp_path, dur=20, noisy=(5, 10), name="cancelload"))
    carrier = LoadResampleWorkerSignals()
    got: list = []
    carrier.ready.connect(got.append)
    carrier.failed.connect(lambda *a: got.append(a))

    cancel = threading.Event()
    cancel.set()
    LoadResampleTask(handle, "II", "II", ["II"], 250.0, "ecg", carrier, cancel=cancel, gen=0).run()
    qapp.processEvents()
    assert got == [], "a cancelled load still published its result"

    LoadResampleTask(handle, "II", "II", ["II"], 250.0, "ecg", carrier,
                     cancel=threading.Event(), gen=7).run()      # not cancelled -> emits, stamped
    qapp.processEvents()
    assert got and got[0]["_rgen"] == 7


def test_quit_cancels_in_flight_work(qapp):
    """Nothing cancelled in-flight work at quit, and the pool destructor then waited for it — the
    window vanished while the process stayed alive for the rest of a streamed read / ollama timeout."""
    engine = build_engine()
    coordinator = engine._biosqa_controllers[11]
    tok = coordinator._new_cancel_token()
    assert not tok.is_set() and not coordinator._quitting.is_set()
    coordinator.shutdown()
    assert tok.is_set(), "quit did not cancel the in-flight inference/stream"
    assert coordinator._quitting.is_set(), "quit did not short-circuit queued LLM audits"


def test_audit_task_short_circuits_when_cancelled(qapp):
    """The audit pool is single-threaded and an ollama call is up to ``samples`` x timeout long, so a
    queued audit must not start one during teardown."""
    from biosqa.workers.qt_threads import AuditTask
    from biosqa.workers.signals import AuditWorkerSignals

    carrier = AuditWorkerSignals()
    got: list = []
    carrier.auditReady.connect(got.append)
    carrier.failed.connect(got.append)
    cancel = threading.Event()
    cancel.set()
    AuditTask(None, np.zeros(64, dtype=np.float32), {}, None, carrier, cancel=cancel).run()
    qapp.processEvents()
    assert got == []      # neither a judgment nor the AttributeError a live run would have raised


def test_emit_through_a_destroyed_carrier_does_not_escape_run(qapp):
    """At teardown the carrier can be destroyed under a running worker; the result emit raised, and
    then the ``failed.emit`` in the handler raised the SAME error from inside ``except``, escaping the
    Python override of ``QRunnable::run()``."""
    from PySide6.QtCore import QEvent

    from biosqa.workers.qt_threads import _emit
    from biosqa.workers.signals import LoadResampleWorkerSignals

    carrier = LoadResampleWorkerSignals()
    failed = carrier.failed
    carrier.deleteLater()
    qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    del carrier
    _emit(failed, "ecg", "boom")      # must not raise, whatever state the carrier is in


def test_emit_does_not_swallow_a_runtime_error_that_is_not_a_dead_carrier(qapp):
    """``_emit`` swallowed EVERY ``RuntimeError``. Emits are queued today, so nothing else could get
    in — but the first task run inline on the GUI thread (a test double, a future synchronous path)
    makes the connection DIRECT, and a genuine error raised inside the receiving slot would then be
    discarded from a pool thread with no log. Only the teardown case is the teardown case."""
    from biosqa.workers.qt_threads import _emit

    class _Signal:
        """Stands in for a bound Signal so the RuntimeError comes out of ``emit()`` itself — PySide
        prints a slot's exception rather than propagating it, which is precisely why a swallowed
        ``emit()`` error would leave no trace at all."""
        def __init__(self, message):
            self.message = message
            self.calls = 0

        def emit(self, *_a):
            self.calls += 1
            raise RuntimeError(self.message)

    gone = _Signal("Signal source has been deleted")
    _emit(gone, "ecg", "boom")                          # teardown: still swallowed
    assert gone.calls == 1

    collected = _Signal("Internal C++ object (LoadResampleWorkerSignals) already deleted.")
    _emit(collected, "ecg", "boom")                     # the sibling wording: also swallowed
    assert collected.calls == 1

    real = _Signal("cannot send events to objects owned by a different thread")
    with pytest.raises(RuntimeError, match="different thread"):
        _emit(real, "ecg", "boom")                      # a GENUINE error must not vanish


# --- the GUI-thread freeze ------------------------------------------------------------------------

def test_sqi_breakdown_is_computed_off_the_gui_thread(qapp, tmp_path):
    """``sqiRequested`` was a DIRECT connection between two GUI-thread QObjects, running the whole
    classical-SQI bank TWICE (raw + band-pass-filtered) plus the usability verdicts over the selected
    span. A long clean record is ONE segment, so selecting it froze the window for ~10-20 s.

    The request must now only DISPATCH: nothing is computed before the event loop turns again."""
    rec = _write_ecg(tmp_path, name="sqirec")
    engine = build_engine()
    ctl = engine._biosqa_controllers
    recordings, segments, guard = ctl[1], ctl[4], ctl[9]

    recordings.open(rec)
    _drain(qapp, lambda: segments.totalCount > 0)
    assert guard.sqiBreakdown == []

    guard.requestSqi(0.0, 20.0)          # synchronous emit -> the Coordinator may only start a task
    assert guard.sqiBreakdown == [], "the SQI bank still ran inline on the GUI thread"

    _drain(qapp, lambda: bool(guard.sqiBreakdown), timeout_s=20.0)
    assert guard.sqiBreakdown, "the off-thread SQI breakdown never landed"


def test_sqi_result_for_a_superseded_selection_is_dropped(qapp):
    """Off-thread means late, so the selection guard has to hold — and fail closed."""
    engine = build_engine()
    ctl = engine._biosqa_controllers
    guard, coordinator = ctl[9], ctl[11]
    coordinator._on_sqi_ready({"rows": [{"name": "x", "value": 1.0}], "filtered": [],
                               "consensus": 0.5, "usability": []})   # sender() is None
    assert guard.sqiBreakdown == []


def test_selection_driven_sqi_fills_the_breakdown_and_usability_panels(qapp, tmp_path):
    """THE regression the off-thread move introduced: both panels were PERMANENTLY EMPTY in the app.

    The result was guarded on a selection-generation COUNTER stamped at dispatch — but QML requests
    the SQI from the very ``selection.selectedSegmentChanged`` signal that bumps that counter, and
    QML's handler runs FIRST. So the stamp was always one behind the live value and every
    selection-driven result was rejected as stale. The suite missed it because the only SQI test
    called ``guard.requestSqi(...)`` directly — the one path that still worked.

    This test therefore drives the REAL path (``selectByIndex`` -> ``selectedSegmentChanged`` -> the
    QML ``Connections`` handler in QualityInspectorPanel.qml -> ``requestSqi``) and asserts the panels
    fill. EEG, because ``usability_verdicts`` is per-band for EEG and ``[]`` for ECG by design."""
    from biosqa.io.synth import write_test_recording

    rec = write_test_recording("eeg", tmp_path, minutes=1.0)
    engine = build_engine()
    ctl = engine._biosqa_controllers
    recordings, segments, selection, guard, coordinator = ctl[1], ctl[4], ctl[5], ctl[9], ctl[11]

    recordings.open(rec)
    _drain(qapp, lambda: segments.totalCount > 0, timeout_s=60.0)
    assert segments.totalCount > 0, "inference never produced segments"
    guard.setSqiBreakdown([])
    guard.setUsability([])

    selection.selectByIndex(0)                 # the real path — nothing calls requestSqi here
    assert coordinator._sqi_carriers, "selecting a segment did not request the SQI breakdown"
    _drain(qapp, lambda: bool(guard.sqiBreakdown), timeout_s=60.0)
    assert guard.sqiBreakdown, "the selection-driven SQI breakdown was rejected as stale"
    assert guard.usabilityVerdicts, "the usability panel stayed empty"

    if segments.totalCount > 1:                # ...and it keeps working on every later selection
        guard.setSqiBreakdown([])
        selection.selectByIndex(1)
        _drain(qapp, lambda: bool(guard.sqiBreakdown), timeout_s=60.0)
        assert guard.sqiBreakdown, "the SECOND selection's breakdown was rejected as stale"


def test_sqi_result_for_a_span_nobody_is_looking_at_is_dropped(qapp, tmp_path):
    """The guard still has to reject a genuinely stale result — it is now the requested SPAN, not a
    counter, that is checked against what is on screen (the counter could be bumped out of order;
    the span cannot). A live carrier stamped with someone else's span must not paint the panels."""
    from biosqa.workers.qt_threads import SqiWorkerSignals

    rec = _write_ecg(tmp_path, name="sqispanrec")
    engine = build_engine()
    ctl = engine._biosqa_controllers
    recordings, segments, selection, guard, coordinator = ctl[1], ctl[4], ctl[5], ctl[9], ctl[11]
    recordings.open(rec)
    _drain(qapp, lambda: segments.totalCount > 0)
    selection.selectByIndex(0)
    _drain(qapp, lambda: bool(guard.sqiBreakdown), timeout_s=30.0)
    live = coordinator._live_sqi_span()
    assert live is not None

    payload = {"rows": [{"name": "stale", "value": 1.0}], "filtered": [],
               "consensus": 0.5, "usability": [{"label": "stale", "usable": True, "detail": ""}]}
    guard.setSqiBreakdown([])
    guard.setUsability([])

    stale = SqiWorkerSignals()                      # a LIVE carrier — only its span is wrong
    stale._rgen = coordinator._recording_gen
    stale._span = (live[0] + 1234.0, live[1] + 1234.0)
    stale.sqiReady.connect(coordinator._on_sqi_ready)
    stale.sqiReady.emit(payload)
    assert guard.sqiBreakdown == [], "a result for a span nobody is looking at was accepted"

    fresh = SqiWorkerSignals()                      # ...and the live span still lands
    fresh._rgen = coordinator._recording_gen
    fresh._span = live
    fresh.sqiReady.connect(coordinator._on_sqi_ready)
    fresh.sqiReady.emit(payload)
    assert guard.sqiBreakdown, "the result for the SELECTED span was dropped"
    assert guard.usabilityVerdicts

    unstamped = SqiWorkerSignals()                  # fail closed: no span == unidentifiable
    unstamped._rgen = coordinator._recording_gen
    guard.setSqiBreakdown([])
    unstamped.sqiReady.connect(coordinator._on_sqi_ready)
    unstamped.sqiReady.emit(payload)
    assert guard.sqiBreakdown == [], "an unstamped result was let through"


# --- status honesty -------------------------------------------------------------------------------

def test_streaming_notice_does_not_erase_the_segmentation_status(qapp, tmp_path, monkeypatch):
    """On the streaming path ``notice`` landed right after ``intervalsReady`` and REPLACED the status
    line, so the recordings whose analysis takes longest were the only ones that never reported their
    segment count, model version or latency (and lost the 'N reviews dropped' warning with it)."""
    import biosqa.inference.streaming as st

    monkeypatch.setattr(st, "LARGE_RECORD_SAMPLES", 100)   # force the out-of-core path
    rec = _write_ecg(tmp_path, name="noticerec")
    engine = build_engine()
    ctl = engine._biosqa_controllers
    recordings, segments, inference = ctl[1], ctl[4], ctl[6]

    recordings.open(rec)
    _drain(qapp, lambda: segments.totalCount > 0 and "streaming analysis" in inference.statusText,
           timeout_s=20.0)
    assert "streaming analysis" in inference.statusText          # the notice is still shown...
    assert "segments" in inference.statusText                    # ...alongside the segmentation
    assert inference.modelVersion, "the streaming notice erased the model version"
    # ...and the third number on that line has to be real too. Both streaming dispatch sites record a
    # window count of 0 (the count is not knowable before the record has been streamed), and
    # `_on_intervals` computes latency only when the count is > 0 — so a streamed recording reported a
    # flat latencyMs = 0.0 while the in-memory path on the same signal reported ~80 ms/window.
    assert inference.latencyMs > 0.0, "the streamed run reported a per-window latency of 0.0 ms"


def test_model_card_signals_after_the_reset_completes(qapp, tmp_path):
    """``cardChanged`` was emitted BETWEEN ``beginResetModel`` and the rows being rebuilt, so a handler
    that queried the model saw the new card's windowSec next to the previous card's rows."""
    from biosqa.viewmodels.model_card_model import ModelCardModel

    models = [p for p in (build_engine()._biosqa_controllers[11]._models_dir).glob("*.model_card.json")]
    if len(models) < 2:
        pytest.skip("needs two exported model cards")

    mc = ModelCardModel()
    mc.load(str(models[0]))
    seen: list = []
    mc.cardChanged.connect(lambda: seen.append(
        (mc.windowSec, mc.rowCount(), mc.data(mc.index(0, 0), mc.ValueRole))))
    mc.load(str(models[1]))
    assert seen, "cardChanged was not emitted"
    window_sec, _n, modality = seen[-1]
    from biosqa.model.model_card import load_model_card
    card = load_model_card(str(models[1]))
    assert modality == card.modality                             # the rows are the NEW card's
    assert window_sec == pytest.approx(card.l_m / card.fs_hz)    # ...and so is windowSec
