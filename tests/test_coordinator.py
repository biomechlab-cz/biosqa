"""End-to-end wiring test: open a recording -> inference -> viewmodels (Plan 2 §7/§9).

Guards the Coordinator glue the app audit found missing. Runs the real ONNX models
against a synthetic WFDB recording, headless (offscreen Qt), and asserts the full chain:
modality detection, segment population, short tier codes (the C1 fix), plot decimation,
selection, and CSV export.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import time
from pathlib import Path

import numpy as np
import pytest

wfdb = pytest.importorskip("wfdb")
pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QThreadPool  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from biosqa.io.loaders import detect_modality, open_recording  # noqa: E402
from biosqa.main import build_engine  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _write_ecg(tmp: Path, fs: int = 250, dur: int = 60, noisy=(20, 30)) -> str:
    """Write a synthetic single-lead ECG WFDB record with a noisy stretch."""
    L = fs * dur
    t = np.arange(L) / fs
    sig = 0.6 * np.sin(2 * np.pi * 1.1 * t) + 0.02 * np.random.default_rng(0).standard_normal(L)
    a, b = noisy
    sig[fs * a:fs * b] += 3.0 * np.random.default_rng(1).standard_normal(fs * (b - a))
    wfdb.wrsamp("rec", fs=fs, units=["mV"], sig_name=["II"],
                p_signal=sig[:, None].astype(float), write_dir=str(tmp))
    return str(tmp / "rec.hea")


def _drain(qapp, predicate, timeout_s: float = 20.0) -> None:
    QThreadPool.globalInstance().waitForDone(int(timeout_s * 1000))
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        qapp.processEvents()
        if predicate():
            return
        time.sleep(0.02)


def test_detect_modality_ecg(tmp_path):
    handle = open_recording(_write_ecg(tmp_path))
    assert detect_modality(handle) == "ecg"


def test_stale_load_result_is_dropped_by_generation_guard(qapp):
    """The generation guard must DROP a first-recording load result that arrives AFTER a second open
    superseded it. The stale ordering is forced EXPLICITLY (emit current, then stale): the previous test
    just opened a 30 s then a 90 s record and asserted 90 s won — but the larger record's load simply
    finishes last, so it passed even with the guard disabled, and never asserted the stale one was dropped."""
    from biosqa.workers.signals import LoadResampleWorkerSignals
    engine = build_engine()
    coordinator = engine._biosqa_controllers[11]

    started: list = []
    coordinator._start_inference = lambda m, s: started.append((m, int(getattr(s, "size", 0))))

    def _payload(modality, n):
        return {"modality": modality, "sig": np.zeros(n, dtype=np.float32), "handle": None,
                "plot_channels": ["II"], "plot_fs": 250.0,
                "full_t": np.arange(10, dtype=np.float64), "full_y": np.zeros(10, dtype=np.float32),
                "trace_lo": -1.0, "trace_hi": 1.0, "n_samples_primary": n}

    coordinator._recording_gen = 2                          # two opens advanced the generation 1 (A) → 2 (B)
    current = LoadResampleWorkerSignals(); current._rgen = 2   # B, the live recording
    stale = LoadResampleWorkerSignals(); stale._rgen = 1       # A, superseded by B
    current.ready.connect(coordinator._on_loaded)
    stale.ready.connect(coordinator._on_loaded)

    current.ready.emit(_payload("ecg", 90))                 # B applies (matches current generation)
    assert coordinator._current[0] == "ecg" and coordinator._current[1].size == 90

    stale.ready.emit(_payload("eeg", 30))                   # A arrives LATE — the guard must drop it
    assert coordinator._current[0] == "ecg" and coordinator._current[1].size == 90   # NOT overwritten by A
    assert started == [("ecg", 90)]                          # inference ran once (for B); A was rejected
    assert coordinator._recording_gen == 2


def test_streaming_path_end_to_end(qapp, tmp_path, monkeypatch):
    """Force the out-of-core streaming branch on a small record and assert it populates segments,
    keeps no in-memory signal, and can still audit (previously untested — a regression shipped green)."""
    import biosqa.inference.streaming as st
    monkeypatch.setattr(st, "LARGE_RECORD_SAMPLES", 100)     # 60 s record (15000 samples) → streams
    rec = _write_ecg(tmp_path)
    engine = build_engine()
    (_ac, recordings, _ch, signal_view, segments, _sel, inference,
     _mc, _ex, _g, _st, coordinator) = engine._biosqa_controllers
    recordings.open(rec)
    _drain(qapp, lambda: segments.totalCount > 0 and coordinator._current_stream is not None)
    assert segments.totalCount > 0
    assert coordinator._current is None and coordinator._current_stream is not None   # streamed
    assert signal_view.durationSec == pytest.approx(60.0, abs=1.0)
    # audit re-reads the window from the handle for a streamed record (no in-memory sig)
    assert coordinator._audit_window(0.0, 10.0) is not None


def test_intervals_carry_uncertainty(qapp, tmp_path):
    """The predictive-uncertainty wiring (softmax entropy → QualityInterval.uncertainty) is live."""
    rec = _write_ecg(tmp_path)
    engine = build_engine()
    (_ac, recordings, _ch, _sv, segments, *_rest) = engine._biosqa_controllers
    recordings.open(rec)
    _drain(qapp, lambda: segments.totalCount > 0)
    assert any(iv.uncertainty > 0 for iv in segments._all_intervals)


def test_open_runs_inference_and_populates_viewmodels(qapp, tmp_path):
    rec = _write_ecg(tmp_path)
    engine = build_engine()
    (app_controller, recordings, channels, signal_view, segments,
     selection, inference, model_card, exporter, guard, settings, coordinator) = engine._biosqa_controllers

    dq_events = []
    guard.dataQualityChanged.connect(lambda: dq_events.append(1))

    recordings.open(rec)
    _drain(qapp, lambda: segments.rowCount() > 0)

    # the chain populated end to end
    assert recordings.rowCount() == 1
    assert channels.rowCount() == 1
    assert segments.rowCount() > 0
    assert "ECG" in inference.statusText
    assert inference.modelVersion  # a model card version was surfaced

    # C1: tiers are SHORT codes (Q0..Q3), so filters are non-empty
    tiers = {segments.data(segments.index(r, 0), segments.TierRole) for r in range(segments.rowCount())}
    assert tiers and all(len(t) == 2 and t[0] == "Q" for t in tiers)
    segments.setFilter("all")
    assert segments.rowCount() > 0

    # the plot bound its primary-channel cache on open (QtCharts path), and hover reads from it
    assert len(signal_view._caches) >= 1
    assert signal_view.durationSec == pytest.approx(60.0, abs=1.0)
    assert isinstance(signal_view.valueAt(5.0), float)   # P3: reads the lane cache, not legacy curves

    # guard + data-quality wired end to end (worker -> coordinator -> guard controller)
    assert dq_events, "GuardController never received a data-quality report"
    assert 0.0 <= guard.completeness <= 1.0
    assert isinstance(guard.dataQualityFlags, list)

    # P4: selection lights up the inspector chain
    selection.selectByIndex(0)
    assert selection.selectedSegment is not None
    assert selection.selectedSegment.tier[0] == "Q"

    # human-in-the-loop override is recorded (active-learning reverse channel)
    selection.relabel("Q1")
    assert len(selection.collected_overrides()) == 1

    # C6: overview aggregates are real
    fractions = segments.tierFractions
    assert isinstance(fractions, dict) and abs(sum(fractions.values()) - 1.0) < 1e-6

    # F2: CSV export writes the intervals
    out = tmp_path / "intervals.csv"
    exporter.exportToPath(str(out), "csv")
    assert out.exists() and out.read_text().startswith("start_sec,end_sec,tier")


def test_audit_disabled_via_settings(qapp):
    """With LLM audit disabled in settings, requestAudit resolves to a 'disabled' error, no worker."""
    engine = build_engine()
    (app_controller, recordings, channels, signal_view, segments,
     selection, inference, model_card, exporter, guard, settings, coordinator) = engine._biosqa_controllers
    settings.setAuditEnabled(False)
    guard.requestAudit(0.0, 5.0, "Q3", 0.9)      # auditRequested -> coordinator (same-thread, synchronous)
    _drain(qapp, lambda: guard.auditError, timeout_s=3.0)
    assert guard.auditError and "disabled" in guard.auditText.lower()
    settings.setAuditEnabled(True)               # restore (shared isolated backend)


def test_load_resample_task_emits_ready_offthread(qapp, tmp_path):
    """LoadResampleTask reads + resamples + builds the plot cache off the GUI thread (async open)."""
    from biosqa.workers.qt_threads import LoadResampleTask
    from biosqa.workers.signals import LoadResampleWorkerSignals

    handle = open_recording(_write_ecg(tmp_path, fs=360, dur=20, noisy=(5, 10)))  # 360 Hz -> model 250 Hz
    carrier = LoadResampleWorkerSignals()
    got: list = []
    carrier.ready.connect(lambda p: got.append(p))
    LoadResampleTask(handle, "II", "II", ["II"], 250.0, "ecg", carrier).run()
    qapp.processEvents()
    assert got, "LoadResampleTask never emitted ready"
    p = got[0]
    assert p["modality"] == "ecg" and p["fs_out"] == 250.0
    assert p["sig"].size > 0 and abs(p["sig"].size - 250 * 20) < 300   # resampled 360->250
    assert p["full_t"] is not None and p["full_y"] is not None and p["full_t"].size >= 2


def test_overlap_change_reruns_inference_on_open_recording(qapp, tmp_path):
    """Changing an analysis setting re-segments the OPEN recording live (no re-open / re-read)."""
    rec = _write_ecg(tmp_path)
    engine = build_engine()
    ctl = engine._biosqa_controllers
    recordings, segments, settings, coordinator = ctl[1], ctl[4], ctl[10], ctl[11]

    recordings.open(rec)
    _drain(qapp, lambda: segments.rowCount() > 0)
    assert segments.rowCount() > 0

    calls: list = []
    orig = coordinator._start_inference
    def _spy(m, s):
        calls.append(m); orig(m, s)
    coordinator._start_inference = _spy

    # pick a DIFFERENT overlap so windowOverlapChanged actually fires
    settings.setWindowOverlap(0.25 if settings.windowOverlap != 0.25 else 0.0)
    _drain(qapp, lambda: len(calls) >= 1 and segments.rowCount() > 0, timeout_s=6.0)
    assert calls, "changing window overlap did not re-run inference on the open recording"
    assert segments.rowCount() > 0


def test_pan_zoom_does_not_raise(qapp, tmp_path):
    """setView (the pan/zoom hot path) before a recording is open must no-op, not raise."""
    engine = build_engine()
    signal_view = engine._biosqa_controllers[3]
    signal_view.setView(5.0, 15.0)   # no recording bound yet → no-op
    assert signal_view.viewStartSec == 5.0 and signal_view.viewEndSec == 15.0
