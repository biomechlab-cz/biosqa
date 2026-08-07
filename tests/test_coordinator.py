"""End-to-end wiring test: open a recording -> inference -> viewmodels (Plan 2 §7/§9).

Guards the Coordinator glue the app audit found missing. Runs the real ONNX models
against a synthetic WFDB recording, headless (offscreen Qt), and asserts the full chain:
modality detection, segment population, short tier codes (the C1 fix), plot decimation,
selection, and CSV export.
"""

from __future__ import annotations

import csv
import json
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


def _write_ecg(tmp: Path, fs: int = 250, dur: int = 60, noisy=(20, 30), name: str = "rec") -> str:
    """Write a synthetic single-lead ECG WFDB record with a noisy stretch."""
    L = fs * dur
    t = np.arange(L) / fs
    sig = 0.6 * np.sin(2 * np.pi * 1.1 * t) + 0.02 * np.random.default_rng(0).standard_normal(L)
    a, b = noisy
    sig[fs * a:fs * b] += 3.0 * np.random.default_rng(1).standard_normal(fs * (b - a))
    wfdb.wrsamp(name, fs=fs, units=["mV"], sig_name=["II"],
                p_signal=sig[:, None].astype(float), write_dir=str(tmp))
    return str(tmp / f"{name}.hea")


def _write_ecg_2ch(tmp: Path, fs: int = 250, dur: int = 60, name: str = "rec2ch") -> str:
    """A 2-CHANNEL record whose ECG lead is NOT channel 0 (["RESP", "II"] — the shape of a 12-lead
    ECG too, where channel 0 is "I" but the preferred channel is "II"). Every other fixture here is
    single-channel, which is exactly why grading one channel and plotting/exporting another was
    invisible. RESP swings ~30x wider than the ECG lead, so the plotted trace identifies itself."""
    L = fs * dur
    t = np.arange(L) / fs
    ecg = 0.6 * np.sin(2 * np.pi * 1.1 * t) + 0.02 * np.random.default_rng(0).standard_normal(L)
    ecg[fs * 20:fs * 30] += 3.0 * np.random.default_rng(1).standard_normal(fs * 10)
    resp = 20.0 * np.sin(2 * np.pi * 0.2 * t)
    wfdb.wrsamp(name, fs=fs, units=["mV", "mV"], sig_name=["RESP", "II"],
                p_signal=np.stack([resp, ecg], axis=1).astype(float), write_dir=str(tmp))
    return str(tmp / f"{name}.hea")


def _drain(qapp, predicate, timeout_s: float = 20.0) -> None:
    QThreadPool.globalInstance().waitForDone(int(timeout_s * 1000))
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        qapp.processEvents()
        if predicate():
            return
        time.sleep(0.02)


def _export(exporter, path: Path, fmt: str) -> Path:
    """Run an export through the ExportController and return the path it actually wrote."""
    written: list = []
    exporter.exportSucceeded.connect(written.append)
    exporter.exportToPath(str(path), fmt)
    assert written, f"{fmt} export failed"
    return Path(written[-1])


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


def test_reviews_never_leak_across_recordings(qapp, tmp_path):
    """F2: a human review of recording A must die with A. The override store was process-global and
    keyed by TIME ALONE, so opening B re-attached A's reviewer tier + free-text note to whatever B
    had starting at the same second — fabricated human review, exported silently."""
    rec_a = _write_ecg(tmp_path, name="reca")
    rec_b = _write_ecg(tmp_path, noisy=(40, 50), name="recb")
    engine = build_engine()
    ctl = engine._biosqa_controllers
    recordings, segments, selection, exporter = ctl[1], ctl[4], ctl[5], ctl[8]

    recordings.open(rec_a)
    _drain(qapp, lambda: segments.totalCount > 0)
    i = next(i for i, iv in enumerate(segments._all_intervals) if iv.start_sec <= 10.0 < iv.end_sec)
    selection.selectByAllIndex(i)
    selection.relabel("Q0")
    selection.addNote("reviewer note belonging to recording A")
    assert len(selection.collected_overrides()) == 1        # the review really was recorded, for A

    recordings.open(rec_b)
    _drain(qapp, lambda: segments.totalCount > 0)
    assert selection.analysis_context().recording == rec_b
    assert selection.collected_overrides() == []            # A's review did NOT survive into B

    out = tmp_path / "b.csv"
    exporter.exportToPath(str(out), "csv")
    rows = list(csv.DictReader(out.open()))
    assert rows, "recording B exported no intervals"
    assert all(r["overridden"] == "False" for r in rows)    # no inherited override
    assert all(r["note"] == "" for r in rows)               # no inherited note


def test_failed_model_load_leaves_no_previous_recording_state(qapp, tmp_path):
    """F1: the open is an atomic state transition. A model that won't load used to return EARLY —
    before any invalidation — leaving recording A's segments/bands/overview on screen and exportable
    under recording B's name, fs and provenance.

    The WAVEFORM is per-recording state too, and the first round of this fix forgot it: it cleared the
    segments but never touched the SignalViewController, so A's trace stayed plotted (and A's handle
    stayed bound, so valueAt/curveForRange/zoom kept serving A's samples) under B's name, modality and
    fs. Assert the PLOT is empty, not just the segment table."""
    rec_a = _write_ecg(tmp_path, name="reca")
    rec_b = _write_ecg(tmp_path, noisy=(40, 50), name="recb")
    engine = build_engine()
    ctl = engine._biosqa_controllers
    recordings, channels, signal_view, segments = ctl[1], ctl[2], ctl[3], ctl[4]
    inference, model_card, exporter, coordinator = ctl[6], ctl[7], ctl[8], ctl[11]

    recordings.open(rec_a)
    _drain(qapp, lambda: segments.totalCount > 0)
    assert segments.totalCount > 0
    assert signal_view._caches and signal_view._handle is not None   # A really is on the plot
    assert signal_view.durationSec == pytest.approx(60.0, abs=1.0)
    # the QtCharts series the chart actually draws holds A's points
    assert any(s.count() > 0 for s in signal_view._series_map.values())

    def _boom(modality):
        raise RuntimeError(f"{modality}.onnx is missing")

    coordinator._runner = _boom
    recordings.open(rec_b)
    qapp.processEvents()

    assert segments.totalCount == 0 and segments.rowCount() == 0   # NOT recording A's segments
    assert model_card.rowCount() == 0                              # nor A's model provenance
    assert "Model load failed" in inference.statusText

    # the PLOT is empty: the drawn series holds NO points, no handle is bound, no samples are cached
    # and the duration is 0 — nothing of recording A can still be read through the signal view (nor
    # drawn on screen) under recording B's identity.
    assert all(s.count() == 0 for s in signal_view._series_map.values())   # the trace is GONE
    assert signal_view._handle is None
    assert signal_view._caches == {} and signal_view._channels == []
    assert signal_view.durationSec == 0.0
    assert signal_view.valueAt(5.0) == 0.0                         # hover reads nothing, not A
    assert signal_view.curveForRange(0.0, 10.0) is None             # inspector/grid minis read nothing
    # B's channels are listed (they exist) but NOTHING is graded — no analyzed badge
    assert channels.rowCount() == 1
    assert channels.data(channels.index(0, 0), channels.AnalyzedRole) is False
    assert coordinator._analyzed is None

    failed: list = []
    exporter.exportFailed.connect(failed.append)
    exporter.exportSelection("csv")
    assert failed, "export was still live with a stale recording's segments"
    out = tmp_path / "after_fail.csv"
    exporter.exportToPath(str(out), "csv")
    assert out.read_text().splitlines()[1:] == []                  # header only: no A intervals


def test_opening_a_new_recording_cancels_the_in_flight_inference(qapp, tmp_path, monkeypatch):
    """The cancel token must fire in the case it was built for. ``_start_inference`` cancelled the
    previous run, but ``_invalidate`` did NOT — so opening another recording left the old
    InferenceTask running to completion (including a second full ONNX pass in _recoverability) on a
    pool thread while the new recording's inference queued up behind it.

    The assertion is made BEFORE any event-loop turn after ``open(rec_b)``: ``recordingOpened`` is a
    same-thread emit, so ``_invalidate`` is the ONLY thing that can have run — the new recording's
    own ``_start_inference`` (which also cancels) needs the worker's queued ``ready`` signal, which
    cannot be delivered without ``processEvents``."""
    import biosqa.viewmodels.coordinator as coord_mod

    tokens: list = []
    real_task = coord_mod.InferenceTask

    def _spy(*args, **kwargs):
        tokens.append(kwargs.get("cancel"))      # the token the task ACTUALLY got
        return real_task(*args, **kwargs)

    monkeypatch.setattr(coord_mod, "InferenceTask", _spy)

    rec_a = _write_ecg(tmp_path, name="reca")
    rec_b = _write_ecg(tmp_path, noisy=(40, 50), name="recb")
    engine = build_engine()
    ctl = engine._biosqa_controllers
    recordings, segments, coordinator = ctl[1], ctl[4], ctl[11]

    recordings.open(rec_a)
    _drain(qapp, lambda: segments.totalCount > 0)
    assert tokens and tokens[0] is not None, "InferenceTask was dispatched without a cancel token"
    tok_a = tokens[0]
    assert tok_a is coordinator._infer_cancel and not tok_a.is_set()   # A's run owns the live token

    recordings.open(rec_b)                       # NO processEvents: only _invalidate can have run
    assert tok_a.is_set(), "opening a new recording did not cancel the previous inference"

    _drain(qapp, lambda: len(tokens) >= 2 and segments.totalCount > 0)
    assert tokens[1] is not None and tokens[1] is not tok_a           # B got its own fresh token


def test_streaming_task_gets_a_cancel_token_at_both_call_sites(qapp, tmp_path, monkeypatch):
    """StreamInferenceTask was constructed with ``cancel=None`` at BOTH call sites (open + settings
    re-run), so a superseded STREAMING job read the entire record anyway — the token existed on the
    task and was simply never passed. Assert the task receives a real token on open AND on a re-run,
    and that a new open fires it."""
    import biosqa.inference.streaming as st
    import biosqa.viewmodels.coordinator as coord_mod

    monkeypatch.setattr(st, "LARGE_RECORD_SAMPLES", 100)   # force the out-of-core path
    tasks: list = []
    real_task = coord_mod.StreamInferenceTask

    def _spy(*args, **kwargs):
        task = real_task(*args, **kwargs)
        tasks.append(task)
        return task

    monkeypatch.setattr(coord_mod, "StreamInferenceTask", _spy)

    rec_a = _write_ecg(tmp_path, name="streama")
    rec_b = _write_ecg(tmp_path, noisy=(40, 50), name="streamb")
    engine = build_engine()
    ctl = engine._biosqa_controllers
    recordings, segments, settings, coordinator = ctl[1], ctl[4], ctl[10], ctl[11]

    recordings.open(rec_a)
    _drain(qapp, lambda: segments.totalCount > 0 and coordinator._current_stream is not None)
    assert tasks, "the streaming path never ran"
    assert tasks[0].cancel is not None, "StreamInferenceTask was dispatched with cancel=None"
    tok_a = tasks[0].cancel

    # (2) the settings re-run call site also passes a token (the streamed record re-streams)
    settings.setWindowOverlap(0.25 if settings.windowOverlap != 0.25 else 0.0)
    _drain(qapp, lambda: len(tasks) >= 2, timeout_s=10.0)
    assert len(tasks) >= 2, "an overlap change did not re-stream the open recording"
    assert tasks[1].cancel is not None, "the re-run StreamInferenceTask was dispatched with cancel=None"
    assert tok_a.is_set()                        # the superseded open-time stream was cancelled
    tok_rerun = tasks[1].cancel

    # (3) opening another recording cancels the in-flight stream (same synchronous-emit argument as
    # the InferenceTask test: no processEvents, so only _invalidate can have run)
    recordings.open(rec_b)
    assert tok_rerun.is_set(), "opening a new recording did not cancel the in-flight streaming job"
    _drain(qapp, lambda: segments.totalCount > 0)


def test_tabular_export_names_the_analyzed_channel(qapp, tmp_path):
    """C: the flat tables (csv/parquet/tsv/mat) are what feed downstream analysis, and they did not
    say WHICH channel each grade belongs to. On ["RESP", "II"] inference grades II — the exported
    rows must name it, from the real open->inference->export pipeline."""
    pq = pytest.importorskip("pyarrow.parquet")
    rec = _write_ecg_2ch(tmp_path)
    engine = build_engine()
    ctl = engine._biosqa_controllers
    recordings, segments, exporter = ctl[1], ctl[4], ctl[8]

    recordings.open(rec, "ecg")
    _drain(qapp, lambda: segments.totalCount > 0)
    assert segments.totalCount > 0

    out = tmp_path / "ch.csv"
    exporter.exportToPath(str(out), "csv")
    rows = list(csv.DictReader(out.open()))
    assert rows, "nothing exported"
    assert "channel" in rows[0]
    assert {r["channel"] for r in rows} == {"II"}      # the graded channel, not RESP and not blank

    t = pq.read_table(_export(exporter, tmp_path / "ch.parquet", "parquet"))
    assert "channel" in t.column_names
    assert set(t.column("channel").to_pylist()) == {"II"}


def test_analyzed_channel_is_the_one_plotted_and_exported(qapp, tmp_path):
    """F5: inference runs on the modality-matching channel while the plot/export used channel 0 —
    so on ["RESP", "II"] the app graded II and drew/exported the bands against RESP. The analyzed
    channel is now the single source of truth, and the export names it."""
    rec = _write_ecg_2ch(tmp_path)
    engine = build_engine()
    ctl = engine._biosqa_controllers
    recordings, channels, signal_view, segments = ctl[1], ctl[2], ctl[3], ctl[4]
    selection, exporter, coordinator = ctl[5], ctl[8], ctl[11]

    recordings.open(rec, "ecg")            # force ECG (channel 0 is RESP, so auto-detect is moot)
    _drain(qapp, lambda: segments.totalCount > 0)

    assert coordinator._analyzed[1] == "II"                        # inference graded II, not RESP
    assert channels.data(channels.index(0, 0), channels.NameRole) == "II"
    assert channels.data(channels.index(0, 0), channels.AnalyzedRole) is True
    assert channels.data(channels.index(1, 0), channels.AnalyzedRole) is False   # RESP is not graded
    assert signal_view._channels[0] == "II"                        # the plot's primary lane is II
    assert abs(signal_view.valueAt(5.0)) < 5.0                     # ...and it holds ECG, not the +-20 RESP

    ctx = selection.analysis_context()
    assert ctx.channel == "II" and ctx.channel_index == 1

    out = tmp_path / "two.json"
    exporter.exportToPath(str(out), "json")
    doc = json.loads(out.read_text())
    assert doc["provenance"]["analyzed_channel"] == "II"           # the export says what it graded
    assert doc["provenance"]["recording"] == rec

    exporter.exportToPath(str(tmp_path / "ann.qual"), "wfdb")      # annotations name a signal INDEX
    ann = wfdb.rdann(str(tmp_path / "ann"), "qual")
    assert set(ann.chan) == {1}                                    # channel 1 (II), not 0 (RESP)


def test_segment_mini_plots_name_the_channel_they_serve(qapp, tmp_path):
    """The Segment Inspector's zoomed waveform and the Segment Grid minis both come from
    ``curveForRange``, which always serves the ANALYZED channel — on ["RESP", "II"] that is II.
    The main chart already de-colours its bands and suffixes its hover tooltip when the graded
    channel is not the one on screen (or is hidden), but these two views served II's amplitudes
    with no channel named anywhere, which is the same misleading-attribution defect one layer
    down. The envelope now reports its own channel so the view can say whose numbers those are."""
    rec = _write_ecg_2ch(tmp_path)
    engine = build_engine()
    ctl = engine._biosqa_controllers
    recordings, signal_view, segments, coordinator = ctl[1], ctl[3], ctl[4], ctl[11]

    recordings.open(rec, "ecg")
    _drain(qapp, lambda: segments.totalCount > 0)

    curve = signal_view.curveForRange(0.0, 10.0)
    assert curve is not None
    assert curve["channel"] == "II", "the envelope must name the graded channel, not lane 0 (RESP)"
    assert curve["channel"] == coordinator._analyzed[1]      # same channel inference actually graded
    assert curve["channel"] == signal_view._channels[0]      # ...which is the primary lane

    # the fallback disk-read branch (cache not yet populated) must label identically
    signal_view._caches = {}
    assert signal_view.curveForRange(0.0, 10.0)["channel"] == "II"

    # single-channel recording still reports its channel — QML decides when showing it adds
    # information (WaveformChart's `valueSuffix` convention: only when >1 lane is drawn, or the
    # graded lane is hidden), so the data layer must always carry it.
    recordings.open(_write_ecg(tmp_path, name="solo"), "ecg")
    _drain(qapp, lambda: segments.totalCount > 0)
    assert signal_view.curveForRange(0.0, 10.0)["channel"] == "II"


def test_short_record_reports_not_analysed(qapp, tmp_path):
    """M3: a record shorter than ONE model window produces zero windows. Reported as "0 segments" a
    user reads that as "no problems found"; it means NOTHING WAS ANALYSED. Say so."""
    rec = _write_ecg(tmp_path, dur=5, noisy=(1, 2), name="tiny")   # 5 s < the 10 s ECG window
    engine = build_engine()
    ctl = engine._biosqa_controllers
    recordings, segments, inference = ctl[1], ctl[4], ctl[6]

    recordings.open(rec)
    _drain(qapp, lambda: "NOT ANALYSED" in inference.statusText, timeout_s=10.0)
    assert "NOT ANALYSED" in inference.statusText
    assert segments.totalCount == 0


def test_pan_zoom_does_not_raise(qapp, tmp_path):
    """setView (the pan/zoom hot path) before a recording is open must no-op, not raise."""
    engine = build_engine()
    signal_view = engine._biosqa_controllers[3]
    signal_view.setView(5.0, 15.0)   # no recording bound yet → no-op
    assert signal_view.viewStartSec == 5.0 and signal_view.viewEndSec == 15.0


def _bounds(intervals):
    """What the user reads off the quality track: segment start/end, grade, and the model's original
    grade where refinement relaxed one (empty = never relaxed)."""
    return [(round(iv.start_sec, 6), round(iv.end_sec, 6), iv.tier, iv.model_tier)
            for iv in intervals]


def _relaxed(bounds) -> bool:
    """Did BOUNDARY REFINEMENT actually do something? (some bin was relaxed to a better grade the model
    itself gave a covering window, and the conservative grade it came from is preserved)."""
    return any(model_tier for _s, _e, _t, model_tier in bounds)


#: a 1 s burst — the artefact refinement exists to localize. The default fixture's 10 s noisy stretch
#: FILLS whole model windows, so the smeared poor run has no clean flank to erode and refinement is
#: (correctly) a no-op on it: a test built on that record cannot see this bug at all.
_BURST = (14, 15)


def test_file_size_alone_must_not_change_the_segment_boundaries(qapp, tmp_path, monkeypatch):
    """THE BUG, at the level the user meets it: TWO FILES holding the BYTE-IDENTICAL signal, opened
    through the real app; one is big enough to take the out-of-core path. They must be segmented the
    same.

    They were not. ``StreamInferenceTask`` ran ``run_length_encode`` and emitted — it never called
    ``refine_intervals`` — so the streamed file's poor segments kept COARSE window-resolution edges
    while the small file's were localized to the artefact. At the app's shipped 0.5 overlap that is a
    visibly different segmentation of the same recording, and nothing on screen said why.

    The guard and the recoverability pass are switched off here because they GENUINELY cannot run
    streamed (they need a whole-signal filtered view) — leaving them on would compare two different
    analyses and hide the one variable under test."""
    import biosqa.inference.streaming as st

    # same generator + same args -> the two files hold identical samples; only their names differ
    rec_mem = _write_ecg(tmp_path, noisy=_BURST, name="small")
    rec_str = _write_ecg(tmp_path, noisy=_BURST, name="large")
    engine = build_engine()
    ctl = engine._biosqa_controllers
    recordings, segments, settings = ctl[1], ctl[4], ctl[10]

    prev = (settings.windowOverlap, settings.guardEnabled, settings.recoveryEnabled,
            settings.refineBoundaries)
    settings.setWindowOverlap(0.5)          # what the app ships (and where refinement bites)
    settings.setGuardEnabled(False)
    settings.setRecoveryEnabled(False)
    settings.setRefineBoundaries(True)
    try:
        recordings.open(rec_mem)            # 15 000 samples: the in-memory path
        _drain(qapp, lambda: segments.totalCount > 0)
        in_memory = _bounds(segments._all_intervals)
        assert _relaxed(in_memory), \
            "refinement relaxed nothing in memory either — this record cannot detect the bug"

        monkeypatch.setattr(st, "LARGE_RECORD_SAMPLES", 100)   # the same signal, now 'too big'
        recordings.open(rec_str)
        _drain(qapp, lambda: segments.totalCount > 0)
        streamed = _bounds(segments._all_intervals)

        assert streamed == in_memory, "file size alone changed the segmentation"
    finally:
        settings.setWindowOverlap(prev[0])
        settings.setGuardEnabled(prev[1])
        settings.setRecoveryEnabled(prev[2])
        settings.setRefineBoundaries(prev[3])


def test_refine_toggle_re_segments_a_streamed_recording(qapp, tmp_path, monkeypatch):
    """"Refine boundaries" was wired to ``_schedule_rerun_normal`` -> ``_rerun_inference_normal``,
    which returns immediately when there is no in-memory signal. On a streamed record the switch
    therefore did NOTHING AT ALL: no re-run, no changed segments, no notice — and it had nothing to
    turn off anyway, because the streaming path never refined. Assert it now re-streams and that the
    segmentation actually changes."""
    import biosqa.inference.streaming as st
    import biosqa.viewmodels.coordinator as coord_mod

    monkeypatch.setattr(st, "LARGE_RECORD_SAMPLES", 100)
    tasks: list = []
    real_task = coord_mod.StreamInferenceTask
    monkeypatch.setattr(coord_mod, "StreamInferenceTask",
                        lambda *a, **k: tasks.append(real_task(*a, **k)) or tasks[-1])

    rec = _write_ecg(tmp_path, noisy=_BURST, name="refinestream")
    engine = build_engine()
    ctl = engine._biosqa_controllers
    recordings, segments, inference, settings = ctl[1], ctl[4], ctl[6], ctl[10]

    prev = (settings.windowOverlap, settings.refineBoundaries)
    settings.setWindowOverlap(0.5)
    settings.setRefineBoundaries(True)
    try:
        recordings.open(rec)
        _drain(qapp, lambda: segments.totalCount > 0 and len(tasks) >= 1)
        assert tasks[0].refine_enabled is True          # the setting reached the streaming worker
        refined = _bounds(segments._all_intervals)
        assert _relaxed(refined)                        # refinement really ran on the streamed record
        assert "boundary refinement is applied" in inference.statusText   # ...and the app says so

        settings.setRefineBoundaries(False)             # used to be a silent no-op here
        _drain(qapp, lambda: len(tasks) >= 2 and not _relaxed(_bounds(segments._all_intervals)),
               timeout_s=10.0)
        assert len(tasks) >= 2, "toggling 'Refine boundaries' did not re-segment the streamed record"
        assert tasks[-1].refine_enabled is False
        coarse = _bounds(segments._all_intervals)
        assert coarse != refined                        # the toggle visibly changed the boundaries
        assert not _relaxed(coarse)
    finally:
        settings.setWindowOverlap(prev[0])
        settings.setRefineBoundaries(prev[1])


def test_failed_open_leaves_no_phantom_lane(qapp, tmp_path):
    """A failed open leaves no stale DATA (asserted above) — but it still left a LANE.

    ``_set_channel_list`` lists the new recording's channels (they exist; nothing was graded) and
    ``SignalView.qml`` mirrors the visible ones into plot lanes on ``countChanged``. With no handle and
    no cache that is an empty lane placeholder for a recording that was never analysed. Wire the QML
    connection the app makes and assert the view comes out genuinely empty."""
    rec_a = _write_ecg(tmp_path, name="lanea")
    rec_b = _write_ecg(tmp_path, noisy=(40, 50), name="laneb")
    engine = build_engine()
    ctl = engine._biosqa_controllers
    recordings, channels, signal_view, segments, coordinator = ctl[1], ctl[2], ctl[3], ctl[4], ctl[11]

    # exactly what SignalView.qml does: channels.countChanged -> signalView.setLaneChannels(visibleNames())
    channels.countChanged.connect(lambda: signal_view.setLaneChannels(channels.visibleNames()))

    recordings.open(rec_a)
    _drain(qapp, lambda: segments.totalCount > 0)
    assert signal_view._lanes == ["II"] and signal_view._caches   # A is really drawn

    def _boom(modality):
        raise RuntimeError(f"{modality}.onnx is missing")

    coordinator._runner = _boom
    recordings.open(rec_b)
    qapp.processEvents()

    assert channels.rowCount() == 1                  # B's channels are listed (they exist)...
    assert signal_view._handle is None and signal_view._caches == {}
    assert signal_view._lanes == []                  # ...but NOTHING is drawn: no phantom lane
    assert signal_view.laneCount == 0 and signal_view.laneChannels == []
    assert segments.totalCount == 0 and signal_view.durationSec == 0.0
