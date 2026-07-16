"""Out-of-core streaming inference + block-wise plot cache. The headline guarantee: chunked inference
produces the SAME per-window NUMBERS (tiers AND calibrated confidence/uncertainty/grade probs) as
processing the whole signal — a record must not export different numbers just because it was big
enough to take the streaming path.

And the same goes for the SEGMENT BOUNDARIES, which is a step further than "same per-window numbers":
the boundaries come out of ``run_length_encode`` *and* ``refine.refine_intervals``, and refinement was
left off the streaming path entirely — so the same signal was segmented at window resolution above
``LARGE_RECORD_SAMPLES`` and at bin resolution below it. That whole-pipeline parity is asserted here by
driving BOTH production workers (``InferenceTask`` / ``StreamInferenceTask``) on the same record."""
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

MODELS = Path(__file__).resolve().parent.parent / "models"


def _temp_ecg_wfdb(tmp_path, secs: float = 60):
    wfdb = pytest.importorskip("wfdb")
    fs = 250
    n = int(round(fs * secs))
    t = np.arange(n) / fs
    rng = np.random.default_rng(0)
    # quasi-ECG with a noisy middle stretch so grades vary along the record
    sig = 0.1 * np.sin(2 * np.pi * 1.1 * t)
    sig[fs * 20:fs * 35] += 0.6 * rng.standard_normal(min(fs * 15, max(0, n - fs * 20)))
    wfdb.wrsamp("s", fs=fs, units=["mV"], sig_name=["II"],
                p_signal=sig.reshape(-1, 1).astype(np.float64), write_dir=str(tmp_path))
    from biosqa.io.loaders import open_recording
    return open_recording(str(tmp_path / "s.hea"))


def _ecg_runner():
    pytest.importorskip("onnxruntime")
    if not (MODELS / "ecg.onnx").exists():
        pytest.skip("ecg.onnx model not present")
    from biosqa.inference.onnx_runner import OnnxRunner
    r = OnnxRunner("ecg", MODELS)
    r.load()
    return r


class _Card:
    """Temperatures are provenance for constants already baked into the graph."""
    grade_temperature = 0.4
    heads = ()


# ---- shared sanitation (graph outputs are already calibrated exactly once) -------------------
def test_calibrate_grade_probs_does_not_reapply_card_temperature():
    from biosqa.inference.postprocess import (
        calibrate_grade_probs, confidences_from, normalized_entropy,
    )

    raw = np.array([[0.10, 0.20, 0.45, 0.25]])
    p, non_finite = calibrate_grade_probs(raw, _Card())
    assert not non_finite.any()
    assert np.allclose(p, raw)  # ONNX already applied the card temperature
    assert round(float(confidences_from(p, non_finite)[0]), 3) == 0.450
    assert round(float(normalized_entropy(p)[0]), 3) == 0.907


def test_calibrate_grade_probs_fails_safe_on_a_non_finite_row():
    from biosqa.inference.postprocess import calibrate_grade_probs, confidences_from

    raw = np.array([[np.nan, np.inf, np.nan, np.nan], [0.10, 0.20, 0.45, 0.25]])
    p, non_finite = calibrate_grade_probs(raw, _Card())
    assert non_finite.tolist() == [True, False]
    assert np.isfinite(p).all()
    assert int(p[0].argmax()) == 0                             # uniform row -> worst tier Q0
    assert float(confidences_from(p, non_finite)[0]) == 0.0    # ...at ZERO confidence, never NaN


# ---- streamed vs in-memory parity ------------------------------------------------------------
def test_stream_infer_matches_calibrated_in_memory_at_shipped_overlap(tmp_path):
    """The app SHIPS overlap 0.5, so parity must hold with windows straddling block boundaries — and
    on the exported NUMBERS, not merely on the tiers (temperature scaling is monotone, so tier
    equality would pass even with the calibration missing entirely)."""
    from biosqa.inference.postprocess import (
        calibrate_grade_probs, confidences_from, normalized_entropy,
    )
    from biosqa.inference.streaming import stream_infer
    from biosqa.io.loaders import read_window

    r = _ecg_runner()
    h = _temp_ecg_wfdb(tmp_path)          # fs_in == model fs (250) -> resample is identity -> exact
    sig = np.asarray(read_window(h, ["II"], 0, h.n_samples["II"]), dtype=np.float32).reshape(-1)
    codes = [g.split("_")[0] for g in r.card.primary_head.class_order]

    # exactly what the in-memory InferenceTask exports
    q, non_finite = calibrate_grade_probs(
        r.run_sliding_window_multihead(sig, overlap=0.5).primary, r.card)
    full_tiers = [codes[i] for i in q.argmax(axis=1)]
    full_conf = confidences_from(q, non_finite)
    full_unc = normalized_entropy(q)

    # block_sec=5 on a 60 s record -> ~12 block boundaries the 10 s windows straddle
    tiers, confs, _arts, uncs, gprobs, starts, _ss, _ws, nwin, _sig = stream_infer(
        h, "II", r, overlap=0.5, block_sec=5.0)
    assert nwin == len(full_tiers)
    assert tiers == full_tiers
    assert np.allclose(confs, full_conf, atol=1e-9)      # CALIBRATED confidence, not the raw max
    assert np.allclose(uncs, full_unc, atol=1e-9)
    assert np.allclose(gprobs, q, atol=1e-9)             # carried through -> streamed APS sets too
    # ... and on the window TIMES: the same grid the in-memory path scored
    assert np.allclose(starts, r.window_starts_sec(sig, overlap=0.5))


def test_stream_infer_matches_full_signal(tmp_path):
    from biosqa.inference.streaming import stream_infer
    from biosqa.io.loaders import read_window

    r = _ecg_runner()
    h = _temp_ecg_wfdb(tmp_path)
    sig = np.asarray(read_window(h, ["II"], 0, h.n_samples["II"]), dtype=np.float32).reshape(-1)
    codes = [g.split("_")[0] for g in r.card.primary_head.class_order]
    full = [codes[i] for i in r.run_sliding_window_multihead(sig).primary.argmax(axis=1)]

    tiers, confs, _arts, uncs, _gp, _st, _ss, _ws, nwin, _sig = stream_infer(h, "II", r, overlap=0.0,
                                                                             block_sec=5.0)
    assert nwin == len(full)
    assert tiers == full                  # streamed == whole-signal, across ~12 block boundaries
    assert len(confs) == len(full)
    assert len(uncs) == len(full) and all(0.0 <= u <= 1.0001 for u in uncs)   # per-window uncertainty carried


@pytest.mark.parametrize("overlap", [0.0, 0.5])
def test_stream_infer_grades_the_tail_of_a_partial_window_record(tmp_path, overlap):
    """A 61.2 s record is 6.12 windows. The streamed grid must END-ANCHOR its final window at
    [51.2, 61.2] exactly as `make_windows` does in memory -- otherwise the last 1.2 s of the recording
    is never graded (an ungraded tail reads to a user as 'no problems found there'), and it must report
    the REAL start times, because bounding the intervals on `i * stride` puts the tail window's grade
    past the end of the record."""
    from biosqa.inference.preprocess import window_starts
    from biosqa.inference.segmenter import run_length_encode
    from biosqa.inference.streaming import stream_infer
    from biosqa.io.loaders import read_window

    r = _ecg_runner()
    h = _temp_ecg_wfdb(tmp_path, secs=61.2)               # fs_in == model fs -> resample is identity
    n = int(h.n_samples["II"])
    sig = np.asarray(read_window(h, ["II"], 0, n), dtype=np.float32).reshape(-1)

    tiers, confs, _a, uncs, _gp, starts, stride_sec, window_sec, nwin, _sig = stream_infer(
        h, "II", r, overlap=overlap, block_sec=5.0)

    expected = window_starts(n, r.card, overlap).astype(np.float64) / float(r.card.fs_hz)
    assert np.allclose(starts, expected)                  # ... including the end-anchored tail
    assert starts[-1] == pytest.approx(51.2)
    assert nwin == len(expected) == len(tiers)
    # the streamed tail window is the SAME window the in-memory path scores
    full = [g.split("_")[0] for g in r.card.primary_head.class_order]
    mem = r.run_sliding_window_multihead(sig, overlap=overlap).primary.argmax(axis=1)
    assert tiers == [full[i] for i in mem]

    ivs = run_length_encode(tiers, confs, stride_sec, window_sec, uncertainty_per_window=uncs,
                            window_starts_sec=starts)
    assert max(iv.end_sec for iv in ivs) == pytest.approx(61.2)      # never past the record end
    assert sum(iv.duration_sec for iv in ivs) == pytest.approx(61.2)  # ... and the tail IS graded


def test_stream_infer_stops_on_cancel(tmp_path):
    """A superseded stream stops reading blocks instead of running the whole record to completion."""
    import threading

    from biosqa.inference.streaming import stream_infer

    r = _ecg_runner()
    h = _temp_ecg_wfdb(tmp_path)
    cancel = threading.Event()
    cancel.set()
    tiers, _c, _a, _u, _gp, starts, _ss, _ws, nwin, sig = stream_infer(
        h, "II", r, overlap=0.0, block_sec=5.0, cancel=cancel, collect_signal=True)
    assert nwin == 0 and tiers == [] and len(starts) == 0  # cancelled before the first block: nothing ran
    assert sig is None            # ... and a PARTIAL signal is never handed out as "the signal"


# ---- block-wise plot cache -------------------------------------------------------------------
def test_blockwise_plot_cache_matches_in_memory_envelope(tmp_path):
    from biosqa.inference.streaming import build_plot_cache_blockwise
    from biosqa.io.loaders import read_window
    from biosqa.workers.qt_threads import build_plot_cache

    h = _temp_ecg_wfdb(tmp_path)
    raw = np.asarray(read_window(h, ["II"], 0, h.n_samples["II"])).reshape(-1)
    fs, cap = float(h.fs_hz["II"]), 5000
    ft, fy, lo, hi = build_plot_cache_blockwise(h, "II", fs, cap_points=cap, block_samples=3000)
    et, ey, _lo, _hi = build_plot_cache(raw, fs, cap=cap)
    assert np.allclose(fy, ey) and np.allclose(ft, et)     # identical points, built without a full read
    assert np.all(np.diff(ft) >= 0)                        # min/max pairs stay in time order
    assert lo <= float(raw.min()) + 1e-9 and hi >= float(raw.max()) - 1e-9


def test_blockwise_plot_cache_keeps_a_single_sample_spike(tmp_path):
    """The whole point of the envelope: an artifact spike must survive decimation. Naive striding
    (raw[::stride]) drops it, and no display path ever re-reads raw — so it was gone at every zoom."""
    from biosqa.inference.streaming import build_plot_cache_blockwise
    from biosqa.io.loaders import read_window

    wfdb = pytest.importorskip("wfdb")
    fs, n = 250, 15000
    sig = 0.05 * np.sin(2 * np.pi * 1.1 * np.arange(n) / fs)
    spike_i = 7013                                    # deliberately off any plausible stride grid
    sig[spike_i] = 5.0
    wfdb.wrsamp("k", fs=fs, units=["mV"], sig_name=["II"],
                p_signal=sig.reshape(-1, 1), write_dir=str(tmp_path))
    from biosqa.io.loaders import open_recording
    h = open_recording(str(tmp_path / "k.hea"))
    raw = np.asarray(read_window(h, ["II"], 0, n)).reshape(-1)

    cap = 1000
    stride = int(np.ceil(n / cap))
    assert float(raw[::stride].max()) < 0.5           # the old strided cache would have MISSED the spike
    _ft, fy, _lo, _hi = build_plot_cache_blockwise(h, "II", float(fs), cap_points=cap,
                                                   block_samples=4000)
    assert float(fy.max()) == pytest.approx(float(raw.max()))    # the envelope keeps it
    assert fy.size <= cap + 2                                    # at the same point budget


def test_estimate_analysis_samples():
    from biosqa.inference.streaming import estimate_analysis_samples

    class _H:
        n_samples = {"II": 1234}
    assert estimate_analysis_samples(_H(), "II") == 1234
    assert estimate_analysis_samples(_H(), "nope") == 0


# ---- whole-pipeline parity: the SAME signal must get the SAME segment BOUNDARIES either way -------
def _qapp():
    pytest.importorskip("PySide6.QtWidgets")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _record(tmp_path, modality: str, secs: float, bursts, name: str, fs_native: float | None = None):
    """A synthetic record carrying short high-amplitude bursts — the localizable artefact refinement
    exists to tighten a poor segment down to.

    ``fs_native`` defaults to THE MODEL'S OWN RATE (the resample is then the identity). Pass a different
    rate to exercise the resampling path, which is where a chunked reader can drift."""
    wfdb = pytest.importorskip("wfdb")
    from biosqa.io.loaders import open_recording

    fs = float(fs_native or _runner(modality).card.fs_hz)
    n = int(round(secs * fs))
    t = np.arange(n) / fs
    rng = np.random.default_rng(7)
    x = 0.6 * np.sin(2 * np.pi * 1.2 * t) + 0.02 * rng.standard_normal(n)
    for a, b in bursts:
        s0, s1 = int(a * fs), min(n, int(b * fs))
        if s1 > s0:
            x[s0:s1] += 4.0 * rng.standard_normal(s1 - s0)
    wfdb.wrsamp(name, fs=int(round(fs)), units=["mV"], sig_name=["II"],
                p_signal=x.reshape(-1, 1).astype(float), write_dir=str(tmp_path))
    return open_recording(str(tmp_path / f"{name}.hea"))


_RUNNERS: dict = {}


def _runner(modality: str):
    pytest.importorskip("onnxruntime")
    if not (MODELS / f"{modality}.onnx").exists():
        pytest.skip(f"{modality}.onnx model not present")
    if modality not in _RUNNERS:
        from biosqa.inference.onnx_runner import OnnxRunner
        r = OnnxRunner(modality, MODELS)
        r.load()
        _RUNNERS[modality] = r
    return _RUNNERS[modality]


def _bounds(intervals):
    """What the user actually sees on the quality track: where each segment starts/ends, its grade, and
    (when refinement relaxed a bin) the model's conservative grade it was relaxed FROM."""
    return [(round(iv.start_sec, 6), round(iv.end_sec, 6), iv.tier, iv.model_tier)
            for iv in intervals]


def _in_memory_intervals(modality, handle, overlap):
    """The in-memory production worker, with the two whole-signal-only features (the false-clean guard
    and the recoverability pass) OFF — they are legitimately absent from the streaming path, so leaving
    them on would compare two different analyses. Refinement is the ONE variable under test.

    The full-signal resample in front of it is what ``LoadResampleTask`` does on open (a no-op when the
    record is already at the model rate); leaving it out would compare against a signal the app never
    actually scores."""
    from biosqa.io.loaders import read_window
    from biosqa.workers.qt_threads import InferenceTask, resample_signal
    from biosqa.workers.signals import InferenceWorkerSignals

    app = _qapp()
    r = _runner(modality)
    raw = np.asarray(read_window(handle, ["II"], 0, handle.n_samples["II"]),
                     dtype=np.float32).reshape(-1)
    sig = resample_signal(raw, float(handle.fs_hz["II"]), float(r.card.fs_hz))
    win = r.card.l_m / float(r.card.fs_hz)
    carrier, got = InferenceWorkerSignals(), []
    carrier.intervalsReady.connect(lambda _m, ivs: got.append(ivs))
    InferenceTask(r, sig, win * (1.0 - overlap), win, carrier, overlap=overlap,
                  guard_enabled=False, recovery_enabled=False, refine_enabled=True).run()
    app.processEvents()
    assert got, "InferenceTask emitted no intervals"
    return got[-1]


def _streamed_intervals(modality, handle, overlap, *, refine=True, block_sec=5.0):
    """The out-of-core production worker. ``block_sec=5`` forces MANY block boundaries for the model
    windows to straddle, which is where a chunked pipeline drifts if it is going to."""
    from biosqa.workers.qt_threads import StreamInferenceTask
    from biosqa.workers.signals import StreamWorkerSignals

    app = _qapp()
    carrier, got, notices, failures = StreamWorkerSignals(), [], [], []
    carrier.intervalsReady.connect(lambda _m, ivs: got.append(ivs))
    carrier.notice.connect(notices.append)
    carrier.failed.connect(lambda _m, e: failures.append(e))
    StreamInferenceTask(handle, "II", "II", ["II"], modality, _runner(modality), overlap, carrier,
                        rebuild_plot=False, refine_enabled=refine, block_sec=block_sec).run()
    app.processEvents()
    assert not failures, f"StreamInferenceTask failed: {failures}"
    assert got, "StreamInferenceTask emitted no intervals"
    return got[-1], (notices[-1] if notices else "")


@pytest.mark.parametrize(
    "modality,secs,bursts",
    [("ecg", 60.0, [(14, 15), (44.5, 45.5)]),      # whole number of windows
     ("ecg", 61.2, [(14, 15), (44.5, 45.5)]),      # + an END-ANCHORED tail window (off the stride grid)
     ("ppg", 137.5, [(14, 15), (98.0, 99.0)])],    # another modality + another window/rate pair
)
@pytest.mark.parametrize("overlap", [0.0, 0.5])    # 0.5 is what the app SHIPS
def test_streaming_and_in_memory_paths_produce_identical_segment_boundaries(
        tmp_path, modality, secs, bursts, overlap):
    """THE INVARIANT: one signal, one segmentation — whichever path graded it.

    ``StreamInferenceTask`` ran ``run_length_encode`` and emitted, and never called
    ``refine_intervals``. So a record that crossed ``LARGE_RECORD_SAMPLES`` got COARSE,
    window-resolution poor-segment boundaries while the byte-identical signal below the threshold got
    boundaries localized to the actual artefact. File size silently changed the answer.

    This drives both production workers on the same record and compares what a user sees: the segment
    start/end times, their grades, and the relaxed-from provenance. It cannot pass on the old code at
    overlap > 0 — the paths returned different numbers of intervals there (the in-memory one splits the
    smeared poor run around the burst core; the streamed one did not)."""
    h = _record(tmp_path, modality, secs, bursts, f"p{modality}{int(secs * 10)}{int(overlap * 100)}")
    mem = _bounds(_in_memory_intervals(modality, h, overlap))
    streamed, notice = _streamed_intervals(modality, h, overlap)
    assert _bounds(streamed) == mem
    assert notice.startswith("Large recording")


@pytest.mark.parametrize("modality,secs,bursts",
                         [("ecg", 61.2, [(14, 15), (44.5, 45.5)]),
                          ("ppg", 137.5, [(14, 15), (98.0, 99.0)])])
def test_refinement_is_load_bearing_on_the_streaming_path(tmp_path, modality, secs, bursts):
    """Guards the test above from being vacuously true.

    Refinement is a deliberate NO-OP at overlap 0 (each time is covered by exactly one window, so the
    model made exactly one statement about it — see :mod:`inference.refine`). At the SHIPPED overlap of
    0.5 it is not: it must actually MOVE the boundaries. Assert the refined streamed result differs
    from the unrefined one, and that the difference is precisely what the old streaming path shipped —
    otherwise "streamed == in-memory" would prove nothing about refinement at all."""
    h = _record(tmp_path, modality, secs, bursts, f"lb{modality}")
    refined, _ = _streamed_intervals(modality, h, 0.5, refine=True)
    unrefined, notice = _streamed_intervals(modality, h, 0.5, refine=False)

    assert _bounds(refined) != _bounds(unrefined), "refinement changed nothing — the parity test is vacuous"
    assert len(refined) > len(unrefined)                       # the smeared poor run got split at the burst
    assert any(iv.model_tier for iv in refined)                # ...and the relaxation is recorded, auditable
    assert not any(iv.model_tier for iv in unrefined)
    # the unrefined stream is EXACTLY what the pre-fix code produced -> that was the drift
    assert _bounds(unrefined) != _bounds(_in_memory_intervals(modality, h, 0.5))
    # refinement OFF is the user's own setting, so the notice must NOT blame the record's size
    assert "no effect here" not in notice

    # NEVER better than the model said: a refined segment may only be relaxed to a grade the model
    # itself gave a window covering it, and the grade it was relaxed FROM is preserved.
    rank = {"Q0": 0, "Q1": 1, "Q2": 2, "Q3": 3}
    for iv in refined:
        if iv.model_tier:
            assert rank[iv.tier] > rank[iv.model_tier]


def test_streaming_refuses_to_refine_over_the_memory_budget_and_says_so(tmp_path, monkeypatch):
    """Refinement is NOT chunkable (``fine_badness`` scores every bin against whole-signal robust
    statistics), so past ``REFINE_MAX_MODEL_SAMPLES`` the streaming pass keeps its memory promise and
    does not run it. That is a real behaviour difference, so the app must SAY it, name the setting that
    stopped working, and never pretend the coarse boundaries are refined ones."""
    import biosqa.inference.streaming as st

    monkeypatch.setattr(st, "REFINE_MAX_MODEL_SAMPLES", 1000)   # any real record is now "too large"
    h = _record(tmp_path, "ecg", 61.2, [(14, 15), (44.5, 45.5)], "budget")

    tiers, _c, _a, _u, _gp, _s, _ss, _ws, nwin, sig = st.stream_infer(
        h, "II", _runner("ecg"), overlap=0.5, block_sec=5.0, collect_signal=True)
    assert nwin > 0 and sig is None            # the pass ran; the signal was NOT retained

    over, notice = _streamed_intervals("ecg", h, 0.5, refine=True)
    assert not any(iv.model_tier for iv in over)               # nothing was relaxed -> nothing refined
    assert "boundary refinement" in notice and "no effect here" in notice
    assert "window resolution" in notice
    # ... and it is exactly the unrefined segmentation, not a half-refined one
    unrefined, _ = _streamed_intervals("ecg", h, 0.5, refine=False)
    assert _bounds(over) == _bounds(unrefined)

    # under the budget the very same record IS refined, and the notice says that instead
    monkeypatch.setattr(st, "REFINE_MAX_MODEL_SAMPLES", 24_000_000)
    under, notice2 = _streamed_intervals("ecg", h, 0.5, refine=True)
    assert _bounds(under) != _bounds(unrefined)
    assert "boundary refinement is applied" in notice2 and "no effect here" not in notice2


@pytest.mark.parametrize("modality,fs_native", [("ecg", 360), ("ecg", 500), ("ecg", 1000),
                                                ("ppg", 250), ("eeg", 500), ("eda", 32)])
def test_streamed_resample_is_the_whole_signal_resample(tmp_path, modality, fs_native):
    """The streamed reader must reconstruct EXACTLY the model-rate signal the in-memory path scores —
    for a record whose native rate is not the model rate, which is the common case and the ONE case the
    streaming tests never covered (every other fixture here is written at the model's own rate, where
    the resample is the identity and this class of bug is invisible).

    ``resample_poly`` is polyphase: its filter phase at a block start depends on that start index MOD
    ``down``. Blocks cut on a wall-clock grid land on arbitrary phases, and each block's output length
    rounds to +-1 sample — both errors ACCUMULATE. On a 137.5 s 500 Hz ECG the streamed signal came out
    3 samples long with a 0.11 max error on a 1.3 peak-to-peak signal: a different signal, so different
    grades and different segment boundaries, for the same recording."""
    from biosqa.inference.streaming import stream_infer
    from biosqa.io.loaders import read_window
    from biosqa.workers.qt_threads import resample_signal

    r = _runner(modality)
    h = _record(tmp_path, modality, 137.5, [(14, 15)], f"rs{modality}{fs_native}", fs_native=fs_native)
    raw = np.asarray(read_window(h, ["II"], 0, h.n_samples["II"]), dtype=np.float32).reshape(-1)
    in_memory = resample_signal(raw, float(fs_native), float(r.card.fs_hz))   # what LoadResampleTask makes

    # block_sec=7 on a 137.5 s record -> ~20 block boundaries for the error to accumulate over
    *_rest, streamed = stream_infer(h, "II", r, overlap=0.5, block_sec=7.0, collect_signal=True)
    assert streamed is not None
    assert streamed.shape == in_memory.shape, "the streamed signal drifted in LENGTH"
    assert np.allclose(streamed, in_memory, atol=1e-6), "the streamed signal drifted in VALUE"


@pytest.mark.parametrize("modality,fs_native", [("ecg", 500), ("ppg", 250)])
def test_resampled_record_is_segmented_the_same_by_both_paths(tmp_path, modality, fs_native):
    """...and the consequence, at the level the user meets it: the same recording, whose native rate is
    not the model rate, must be segmented identically whichever path grades it. This is the parity test
    above minus its most convenient assumption."""
    h = _record(tmp_path, modality, 137.5, [(14, 15), (98.0, 99.0)],
                f"rp{modality}{fs_native}", fs_native=fs_native)
    streamed, _notice = _streamed_intervals(modality, h, 0.5, block_sec=7.0)
    assert _bounds(streamed) == _bounds(_in_memory_intervals(modality, h, 0.5))


def test_collected_signal_is_the_signal_the_windows_were_scored_on(tmp_path):
    """The signal handed to refinement must BE the model-rate signal the model graded — a
    re-read/re-resampled copy could drift from it, and refinement would then localize an artefact
    against samples the grades do not describe."""
    from biosqa.inference.streaming import stream_infer
    from biosqa.io.loaders import read_window

    r = _ecg_runner()
    h = _temp_ecg_wfdb(tmp_path, secs=61.2)      # fs_in == model fs -> the resample is the identity
    raw = np.asarray(read_window(h, ["II"], 0, h.n_samples["II"]), dtype=np.float32).reshape(-1)

    *_rest, sig = stream_infer(h, "II", r, overlap=0.5, block_sec=5.0, collect_signal=True)
    assert sig is not None
    assert sig.shape == raw.shape and np.allclose(sig, raw)    # exactly the scored samples, no drift
    # and it is NOT collected when the caller does not ask (the default: no wasted RAM)
    *_rest2, none_sig = stream_infer(h, "II", r, overlap=0.5, block_sec=5.0)
    assert none_sig is None
