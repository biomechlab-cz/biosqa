"""Regressions for inference-side conventions that were wrong, unpinned, or mis-documented.

Four independent defects, one file because they all live on the same window->grade path:
  * the flatline detector called essentially every real EDA recording a dead sensor;
  * the EEG panel's "Spec. entropy" row displayed an amplitude histogram, inverted;
  * the grade temperature is BAKED into the ONNX graph and must never be reapplied host-side;
  * the ECG dual-branch x_spec convention (z-scored, not raw) was documented nowhere and pinned
    by nothing, and the integrity guard did work whose result was provably discarded.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from biosqa.inference import conformal as conformal_mod
from biosqa.inference import data_quality as dq
from biosqa.inference.data_quality import record_quality
from biosqa.inference.sqi_breakdown import sqi_breakdown, sqi_consensus

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from biosqa.inference import integrity as integrity_mod  # noqa: E402
from biosqa.inference import prefilter as prefilter_mod  # noqa: E402
from biosqa.inference.onnx_runner import OnnxRunner  # noqa: E402
from biosqa.inference.spectral import spectral_band_channels  # noqa: E402


# ---- flatline: a live EDA trace is not a disconnected electrode ------------------------
def _tonic_eda(fs=8.0, secs=600.0, seed=0):
    """A resting EDA trace matched to REAL 8 Hz EDA, not to what makes the detector comfortable.

    The record's 1-99 range is set by the SCRs, so the long quiet stretches between them travel far
    less than 0.1% of it in any half second. The tonic wander is scaled so the trace's normalized
    local activity brackets the 10 real EDABE test records resampled to the EDA model rate: median
    act/range 7.3e-04 over a 0.5 s horizon (real: 3.2e-04 .. 1.05e-03) and 7.6e-03 over an 8 s one
    (real: 5.0e-03 .. 1.5e-02). Those records scored flatline_frac 0.42-0.82, 9 of 10 flagged
    'dead/disconnected sensor?' and 8 of 10 usable=False. The existing synthetic fixture in
    test_data_quality.py passes only because its tonic drift is an order of magnitude larger.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * secs)) / fs
    wander = 0.043                                           # microsiemens
    x = 4.0 + wander * np.sin(2 * np.pi * 0.02 * t) + 0.5 * wander * np.sin(2 * np.pi * 0.054 * t + 0.7)
    for t0 in (80.0, 200.0, 210.0, 430.0):                   # a few SCRs set the dynamic range
        d = np.clip(t - t0, 0.0, None)
        x += 2.5 * (np.exp(-d / 5.0) - np.exp(-d / 0.7)) * (t >= t0)
    return np.round(x + 0.001 * rng.standard_normal(t.size), 3), fs   # quantised, as an E4 gives


def test_resting_eda_at_the_model_rate_is_not_a_dead_sensor():
    x, fs = _tonic_eda()
    q = record_quality(x, fs)
    assert q.flatline_frac < 0.05 and q.usable
    assert not any("flatline" in f for f in q.flags)


def test_the_apps_own_test_eda_is_not_reported_as_a_dead_sensor(tmp_path):
    """The end-to-end repro a user hits by clicking 'load test data > EDA': through the real WFDB
    path the bundled 5-minute EDA scored flatline_frac 0.475 and flagged '48% flatline
    (dead/disconnected sensor?)' (ecg/ppg/eeg scored 0.000/0.051/0.000)."""
    from biosqa.io.loaders import open_recording, read_window
    from biosqa.io.synth import write_test_recording

    h = open_recording(write_test_recording("eda", tmp_path))
    n = max(h.n_samples.values())
    x = read_window(h, h.channel_names[:1], 0, n).reshape(-1)
    q = record_quality(x, float(next(iter(h.fs_hz.values()))))
    assert q.flatline_frac < 0.05 and not any("flatline" in f for f in q.flags)


def test_the_old_half_second_horizon_is_what_condemned_it(monkeypatch):
    """Pin the CAUSE, not just the symptom: with the sample floor removed the same trace fails, so a
    future 'simplification' back to a seconds-only horizon cannot pass silently."""
    x, fs = _tonic_eda()
    monkeypatch.setattr(dq, "_FLAT_MIN_WIN_N", 1)
    assert record_quality(x, fs).flatline_frac > 0.3


def test_dead_eda_channel_at_the_model_rate_is_still_caught():
    """The same 4-sample horizon made the rolling-activity estimate too noisy for the stationarity
    test in _is_dead_channel, so an 8 Hz channel that was dead END TO END scored 0.000."""
    rng = np.random.default_rng(1)
    dead = 4.0 + 1e-3 * rng.standard_normal(int(8.0 * 300))
    q = record_quality(dead, 8.0)
    assert q.flatline_frac > 0.9 and not q.usable
    assert any("flatline" in f for f in q.flags)


# ---- the EEG "Spec. entropy" row measures the spectrum, and abstains -------------------
def _eeg(fs=256.0, secs=5.0, seed=0):
    t = np.arange(int(fs * secs)) / fs
    return t, fs


def test_eeg_spectral_entropy_is_spectral_and_ordered_correctly():
    """It used to display _amp_entropy (a 32-bin AMPLITUDE histogram): a 10 Hz sine scored 0.94 and
    white noise 0.84, i.e. backwards for a row that says 'flat -> broadband noise'."""
    t, fs = _eeg()
    sine = np.sin(2 * np.pi * 10.0 * t)
    noise = np.random.default_rng(0).standard_normal(t.size)
    v = lambda x: {r["name"]: r["value"] for r in sqi_breakdown(x, fs, "eeg")}["Spec. entropy"]  # noqa: E731
    assert v(sine) < 0.1                                   # a pure rhythm concentrates its spectrum
    assert v(noise) > 0.8                                  # broadband noise is flat
    assert v(noise) > v(sine)


def test_eeg_spectral_entropy_does_not_vote_in_the_consensus():
    """On the reference corpus it does not separate the grade classes (Q0..Q3 = .71/.68/.63/.72), so
    it explains without dragging a clean window below the discordance banner's 0.5."""
    t, fs = _eeg()
    rows = sqi_breakdown(np.sin(2 * np.pi * 10.0 * t), fs, "eeg")
    ent = next(r for r in rows if r["name"] == "Spec. entropy")
    assert ent.get("informational") is True
    bars = [r["bar"] for r in rows if not r.get("informational")]
    assert sqi_consensus(rows) == pytest.approx(sum(bars) / len(bars))


# ---- the grade temperature is baked into the graph ------------------------------------
def test_temperature_scale_is_not_part_of_the_public_surface():
    """Every shipped graph divides the grade log-probabilities by the card's T, so re-applying it
    host-side scales twice (6.25x on ECG). It stays importable for offline use, but not exported."""
    assert "temperature_scale" not in conformal_mod.__all__
    assert "already" in conformal_mod.temperature_scale.__doc__ or \
           "DO NOT" in conformal_mod.temperature_scale.__doc__


def test_calibrate_grade_probs_only_sanitizes():
    from biosqa.inference.postprocess import calibrate_grade_probs

    class _Card:
        grade_temperature = 0.4

    p = np.array([[0.1, 0.2, 0.3, 0.4], [np.nan, 0.0, 0.0, 0.0]])
    out, non_finite = calibrate_grade_probs(p, _Card())
    np.testing.assert_allclose(out[0], p[0])              # untouched, NOT re-scaled by T=0.4
    np.testing.assert_allclose(out[1], 0.25)              # non-finite row -> uniform
    assert non_finite.tolist() == [False, True]


# ---- ECG dual-branch: x_spec convention + guard short-circuit --------------------------
SPEC_L = 64
_HEADS = [
    {"name": "grade", "output_name": "q_logits", "kind": "ordinal", "activation": "softmax",
     "class_order": ["Q0", "Q1", "Q2", "Q3"]},
    {"name": "usable", "output_name": "bin_logits", "kind": "binary", "activation": "softmax",
     "class_order": ["BAD", "OK"]},
]
_CARD = {
    "modality": "ecg", "L_m": SPEC_L, "fs_hz": 250, "class_order": ["Q0", "Q1", "Q2", "Q3"],
    "normalization": {"method": "none"}, "training_data_hash": "sha256:test",
    "model_version": "test", "heads": _HEADS,
    "spectral_preprocessing": {"bands_hz": [[15, 50], [50, 110]], "frame_s": 0.05, "hop_s": 0.02},
}


def _dual_runner(tmp_path):
    """A tiny 2-input graph (x_raw + x_spec) behind the real OnnxRunner load path."""
    rng = np.random.default_rng(4)
    n_bands = len(_CARD["spectral_preprocessing"]["bands_hz"])
    raw = helper.make_tensor_value_info("window", TensorProto.FLOAT, ["batch", 1, SPEC_L])
    spec = helper.make_tensor_value_info("spec", TensorProto.FLOAT, ["batch", n_bands, SPEC_L])
    nodes = [helper.make_node("Flatten", ["window"], ["flat"], axis=1),
             helper.make_node("Flatten", ["spec"], ["sflat"], axis=1)]
    inits, value_infos = [], []
    for name, dim in (("q_logits", 4), ("bin_logits", 2)):
        inits.append(numpy_helper.from_array(
            rng.standard_normal((SPEC_L, dim)).astype(np.float32), f"Wr_{name}"))
        inits.append(numpy_helper.from_array(
            rng.standard_normal((n_bands * SPEC_L, dim)).astype(np.float32), f"Ws_{name}"))
        nodes.append(helper.make_node("MatMul", ["flat", f"Wr_{name}"], [f"{name}_r"]))
        nodes.append(helper.make_node("MatMul", ["sflat", f"Ws_{name}"], [f"{name}_s"]))
        nodes.append(helper.make_node("Add", [f"{name}_r", f"{name}_s"], [name]))
        value_infos.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, ["batch", dim]))
    graph = helper.make_graph(nodes, "dualbranch", [raw, spec], value_infos, initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    onnx.save(model, str(tmp_path / "ecg.onnx"))
    (tmp_path / "ecg.model_card.json").write_text(json.dumps(_CARD))
    runner = OnnxRunner("ecg", tmp_path)
    runner.load()
    return runner


class _CapturingSession:
    def __init__(self, session, feeds):
        self._session, self._feeds = session, feeds

    def run(self, output_names, feed):
        self._feeds.append({k: np.asarray(v).copy() for k, v in feed.items()})
        return self._session.run(output_names, feed)


def test_x_spec_is_built_from_the_per_window_z_scored_signal(tmp_path):
    """The app deliberately differs from the export script, which used the raw store array: the card
    declares normalization 'none', so the app holds raw file units and log1p(band power) is not
    affine in gain. That trade is invisible in the grade head, so pin the actual array."""
    runner = _dual_runner(tmp_path)
    feeds: list[dict] = []
    runner._session = _CapturingSession(runner._session, feeds)
    windows = (np.random.default_rng(7).standard_normal((3, SPEC_L)) * 4.0 + 10.0).astype(np.float32)

    runner.predict_windows(windows)

    batch = windows.reshape(-1, 1, SPEC_L)
    p = _CARD["spectral_preprocessing"]
    zb = (batch - batch.mean(-1, keepdims=True)) / (batch.std(-1, keepdims=True) + 1e-6)
    expected = spectral_band_channels(zb, 250.0, [tuple(b) for b in p["bands_hz"]],
                                      frame_s=p["frame_s"], hop_s=p["hop_s"]).astype(np.float32)
    np.testing.assert_array_equal(feeds[0]["spec"], expected)
    np.testing.assert_array_equal(feeds[0]["window"], batch)          # x_raw stays RAW
    raw_conv = spectral_band_channels(batch, 250.0, [tuple(b) for b in p["bands_hz"]],
                                      frame_s=p["frame_s"], hop_s=p["hop_s"]).astype(np.float32)
    assert not np.allclose(expected, raw_conv)                        # the two conventions differ


def test_guard_record_skips_the_voter_bank_on_raw_input(tmp_path, monkeypatch):
    """integrity_guard ANDs its verdict with prefilter_verdict.prefiltered, so on raw-looking input
    the per-window mask is provably all-False -- and it cost 2.28 ms/window to prove it."""
    runner = _dual_runner(tmp_path)
    signal = np.random.default_rng(3).standard_normal(SPEC_L * 8).astype(np.float32)

    class _Verdict:
        def __init__(self, prefiltered):
            self.prefiltered, self.reasons, self.score = prefiltered, [], 0.0

    def _boom(*a, **k):
        raise AssertionError("integrity_guard must not run when the override is provably False")

    monkeypatch.setattr(prefilter_mod, "detect_prefiltering", lambda *a, **k: _Verdict(False))
    monkeypatch.setattr(integrity_mod, "integrity_guard", _boom)
    rep = runner.guard_record(signal)
    assert rep["prefiltered"] is False and rep["n_overridden"] == 0
    assert rep["override_mask"].dtype == bool and not rep["override_mask"].any()
    assert len(rep["override_mask"]) > 0                  # one entry per window, as before

    # ... and it IS run on pre-filtered input, where it is load-bearing.
    calls = []

    class _Guard:
        corrupt_override = True

    monkeypatch.setattr(prefilter_mod, "detect_prefiltering", lambda *a, **k: _Verdict(True))
    monkeypatch.setattr(integrity_mod, "integrity_guard",
                        lambda *a, **k: (calls.append(1), _Guard())[1])
    rep = runner.guard_record(signal)
    assert calls and rep["n_overridden"] == len(rep["override_mask"])
