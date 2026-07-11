"""Out-of-core streaming inference + block-wise plot cache. The headline guarantee: chunked
inference produces the SAME per-window grades as processing the whole signal."""
from pathlib import Path

import numpy as np
import pytest

MODELS = Path(__file__).resolve().parent.parent / "models"


def _temp_ecg_wfdb(tmp_path):
    wfdb = pytest.importorskip("wfdb")
    fs, secs = 250, 60
    n = fs * secs
    t = np.arange(n) / fs
    rng = np.random.default_rng(0)
    # quasi-ECG with a noisy middle stretch so grades vary along the record
    sig = 0.1 * np.sin(2 * np.pi * 1.1 * t)
    sig[fs * 20:fs * 35] += 0.6 * rng.standard_normal(fs * 15)
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


def test_stream_infer_matches_full_signal(tmp_path):
    from biosqa.inference.streaming import stream_infer
    from biosqa.io.loaders import read_window

    r = _ecg_runner()
    h = _temp_ecg_wfdb(tmp_path)          # fs_in == model fs (250) -> resample is identity -> exact
    sig = np.asarray(read_window(h, ["II"], 0, h.n_samples["II"]), dtype=np.float32).reshape(-1)
    codes = [g.split("_")[0] for g in r.card.primary_head.class_order]
    full = [codes[i] for i in r.run_sliding_window_multihead(sig).primary.argmax(axis=1)]

    tiers, confs, _arts, uncs, _ss, _ws, nwin = stream_infer(h, "II", r, overlap=0.0, block_sec=5.0)
    assert nwin == len(full)
    assert tiers == full                  # streamed == whole-signal, across ~12 block boundaries
    assert len(confs) == len(full)
    assert len(uncs) == len(full) and all(0.0 <= u <= 1.0001 for u in uncs)   # per-window uncertainty carried


def test_blockwise_plot_cache_matches_strided(tmp_path):
    from biosqa.inference.streaming import build_plot_cache_blockwise
    from biosqa.io.loaders import read_window

    h = _temp_ecg_wfdb(tmp_path)
    raw = np.asarray(read_window(h, ["II"], 0, h.n_samples["II"])).reshape(-1)
    n, fs, cap = raw.shape[0], float(h.fs_hz["II"]), 5000
    ft, fy, lo, hi = build_plot_cache_blockwise(h, "II", fs, cap_points=cap, block_samples=3000)
    stride = max(1, int(np.ceil(n / cap)))
    assert np.allclose(fy, raw[::stride])                 # identical points, built without a full read
    assert np.allclose(ft, np.arange(0, n, stride) / fs)
    assert lo <= float(raw.min()) + 1e-9 and hi >= float(raw.max()) - 1e-9


def test_estimate_analysis_samples():
    from biosqa.inference.streaming import estimate_analysis_samples

    class _H:
        n_samples = {"II": 1234}
    assert estimate_analysis_samples(_H(), "II") == 1234
    assert estimate_analysis_samples(_H(), "nope") == 0
