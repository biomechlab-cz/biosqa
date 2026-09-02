"""Occlusion-saliency XAI: faithfulness (importance localizes to a real artifact) + fail-safe on
degenerate windows. Runs the real ECG ONNX model (gradient-free, forward-only)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnxruntime")

from biosqa.inference.onnx_runner import OnnxRunner  # noqa: E402
from biosqa.inference.saliency import occlusion_saliency, signal_saliency  # noqa: E402

MODELS = Path(__file__).resolve().parents[1] / "models"


@pytest.fixture(scope="module")
def ecg_runner():
    r = OnnxRunner("ecg", str(MODELS))
    r.load()
    return r


def _ecg_with_burst(l_m, fs=250, b0=1200, b1=1500, seed=0):
    rng = np.random.default_rng(seed)
    x = 0.02 * rng.standard_normal(l_m)
    for i in range(int(0.4 * fs), l_m - 3, int(fs * 60 / 70)):
        x[i:i + 3] += 1.3
    x[b0:b1] += 2.5 * rng.standard_normal(b1 - b0)          # a LOCALIZED noise burst
    return x


def test_occlusion_saliency_localizes_the_artifact(ecg_runner):
    l_m = ecg_runner.card.l_m
    b0, b1 = 1200, 1500
    x = _ecg_with_burst(l_m, b0=b0, b1=b1)
    sal = occlusion_saliency(x, ecg_runner, target="unusable")
    assert sal.shape[0] == l_m and 0.0 <= sal.min() and sal.max() <= 1.0
    inside = sal[b0:b1].mean()
    outside = np.concatenate([sal[:b0], sal[b1:]]).mean()
    assert inside > 3.0 * outside                          # importance concentrates on the artifact (faithful)
    assert b0 <= int(np.argmax(sal)) <= b1                 # peak lands inside the burst


def test_occlusion_saliency_clean_window_stays_faint(ecg_runner):
    """Review regression: a confidently-graded CLEAN window the model is INSENSITIVE to must render FAINT
    (absolute scale), not a full-intensity heatmap of amplified noise (which per-window max-norm produced)."""
    l_m = ecg_runner.card.l_m
    rng = np.random.default_rng(5)
    fs = 250
    x = 0.02 * rng.standard_normal(l_m)                     # clean ECG, no artifact
    for i in range(int(0.4 * fs), l_m - 3, int(fs * 60 / 70)):
        x[i:i + 3] += 1.3
    sal = occlusion_saliency(x, ecg_runner, target="unusable")
    assert sal.max() < 0.6 and float((sal > 0.5).mean()) < 0.1   # faint, no spurious full-intensity hotspots


def test_occlusion_saliency_degenerate_is_safe(ecg_runner):
    l_m = ecg_runner.card.l_m
    assert not occlusion_saliency(np.zeros(l_m), ecg_runner).any()      # constant → zeros, no false heatmap
    assert occlusion_saliency(np.ones(4), ecg_runner).shape[0] == 4     # too short → zeros, no crash


def test_signal_saliency_spans_multiple_windows(ecg_runner):
    l_m = ecg_runner.card.l_m
    rng = np.random.default_rng(1)
    sig = np.concatenate([0.02 * rng.standard_normal(l_m), 0.02 * rng.standard_normal(l_m)])
    for i in range(int(0.4 * 250), len(sig) - 3, int(250 * 60 / 70)):
        sig[i:i + 3] += 1.3
    s = signal_saliency(sig, ecg_runner)
    assert s.shape[0] == len(sig) and 0.0 <= s.min() and s.max() <= 1.0
