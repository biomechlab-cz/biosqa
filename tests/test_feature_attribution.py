"""Group-Shapley feature attribution (XAI 'why this grade'): faithfulness (a targeted corruption lands on
the matching quality group), Shapley soundness (efficiency: Σφ = f(x)−f(reference)), gating (ECG has no
fused SQI vector → None), and fail-safe behavior. Runs the real fusion ONNX models, forward-only."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnxruntime")

from biosqa.inference.onnx_runner import OnnxRunner  # noqa: E402
from biosqa.inference.feature_attribution import (  # noqa: E402
    grade_group_attribution,
    has_feature_attribution,
)

MODELS = Path(__file__).resolve().parents[1] / "models"


def _runner(m):
    r = OnnxRunner(m, str(MODELS))
    r.load()
    return r


@pytest.fixture(scope="module")
def ppg_runner():
    return _runner("ppg")


def test_ppg_hf_noise_attributes_to_noise_group(ppg_runner):
    """A broadband HF-corrupted PPG segment must attribute its (un)usability mainly to the Noise/HF group —
    the feature-level faithfulness analogue of the saliency burst-localization test."""
    fs, L = 64.0, 640 * 3
    t = np.arange(L) / fs
    rng = np.random.default_rng(1)
    sig = (np.sin(2 * np.pi * 1.2 * t) + 1.5 * rng.standard_normal(L)).astype(np.float32)
    res = grade_group_attribution(sig, ppg_runner)
    assert res is not None and res["groups"]
    top = res["groups"][0]
    assert top["group"] == "Noise / HF"          # the corrupted property dominates
    assert top["phi"] > 0                          # and it pushes toward UNUSABLE
    assert top["share"] > 0.4                       # by a clear margin


def test_bar_length_is_absolute_not_relative(ppg_runner):
    """Review regression (the saliency amplified-noise lesson, applied to bars): the UI bar length
    (``scaled``) must reflect the ABSOLUTE φ magnitude, so a confidently-CLEAN window whose groups barely
    move the grade renders FAINT — not a full-width bar for its largest-of-tiny contribution."""
    fs = 64.0
    clean = np.sin(2 * np.pi * 1.2 * np.arange(640 * 2) / fs).astype(np.float32)
    rng = np.random.default_rng(1)
    dirty = (np.sin(2 * np.pi * 1.2 * np.arange(640 * 2) / fs) + 1.5 * rng.standard_normal(640 * 2)).astype(np.float32)
    rc = grade_group_attribution(clean, ppg_runner)
    rd = grade_group_attribution(dirty, ppg_runner)
    assert "scaled" in rc["groups"][0]
    assert rc["groups"][0]["scaled"] < 0.4               # clean: faint bars (small absolute swing)
    assert rd["groups"][0]["scaled"] > 0.9               # corrupted: (near-)full bar for the dominant group


def test_shapley_efficiency_holds(ppg_runner):
    """Exact group-Shapley must satisfy the efficiency axiom: Σφ = f(worst window) − f(reference)."""
    fs, L = 64.0, 640 * 2
    t = np.arange(L) / fs
    rng = np.random.default_rng(3)
    sig = (np.sin(2 * np.pi * 1.1 * t) + 0.8 * rng.standard_normal(L)).astype(np.float32)
    res = grade_group_attribution(sig, ppg_runner)
    assert res is not None
    total_phi = sum(g["phi"] for g in res["groups"])
    assert abs(total_phi - (res["base_unusable"] - res["reference_unusable"])) < 1e-3


def test_groups_sorted_and_bounded(ppg_runner):
    fs, L = 64.0, 640 * 2
    sig = np.sin(2 * np.pi * 1.2 * np.arange(L) / fs).astype(np.float32)
    res = grade_group_attribution(sig, ppg_runner, top_k=3)
    assert res is not None and len(res["groups"]) <= 3
    mags = [abs(g["phi"]) for g in res["groups"]]
    assert mags == sorted(mags, reverse=True)                       # sorted by |φ| desc
    assert all(0.0 <= g["share"] <= 1.0 + 1e-6 for g in res["groups"])


def test_ecg_is_gated_out():
    """ECG fuses spectral CHANNELS (grade<-raw), not the hand-crafted SQI vector — attribution is N/A."""
    r = _runner("ecg")
    assert not has_feature_attribution(r)
    sig = np.sin(2 * np.pi * 1.2 * np.arange(2500) / 250.0).astype(np.float32)
    assert grade_group_attribution(sig, r) is None


def test_degenerate_segment_is_safe(ppg_runner):
    assert grade_group_attribution(np.zeros(640 * 2, dtype=np.float32), ppg_runner) is not None   # no crash
    assert grade_group_attribution(np.ones(64, dtype=np.float32), ppg_runner) is not None          # short, no crash
