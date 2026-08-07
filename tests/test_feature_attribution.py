"""Group-Shapley feature attribution (XAI 'why this grade'): faithfulness (a targeted corruption lands on
the matching quality group), Shapley soundness (efficiency: Σφ = f(x)−f(reference)), gating (ECG has no
fused SQI vector → None), and fail-safe behavior. Runs the real fusion ONNX models, forward-only.

MODEL AVAILABILITY. This release bundles ECG and EDA weights only (see LICENSE-MODELS), so the
modality-agnostic properties -- the efficiency axiom, |φ| ordering, share bounds, absolute bar scaling
and degenerate-input safety -- are exercised on EDA. The PPG cases are kept and skip themselves when no
`models/ppg.onnx` is present: they carry the one assertion that is genuinely modality-specific (a
broadband HF corruption must land on PPG's *Noise / HF* group), and they come back automatically for
anyone who drops their own PPG weights in. EDA's group set has no HF-noise member, so there is no honest
EDA analogue of that assertion -- it is skipped rather than restated as something it does not test.
"""
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


def _have(m: str) -> bool:
    return (MODELS / f"{m}.onnx").exists()


needs_ppg = pytest.mark.skipif(
    not _have("ppg"), reason="no ppg.onnx bundled in this release (see LICENSE-MODELS)")


@pytest.fixture(scope="module")
def ppg_runner():
    if not _have("ppg"):
        pytest.skip("no ppg.onnx bundled in this release")
    return _runner("ppg")


@pytest.fixture(scope="module")
def eda_runner():
    return _runner("eda")


# --------------------------------------------------------------------------- EDA signals
_EDA_FS, _EDA_L = 8.0, 480 * 2


def _eda_clean() -> np.ndarray:
    """A slow tonic drift: the shape a resting EDA trace actually has."""
    t = np.arange(_EDA_L) / _EDA_FS
    return (2.0 + 0.3 * np.sin(2 * np.pi * 0.02 * t)).astype(np.float32)


def _eda_motion(seed: int = 1) -> np.ndarray:
    """The same trace with sharp handling transients: what EDA corruption looks like."""
    rng = np.random.default_rng(seed)
    sig = _eda_clean().copy()
    for c in rng.choice(_EDA_L, 12, replace=False):
        sig[max(0, c - 2):c + 2] += rng.choice([-1, 1]) * 2.5
    return sig.astype(np.float32)


# --------------------------------------------------------------------------- modality-agnostic
def test_shapley_efficiency_holds(eda_runner):
    """Exact group-Shapley must satisfy the efficiency axiom: Σφ = f(window) − f(reference)."""
    res = grade_group_attribution(_eda_motion(3), eda_runner)
    assert res is not None
    total_phi = sum(g["phi"] for g in res["groups"])
    assert abs(total_phi - (res["base_unusable"] - res["reference_unusable"])) < 1e-3


def test_bar_length_is_absolute_not_relative(eda_runner):
    """Review regression (the saliency amplified-noise lesson, applied to bars): the UI bar length
    (``scaled``) must reflect the ABSOLUTE φ magnitude, so a confidently-CLEAN window whose groups barely
    move the grade renders FAINT -- not a full-width bar for its largest-of-tiny contribution."""
    rc = grade_group_attribution(_eda_clean(), eda_runner)
    rd = grade_group_attribution(_eda_motion(1), eda_runner)
    assert "scaled" in rc["groups"][0]
    clean_max = max(g["scaled"] for g in rc["groups"])
    dirty_max = max(g["scaled"] for g in rd["groups"])
    assert clean_max < 0.10                       # measured 0.041: faint, not full-width
    assert dirty_max > 2.5 * clean_max            # measured ratio 4.2-9.6 over seeds 0-4


def test_groups_sorted_and_bounded(eda_runner):
    res = grade_group_attribution(_eda_motion(2), eda_runner, top_k=3)
    assert res is not None and len(res["groups"]) <= 3
    mags = [abs(g["phi"]) for g in res["groups"]]
    assert mags == sorted(mags, reverse=True)                       # sorted by |φ| desc
    assert all(0.0 <= g["share"] <= 1.0 + 1e-6 for g in res["groups"])


def test_degenerate_segment_is_safe(eda_runner):
    assert grade_group_attribution(np.zeros(_EDA_L, dtype=np.float32), eda_runner) is not None  # no crash
    assert grade_group_attribution(np.ones(64, dtype=np.float32), eda_runner) is not None       # short, no crash


# --------------------------------------------------------------------------- modality-specific
@needs_ppg
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


def test_ecg_is_gated_out():
    """ECG fuses spectral CHANNELS (grade<-raw), not the hand-crafted SQI vector — attribution is N/A."""
    r = _runner("ecg")
    assert not has_feature_attribution(r)
    sig = np.sin(2 * np.pi * 1.2 * np.arange(2500) / 250.0).astype(np.float32)
    assert grade_group_attribution(sig, r) is None
