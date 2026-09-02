"""Unified grade-explanation narrative: composes where/why/what from the three XAI signals, degrades
gracefully (ECG has no attribution; a clean/faint segment names no region or driver), never fabricates."""
from __future__ import annotations

import numpy as np

from biosqa.inference.narrative import build_grade_narrative

_PEAK = list(np.concatenate([np.zeros(100), np.ones(10), np.zeros(146)]))   # localized peak at ~40%
_FAINT = list(0.1 * np.ones(256))
_ATTR_DIRTY = {"groups": [{"group": "Noise / HF", "phi": 0.63, "share": 0.81},
                          {"group": "Spectral", "phi": 0.10, "share": 0.13}]}
_ATTR_CLEAN = {"groups": [{"group": "Complexity / dynamics", "phi": 0.02, "share": 0.40}]}


def test_dirty_fusion_mentions_where_and_why():
    s = build_grade_narrative(tier_code="Q1", tier_label="Poor", seg_dur_s=8.0,
                              saliency_map=_PEAK, attribution=_ATTR_DIRTY, artifacts=[])
    assert s.startswith("Graded Poor (Q1)")
    assert "region ≈3." in s                       # peak ~40% of 8 s ≈ 3.1 s
    assert "Noise / HF" in s and "driven mainly" in s


def test_ecg_omits_why_but_keeps_where_and_artifact():
    s = build_grade_narrative(tier_code="Q0", tier_label="Unacceptable", seg_dur_s=6.0,
                              saliency_map=_PEAK, attribution=None, artifacts=["baseline wander", "muscle"])
    assert "driven mainly" not in s and "quality features" not in s     # no attribution -> no why clause
    assert "model focuses on a region" in s.lower()                      # capitalized as the leading clause
    assert "tagged as baseline wander and muscle" in s


def test_clean_says_nothing_stands_out():
    s = build_grade_narrative(tier_code="Q3", tier_label="Excellent", seg_dur_s=10.0,
                              saliency_map=_FAINT, attribution=_ATTR_CLEAN, artifacts=[])
    assert "no single quality property stands out".capitalize() in s or "no single quality property stands out" in s.lower()
    assert "region ≈" not in s                     # a faint map must not invent a location


def test_grade_only_when_no_signals():
    s = build_grade_narrative(tier_code="Q2", tier_label="Acceptable", seg_dur_s=5.0,
                              saliency_map=[], attribution=None, artifacts=[])
    assert s == "Graded Acceptable (Q2) over this 5.0 s window."


def test_diffuse_map_says_spread_not_a_point():
    diffuse = list(np.linspace(0.4, 1.0, 256))     # broadly important, no single hotspot
    s = build_grade_narrative(tier_code="Q1", tier_label="Poor", seg_dur_s=8.0,
                              saliency_map=diffuse, attribution=None, artifacts=[])
    assert "responds across much of the window" in s and "region ≈" not in s


def test_held_up_phrasing_for_negative_phi():
    """A group that keeps the grade USABLE (φ<0) reads as 'held up by', not 'driven by'."""
    attr = {"groups": [{"group": "Pulse morphology", "phi": -0.3, "share": 0.7}]}
    s = build_grade_narrative(tier_code="Q3", tier_label="Excellent", seg_dur_s=6.0,
                              saliency_map=_FAINT, attribution=attr, artifacts=[])
    assert "held up mainly by the Pulse morphology" in s
