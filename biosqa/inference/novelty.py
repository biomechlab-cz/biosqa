"""Feature-space NOVELTY detection (explainable cross-dataset OOD; Lee et al. 2018 Mahalanobis OOD,
research3 cross-dataset weakness).

The grade says whether a window is a good biosignal; novelty says whether it looks like ANYTHING the model
was trained on. A recording from a new device/cohort can be graded confidently-clean yet sit far from the
training feature distribution — the silent cross-dataset failure mode (Rahman 2025). We measure it as the
Mahalanobis distance of the window's INTERPRETABLE SQI vector (the app's own :func:`sqi_breakdown` values)
from a reference (μ, σ, shrinkage inverse-correlation, D² threshold) computed over the training store and
shipped in the model card's ``novelty`` block. Because the space is interpretable, we also report the single
SQI contributing most to the distance — an *explainable* OOD flag ("novel: bSQI unlike training"), not an
opaque score. Reference is over ALL training windows, so novelty is ORTHOGONAL to the grade (a normal corrupt
window resembles training-corrupt windows and is not novel). Pure numpy.
"""
from __future__ import annotations

import numpy as np

__all__ = ["sqi_feature_vector", "novelty_distance"]


def sqi_feature_vector(window, fs: float, modality: str) -> tuple[np.ndarray, list[str]]:
    """The app's :func:`sqi_breakdown` (values, names) as a fixed-order vector — the exact vector the shipped
    novelty reference was fit on (so reference and runtime are reproducible). Values are sanitized (NaN/±inf
    → 0) to match the generator. The NAMES are returned so the caller can guard against a feature REORDER."""
    from biosqa.inference.sqi_breakdown import sqi_breakdown
    rows = sqi_breakdown(window, fs, modality)
    vals = np.nan_to_num(np.array([float(r["value"]) for r in rows], dtype=np.float64),
                         nan=0.0, posinf=0.0, neginf=0.0)
    return vals, [str(r["name"]) for r in rows]


def novelty_distance(features, block, feature_names=None) -> tuple[float, str]:
    """Mahalanobis D² of ``features`` vs the card's ``novelty`` block, plus the single feature that
    contributes most to it (the explanation). Fails SAFE — returns ``(0.0, "")`` — when the block is
    missing/partial/ragged, the vector length mismatches, OR (when ``feature_names`` is given) the runtime
    SQI names differ from the reference's (a REORDER/ADD would otherwise silently mis-score, since the μ/σ/
    inv_corr are position-indexed)."""
    if not block or not all(k in block for k in ("mean", "std", "inv_corr", "feature_names")):
        return 0.0, ""
    if feature_names is not None and list(feature_names) != list(block["feature_names"]):
        return 0.0, ""                                # feature set/order changed → reference invalid, skip
    try:
        mu = np.asarray(block["mean"], dtype=np.float64)
        sd = np.asarray(block["std"], dtype=np.float64)
        rinv = np.asarray(block["inv_corr"], dtype=np.float64)
    except (ValueError, TypeError):                   # ragged / non-numeric block
        return 0.0, ""
    x = np.asarray(features, dtype=np.float64).reshape(-1)
    d = mu.shape[0]
    if x.shape[0] != d or sd.shape[0] != d or rinv.shape != (d, d):
        return 0.0, ""
    z = np.nan_to_num((x - mu) / (sd + 1e-9), posinf=0.0, neginf=0.0)
    d2 = float(z @ rinv @ z)
    # explanation = the feature with the largest STANDALONE contribution (diagonal term z_i²·R⁻¹_ii,
    # always ≥0) — more robust than the signed cross-term decomposition for naming "which SQI is unusual".
    diag_contrib = np.diag(rinv) * z ** 2
    names = block["feature_names"]
    top = names[int(np.argmax(diag_contrib))] if len(names) == d else ""
    return max(0.0, d2), top
