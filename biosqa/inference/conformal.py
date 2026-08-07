"""Split-conformal prediction sets for the ordinal grade head (research3 UQ; Romano, Sesia & Candès 2020,
"Classification with Valid and Adaptive Coverage" — APS; Angelopoulos & Bates 2021).

A single softmax gives a point grade with no coverage notion. APS turns the *calibrated* softmax into a
prediction SET whose size tracks how cleanly the model separates grades: size 1 = confident enough to commit
to one grade; size ≥ 2 (e.g. {Q2, Q3}) = it cannot separate those grades at the calibrated threshold → an
explicit *abstain / ambiguous* signal the point prediction hides.

Coverage caveat (honest): a valid ``1−α`` coverage guarantee requires exchangeability between the threshold's
CALIBRATION set and deployment. The shipped ECG threshold was fit on a reference (possibly synth-augmented)
set whose provenance differs from live recordings, and the app decodes the set from a run-MEAN distribution
(not per-window), so the app surfaces this as an *ambiguity* signal, NOT as a hard coverage percentage. A
fresh split-conformal calibration on the deployed model + a documented reference set would restore the exact %.

The card's ``grade_nonconformity_threshold`` was calibrated on TEMPERATURE-SCALED grade probabilities —
and so is the softmax the app already holds: every shipped graph BAKES its grade temperature (final grade
path ``Log -> Div`` by the card's constant, ``location: onnx_graph``). Feed :func:`aps_prediction_set` the
probabilities onnxruntime returned, unscaled. Pure numpy.
"""
from __future__ import annotations

import numpy as np

# temperature_scale is deliberately NOT exported: it is a correct utility with no correct caller in this
# app, and calling it on the app's softmax would scale a second time (6.25x logit sharpening on ECG).
__all__ = ["aps_prediction_set"]


def temperature_scale(probs: np.ndarray, temperature: float | None) -> np.ndarray:
    """Re-apply temperature ``T`` to an UNSCALED softmax. Because ``log(softmax(z))`` = ``z`` up to a
    per-row constant and softmax is shift-invariant, ``softmax(log(p)/T)`` == ``softmax(z/T)`` exactly —
    so this recovers the temperature-scaled distribution from probabilities alone (the app has no raw
    logits). ``T`` of ``None`` or ``1.0`` is a no-op.

    DO NOT call this on anything the app's ONNX session returned: those graphs already divide the grade
    log-probabilities by the card's temperature, so a second application squares the sharpening (ECG's
    T=0.4 becomes an effective 0.16) and collapses almost every APS prediction set to size 1, silently
    deleting the ambiguity signal this module exists to provide. Kept only for offline analysis of an
    un-baked graph."""
    p = np.asarray(probs, dtype=np.float64)
    if temperature is None or float(temperature) == 1.0:
        return p
    z = np.log(np.clip(p, 1e-9, 1.0)) / float(temperature)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / (e.sum(axis=-1, keepdims=True) + 1e-12)


def aps_prediction_set(probs, class_order, threshold) -> tuple[str, ...]:
    """APS prediction set for one (already temperature-scaled) grade distribution: sort classes by
    probability descending and include the top classes until their cumulative probability first reaches
    ``threshold`` — the smallest set with the calibrated coverage. Returns the included class labels in
    ``class_order`` (index) order. Empty tuple when there is no threshold / no probabilities."""
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    if p.size == 0 or threshold is None:
        return ()
    order = np.argsort(-p)
    cum = np.cumsum(p[order])
    # first index where cum >= threshold; the −1e-12 absorbs float-summation slop (e.g. 0.6+0.3=0.8999…)
    k = min(int(np.searchsorted(cum, float(threshold) - 1e-12)), p.size - 1)
    idx = sorted(int(i) for i in order[:k + 1])
    return tuple(str(class_order[i]) for i in idx if 0 <= i < len(class_order))
