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

The card's ``grade_nonconformity_threshold`` was calibrated on TEMPERATURE-SCALED grade probabilities, so
:func:`temperature_scale` must be applied to the app's raw softmax first (verified empirically: raw softmax
makes almost every window ambiguous; T-scaled gives the intended confident/ambiguous split). Pure numpy.
"""
from __future__ import annotations

import numpy as np

__all__ = ["temperature_scale", "aps_prediction_set"]


def temperature_scale(probs: np.ndarray, temperature: float | None) -> np.ndarray:
    """Re-apply temperature ``T`` to a RAW softmax. Because ``log(softmax(z))`` = ``z`` up to a per-row
    constant and softmax is shift-invariant, ``softmax(log(p)/T)`` == ``softmax(z/T)`` exactly — so this
    recovers the temperature-scaled distribution from probabilities alone (the app has no raw logits).
    ``T`` of ``None`` or ``1.0`` is a no-op."""
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
