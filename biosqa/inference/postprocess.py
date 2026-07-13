"""Shared calibration + sanitation of a model's raw softmax — the ONE place BOTH inference paths
(:class:`workers.qt_threads.InferenceTask` in memory, :func:`inference.streaming.stream_infer`
out-of-core) turn ONNX probabilities into the numbers the user sees.

It exists because the two paths had drifted: the in-memory path fail-safed a non-finite softmax and
temperature-scaled the grade before deriving confidence/uncertainty/conformal, while the streamed
path read them straight off the RAW softmax. Same signal, different exported numbers, decided only by
whether the record tripped ``streaming.LARGE_RECORD_SAMPLES`` — and the RAW ones are the
mis-calibrated ones (every shipped card reports grade ECE ~0.26 raw vs ~0.07 scaled). Calibration is
a property of the MODEL, not of how much RAM the record happened to need, so it lives here and both
paths call it.

Pure numpy (no Qt / no onnxruntime) so it is directly unit-testable.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from biosqa.inference.conformal import temperature_scale

__all__ = [
    "sanitize_probs",
    "calibrate_grade_probs",
    "calibrate_prediction",
    "confidences_from",
    "normalized_entropy",
]


def sanitize_probs(probs):
    """``(probs[float64 copy], non_finite_mask)`` — replace every non-finite ROW with a UNIFORM
    distribution.

    A NaN/inf input window passes through the card's constant normalization to a NaN softmax; argmax
    would still pick a class but the confidence, entropy and conformal set would all be NaN and get
    RLE-encoded + EXPORTED. A uniform row degrades that window to the worst tier at maximal entropy —
    an honest "we could not grade this", never an invented grade."""
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 2 or not p.shape[0]:
        return p, np.zeros(0, dtype=bool)
    p = p.copy()
    non_finite = ~np.isfinite(p).all(axis=1)
    if non_finite.any():
        p[non_finite] = 1.0 / p.shape[1]
    return p, non_finite


def calibrate_grade_probs(q_probs, card):
    """``(calibrated_probs, non_finite_mask)`` for the ordinal grade head.

    Sanitize (above), then temperature-scale ONCE by the card's ``grade_temperature`` so every
    user-facing UQ surface — confidence, entropy-uncertainty, APS prediction set — is derived from
    the SAME calibrated distribution the conformal threshold was fit on. Monotonic, so the tiers
    (argmax) are unchanged; a uniform (ungradeable) row stays uniform."""
    p, non_finite = sanitize_probs(q_probs)
    t = float(getattr(card, "grade_temperature", 1.0) or 1.0)
    if t != 1.0 and p.shape[0]:
        p = temperature_scale(p, t)
    return p, non_finite


def calibrate_prediction(pred, card):
    """A calibrated COPY of a whole multi-head prediction: the ordinal grade head scaled by the card's
    ``grade_temperature`` and the binary usable head by its ``usable_temperature`` (every shipped card
    calibrates both; the usable one was previously loaded and never applied). The multilabel artifact
    head is independent sigmoids, not a distribution, so it is left untouched.

    ``usable_temperature`` is read with ``getattr`` so a card object that predates the property still
    works (identity, not a guess). A prediction that is not the dataclass carrier (a test double) is
    returned unchanged rather than reconstructed."""
    per_head = getattr(pred, "per_head", None)
    if not isinstance(per_head, dict):
        return pred
    out = dict(per_head)
    for head in getattr(card, "heads", ()):
        probs = out.get(head.name)
        if probs is None or not len(probs):
            continue
        if head.kind == "ordinal":
            out[head.name] = calibrate_grade_probs(probs, card)[0]
        elif head.kind == "binary":
            t = float(getattr(card, "usable_temperature", 1.0) or 1.0)
            p, _nf = sanitize_probs(probs)
            out[head.name] = temperature_scale(p, t) if t != 1.0 and p.shape[0] else p
    try:
        return replace(pred, per_head=out)
    except TypeError:  # noqa: BLE001 - not a dataclass carrier; the raw prediction is still valid
        return pred


def confidences_from(probs, non_finite=None):
    """Per-window confidence = max CALIBRATED class probability, forced to zero on a window whose raw
    softmax was non-finite (a garbage window reads as a zero-confidence worst tier, never NaN)."""
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 2 or not p.shape[0]:
        return np.zeros(0, dtype=np.float64)
    conf = p.max(axis=1)
    if non_finite is not None and len(non_finite) == conf.shape[0]:
        mask = np.asarray(non_finite, dtype=bool)
        if mask.any():
            conf = conf.copy()
            conf[mask] = 0.0
    return conf


def normalized_entropy(probs):
    """Per-window normalized predictive entropy of the CALIBRATED grade distribution (0=certain,
    1=uniform) — free (no extra inference); the app's uncertainty column."""
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 2 or not p.shape[0]:
        return np.zeros(0, dtype=np.float64)
    k = p.shape[1]
    if k <= 1:
        return np.zeros(p.shape[0], dtype=np.float64)
    q = np.clip(p, 1e-12, 1.0)
    return -(q * np.log(q)).sum(axis=1) / np.log(k)
