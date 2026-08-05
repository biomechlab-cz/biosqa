"""THE frozen metric harness (Plan 1 §10).

This module is written **once** in Phase 0 and must never be edited by
experiment code. Every run — baseline, shared, SSL, weak-sup, cross-disciplinary
— calls :func:`evaluate` so that all reported numbers are strictly comparable.
Changing a metric definition here silently invalidates historical comparisons;
if a new metric is genuinely needed, *add* it, never redefine an existing one.

Design contract
---------------
``evaluate(y_true, y_pred, y_prob=None, ...) -> dict[str, float | dict]``

The Q0..Q3 quality scale is **ordinal**, so we report both nominal agreement
(Cohen's kappa, linear) and the ordinally-aware quadratic-weighted kappa (QWK),
which penalises a Q0/Q3 confusion far more than a Q2/Q3 one.

QWK is **symmetric** in that penalty, so it encodes no error *asymmetry*: it
charges exactly the same for grading an unusable segment Q3 as for grading an
excellent segment Q0. For signal quality only the first is harmful — a
false-clean grade forwards corrupted signal to downstream analysis, whereas a
false-unusable grade only discards good data. (An earlier version of this
docstring stated the opposite direction and attributed the asymmetry to QWK;
both were wrong — audit fix 2026-08-05.) Nothing in this bundle prices that
asymmetry, so read the Q0 row of ``per_class`` — its ``recall`` is the rate at
which unusable segments are actually caught — alongside QWK.

macro-F1 and Cohen's kappa match the CinC-2011 reporting convention.
"""
from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

__all__ = ["evaluate", "summarize_runs", "METRIC_KEYS"]

# The scalar metrics every run reports (used to validate/aggregate run stores).
METRIC_KEYS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "cohen_kappa",
    "cohen_kappa_quadratic",
    "auroc_ovr_macro",
)


def _per_class_stats(cm: np.ndarray, labels: np.ndarray) -> dict[str, dict[str, float]]:
    """Per-class precision / recall (sensitivity) / specificity / F1 / support
    derived directly from the confusion matrix ``cm`` (rows = true, cols = pred)."""
    stats: dict[str, dict[str, float]] = {}
    total = cm.sum()
    for i, lab in enumerate(labels):
        tp = float(cm[i, i])
        fn = float(cm[i, :].sum() - tp)
        fp = float(cm[:, i].sum() - tp)
        tn = float(total - tp - fn - fp)
        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # sensitivity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        stats[str(lab)] = {
            "precision": precision,
            "recall": recall,            # a.k.a. sensitivity
            "sensitivity": recall,
            "specificity": specificity,
            "f1": f1,
            "support": int(support),
        }
    return stats


def _balanced_accuracy_from_cm(cm: np.ndarray) -> float:
    """Macro-averaged recall computed from ``cm``, i.e. honouring the caller's
    ``labels`` (rows with zero support are skipped — recall is undefined there).

    Sibling of the ``balanced_accuracy`` key, which calls sklearn directly and
    therefore *ignores* ``labels``: sklearn's ``balanced_accuracy_score`` has no
    such parameter, so it also averages over samples whose class lies outside
    ``labels``. Reported as ``balanced_accuracy_fixed``; the original key is left
    untouched (frozen harness — add, never redefine).
    """
    support = cm.sum(axis=1)
    rows = support > 0
    if not rows.any():
        return float("nan")
    return float(np.mean(np.diag(cm)[rows] / support[rows]))


def evaluate(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: np.ndarray | None = None,
    *,
    labels: Sequence[int] | None = None,
    class_names: Sequence[str] | None = None,
    ordinal: bool = True,
) -> dict:
    """Compute the full frozen metric bundle.

    Parameters
    ----------
    y_true, y_pred : array-like of int class indices.
    y_prob : optional (N, C) array of class probabilities/scores; enables AUROC.
    labels : the full ordered class set (e.g. ``[0, 1, 2, 3]`` for Q0..Q3). Pass
        this explicitly so classes absent from a given fold still appear in the
        confusion matrix and per-class table (essential for fair aggregation).
        Pass the **fixed** scale, never the classes present in a fold's
        ``y_true``: any sample whose true *or* predicted class falls outside
        ``labels`` is dropped from the confusion matrix (and hence from
        cohen_kappa/QWK) while still counting towards ``n`` and ``accuracy``,
        and macro_f1 then averages over a smaller class set — both inflate the
        fold's score. That drop is counted in ``n_outside_labels`` and warned about.
        Leaving this ``None`` does not opt out of the problem — it infers the scale
        from ``np.unique(y_true | y_pred)``, i.e. present-labels — so that branch
        emits a ``UserWarning`` of its own. It is never right for a reported number.
    class_names : optional display names aligned with ``labels``.
    ordinal : if True, also report the quadratic-weighted kappa (QWK).

    Returns a dict of scalar metrics plus ``per_class`` and ``confusion_matrix``.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true {y_true.shape} and y_pred {y_pred.shape} must match")
    if y_true.size == 0:
        raise ValueError("evaluate() received empty y_true/y_pred")

    if labels is None:
        # Present-labels IS the default, and it is the unsafe one: the scale is
        # inferred from this fold, so any Q level absent from BOTH y_true and y_pred
        # vanishes from macro_f1's denominator and from the QWK weight matrix.
        # 8 of store_v8's 21 cohorts lack at least one Q level, and 27 of the 85
        # experiment scripts run on a fold-dependent scale. Measured on the real
        # 998-window cinc2011 label vector (Q1 and Q3 empty) against a model that
        # puts 12% of predictions into Q1/Q2: macro_f1 0.474 -> 0.632. The harsher
        # variant those scripts use -- labels=np.unique(y_true) -- also drops the
        # off-scale predictions outright: 120/998 samples, macro_f1 0.436 -> 0.871,
        # QWK 0.721 -> 0.858 on a CinC-shaped fold (2026-08-05 audit).
        # The n_outside_labels warning below cannot catch the labels=None case: the
        # inferred set is the union WITH y_pred, so nothing is ever dropped and
        # n_outside stays 0. Hence a warning of its own here. The returned values are
        # untouched -- adding a warning is not redefining a metric (frozen harness).
        labels = np.unique(np.concatenate([y_true, y_pred]))
        warnings.warn(
            "evaluate(): labels=None infers the class scale from this fold "
            f"(inferred labels={labels.tolist()}). A Q level absent from both y_true "
            "and y_pred is dropped from macro_f1's average and from the QWK weight "
            "matrix, which inflates the fold and makes it incomparable with folds "
            "that saw every level. Pass the FIXED scale explicitly, e.g. "
            "labels=[0,1,2,3] for Q0..Q3.",
            UserWarning,
            stacklevel=2,
        )
    labels = np.asarray(labels)
    if class_names is None:
        class_names = [str(int(x)) for x in labels]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # Samples whose true OR predicted class is outside `labels` never enter `cm`,
    # so cohen_kappa / QWK (both cm-derived) silently drop them, and macro_f1 drops
    # the whole missing class from its average — while `n` and `accuracy` still
    # count every sample. A caller that passed labels=<classes present in y_true>
    # therefore reads an inflated fold score with no signal that it happened
    # (measured: macro_f1 0.645 -> 0.860 on a 600-sample fold). Surface it (audit fix).
    in_scale = np.isin(y_true, labels) & np.isin(y_pred, labels)
    n_outside = int(y_true.size - int(in_scale.sum()))
    if n_outside:
        warnings.warn(
            f"evaluate(): {n_outside}/{y_true.size} samples have a true or predicted "
            f"class outside labels={labels.tolist()}; they are dropped from the "
            "confusion matrix (and hence cohen_kappa/QWK) while still counting in "
            "n/accuracy, and macro_f1 averages over fewer classes. Pass the FIXED "
            "class scale, e.g. labels=[0,1,2,3], not the classes present in y_true.",
            UserWarning,
            stacklevel=2,
        )
    assert int(cm.sum()) == y_true.size - n_outside, "confusion matrix lost samples"

    out: dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        # cm-derived twin that honours `labels` (see _balanced_accuracy_from_cm)
        "balanced_accuracy_fixed": _balanced_accuracy_from_cm(cm),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred, labels=labels)),
        "n": int(y_true.size),
        "n_classes": int(len(labels)),
        # samples dropped from every cm-derived metric because their true or
        # predicted class is not in `labels` (0 when the fixed scale is passed)
        "n_outside_labels": n_outside,
    }

    if ordinal and len(labels) > 2:
        out["cohen_kappa_quadratic"] = float(
            cohen_kappa_score(y_true, y_pred, labels=labels, weights="quadratic")
        )
        out["cohen_kappa_linear"] = float(
            cohen_kappa_score(y_true, y_pred, labels=labels, weights="linear")
        )
    else:
        out["cohen_kappa_quadratic"] = out["cohen_kappa"]

    # AUROC (one-vs-rest, macro). NOT robust to folds missing a class: sklearn
    # cannot form the one-vs-rest term of a class absent from y_true, so the macro
    # average is NaN whenever ANY class in `labels` is missing — not just that
    # class's term (audit fix: the old comment claimed the opposite).
    # ``auroc_status`` says WHY the value is NaN, so a genuinely malformed y_prob
    # is distinguishable from an absent class (the old bare except conflated them).
    out["auroc_ovr_macro"] = float("nan")
    out["auroc_status"] = "no_y_prob"
    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        present = np.unique(y_true)
        n_cols = int(y_prob.shape[1]) if y_prob.ndim == 2 else -1
        shape_ok = y_prob.ndim == 2 and y_prob.shape[0] == y_true.size and n_cols == len(labels)
        err: str | None = None
        # NB: the computation below is deliberately byte-for-byte the original one
        # (same branch conditions, same caught exception types) — only the *diagnosis*
        # is new, so no historical auroc_ovr_macro value can move.
        try:
            if len(labels) == 2 and y_prob.ndim == 2 and y_prob.shape[1] == 2:
                out["auroc_ovr_macro"] = float(roc_auc_score(y_true, y_prob[:, 1]))
            elif len(present) > 1:
                out["auroc_ovr_macro"] = float(
                    roc_auc_score(
                        y_true, y_prob, multi_class="ovr", average="macro", labels=labels
                    )
                )
        except (ValueError, IndexError) as exc:
            err = f"{type(exc).__name__}: {str(exc)[:80]}"
        if not np.isnan(out["auroc_ovr_macro"]):
            out["auroc_status"] = "ok"
        elif not shape_ok:
            out["auroc_status"] = "malformed_y_prob"
            warnings.warn(
                f"evaluate(): y_prob has shape {tuple(y_prob.shape)}; expected "
                f"({y_true.size}, {len(labels)}) for labels={labels.tolist()}. "
                "AUROC is NaN because the input is malformed, not because a class "
                f"is missing from y_true ({err}).",
                UserWarning,
                stacklevel=2,
            )
        elif len(present) < 2:
            out["auroc_status"] = "single_class_fold"
        elif len(present) < len(labels):
            # sklearn returns NaN (not an error) when a class in `labels` is absent
            out["auroc_status"] = "class_absent_from_y_true"
        else:
            out["auroc_status"] = f"value_error: {err}" if err else "undefined"

    per_class = _per_class_stats(cm, labels)
    # attach display names
    named = {}
    for lab, name in zip(labels, class_names):
        named[name] = per_class[str(lab)]
    out["per_class"] = named
    out["confusion_matrix"] = cm.tolist()
    out["labels"] = [int(x) for x in labels]
    out["class_names"] = list(class_names)
    return out


def summarize_runs(run_metrics: list[dict], keys: Sequence[str] | None = None) -> dict:
    """Aggregate a list of per-fold / per-seed metric dicts into mean ± std.

    Returns ``{key: {"mean": ..., "std": ..., "n": ...}}`` for each scalar key.
    Used for confirmation-tier reporting (≥5 seeds, Plan 1 §10).
    """
    if not run_metrics:
        raise ValueError("summarize_runs() received no runs")
    keys = keys or METRIC_KEYS
    agg: dict = {}
    for k in keys:
        vals = np.array(
            [r[k] for r in run_metrics if k in r and r[k] is not None and not _isnan(r[k])],
            dtype=float,
        )
        if vals.size == 0:
            agg[k] = {"mean": float("nan"), "std": float("nan"), "n": 0}
        else:
            agg[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1) if vals.size > 1 else 0.0), "n": int(vals.size)}
    return agg


def _isnan(x) -> bool:
    try:
        return bool(np.isnan(x))
    except (TypeError, ValueError):
        return False
