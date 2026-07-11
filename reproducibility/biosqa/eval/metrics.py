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
which penalises a Q3->Q0 confusion far more than Q3->Q2 — the clinically correct
cost structure (misjudging an excellent segment as unacceptable is the costly
error). macro-F1 and Cohen's kappa match the CinC-2011 reporting convention.
"""
from __future__ import annotations

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
        labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = np.asarray(labels)
    if class_names is None:
        class_names = [str(int(x)) for x in labels]

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    out: dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred, labels=labels)),
        "n": int(y_true.size),
        "n_classes": int(len(labels)),
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

    # AUROC (one-vs-rest, macro). Robust to folds missing a class.
    out["auroc_ovr_macro"] = float("nan")
    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        try:
            present = np.unique(y_true)
            if len(labels) == 2 and y_prob.ndim == 2 and y_prob.shape[1] == 2:
                out["auroc_ovr_macro"] = float(roc_auc_score(y_true, y_prob[:, 1]))
            elif len(present) > 1:
                out["auroc_ovr_macro"] = float(
                    roc_auc_score(
                        y_true, y_prob, multi_class="ovr", average="macro", labels=labels
                    )
                )
        except (ValueError, IndexError):
            pass  # leave NaN; a single-class fold cannot define AUROC

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
