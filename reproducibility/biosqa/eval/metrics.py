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

Beside the bundle, this module exposes standalone reporting helpers that
:func:`evaluate` deliberately does not return — :func:`usable_auroc`,
:func:`usable_operating_points` (AUC-PR + SEN at fixed SPE) and
:func:`overlap_accuracy` (ordinal OAc). They are functions rather than dict keys
because ``evaluate``'s returned dict is a hashed contract; see the section
comment above :data:`SPEC_OPERATING_POINTS` for the full reasoning.
"""
from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

__all__ = [
    "evaluate",
    "summarize_runs",
    "usable_auroc",
    "METRIC_KEYS",
    # --- operating-point / ordinal reporting (added 2026-08-05, see below) ---
    "average_precision",
    "chance_average_precision",
    "sensitivity_at_specificity",
    "usable_operating_points",
    "overlap_accuracy",
    "SPEC_OPERATING_POINTS",
]

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


def usable_auroc(y_true, y_prob) -> float:
    """AUROC of the *usable* decision P(Q2)+P(Q3) against the truth ``Q >= 2``.

    A reported quantity (LODO tables, deployment-parity checks) that the bundle
    above does not produce: ``auroc_ovr_macro`` is one-vs-rest over four classes,
    whereas deployment asks a single binary question — forward this window or
    discard it — and scores it with the summed usable-class mass, not with an
    argmax. Until 2026-08-05 it was hand-rolled in **ten** places (9 under
    ``experiments/``, 1 in ``scripts/verify_deployment_parity.py``), all
    semantically identical — the one exception to "every reported number is scored
    by one frozen module" (``methods.tex``). This is that definition, moved in, and
    **all ten call sites now import it** — zero local definitions remain anywhere in
    ``experiments/``, ``scripts/`` or ``src/`` (pinned by
    ``tests/test_metrics.py::test_no_module_redefines_the_frozen_usable_auroc``).
    Every copy was verified bit-identical to this function before deletion, so no
    published number differs. The unconditional claim is therefore supportable for
    this quantity.

    One RELATED-BUT-DIFFERENT quantity is deliberately not here:
    ``scripts/verify_deployment_parity.py`` also computes a ``usable_head_auroc``,
    which scores the *usable head's own* ``P(usable)`` rather than the grade-derived
    collapse below. It is a diagnostic for that harness, appears in no manuscript
    claim, and would be a different metric — not a restatement of this one.

    Moved in, not rewritten: the body is the byte-for-byte expression from the
    pre-move ``experiments/lodo_cutpoint.py``, so every historical value is
    reproduced exactly — including the degenerate paths (an empty fold raises
    ``ValueError`` from ``.min()``; a NaN/inf probability raises ``ValueError``
    from sklearn). ``y_true`` is used *without* an ``np.asarray`` coercion for the
    same reason: adding one would make a Python list return a number where the ten
    originals raised ``TypeError``, and "add, never redefine" covers exceptions
    too. Pass an ndarray of integer grades.

    Returns NaN when the fold is single-class in the usable sense (all Q<2 or all
    Q>=2) — common on a held-out LODO cohort — so callers must NaN-filter before
    averaging, exactly as they already do for the per-arm aggregates.

    Deliberately NOT a key of :func:`evaluate` and NOT in ``METRIC_KEYS``: it is
    defined only on the fixed 4-class Q0..Q3 scale (it indexes columns 2 and 3),
    while ``evaluate`` also serves binary and arbitrary ``labels``; and the
    harness's own change-detector hashes ``evaluate``'s complete returned dict
    (``tests/test_metrics.py::GOLDEN_SHA``), so a new key there would be a
    contract change, not an addition. Call it alongside ``evaluate``.
    """
    yb = (y_true >= 2).astype(int)
    if yb.min() == yb.max():
        return float("nan")
    return float(roc_auc_score(yb, y_prob[:, 2] + y_prob[:, 3]))


# ---------------------------------------------------------------------------
# Operating-point and ordinal reporting (added 2026-08-05).
#
# WHY these are standalone helpers and NOT keys of evaluate():
#   `evaluate`'s complete returned dict is hashed by
#   tests/test_metrics.py::GOLDEN_SHA. Adding a key there changes the dict that
#   every caller and every historical result JSON is compared against — a
#   contract change, not an addition. `usable_auroc` was added under exactly
#   this reasoning and states it in its own docstring; these follow it.
#   Two of them are also undefined on `evaluate`'s general `labels`: the usable
#   collapse indexes columns 2 and 3 of a fixed Q0..Q3 probability matrix, and
#   AUC-PR/SEN@SPE are properties of a *binary decision with a score*, which a
#   4-class argmax bundle does not have. Call them alongside `evaluate`.
#
# WHY they exist at all (2026-08-05 SOTA survey, items P6 and P11): the SQA
# literature reports discrimination (AUROC, accuracy) and hides operating-point
# behaviour, and no published ordinal ECG SQA line reports QWK. Neither gap is
# a modelling problem; both are reporting gaps we can close for free.
# ---------------------------------------------------------------------------

#: The specificities at which sensitivity is reported. These three are not
#: arbitrary: they are the operating points Peh, Yao & Dauwels (2022),
#: "Transformer Convolutional Neural Networks for Automated Artifact Detection
#: in Scalp EEG", EMBC, pp. 3599-3602, doi:10.1109/embc48229.2022.9871916 use
#: for scalp-EEG artifact detection. Their abstract (verified 2026-08-05) reads:
#: "The resulting detector achieves a sensitivity (SEN) of 42.0%, 32.0%, and
#: 13.3%, at a specificity (SPE) of 95%, 97%, and 99%, respectively." Reporting
#: on the same grid is what makes our EEG numbers comparable to theirs at all.
#: (The 2026-08-05 survey quoted 0.604/0.518/0.353 for this same triple, citing
#: Table VI of the arXiv preprint. That does NOT match the published EMBC
#: abstract above, which is the only version verified here. Use these grid
#: points; do not quote either SEN triple as "the" Peh number without saying
#: which version it came from.)
SPEC_OPERATING_POINTS = (0.95, 0.97, 0.99)


def _spec_tag(specificity: float) -> str:
    """Key suffix for a specificity: 0.95 -> "95", 0.995 -> "99.5"."""
    pct = specificity * 100.0
    return f"{pct:.0f}" if abs(pct - round(pct)) < 1e-9 else f"{pct:g}"


def _as_binary(y_true_bin) -> np.ndarray:
    """Coerce a binary label vector, refusing anything that is not 0/1.

    An `.astype(int)` on a float vector containing 0.5 or NaN silently produces
    a valid-looking label vector, which would make a wrong AUC-PR rather than an
    error. These helpers are new, so they get the strict contract that the
    frozen-by-history functions above could not be given retroactively.
    """
    y = np.asarray(y_true_bin).ravel()
    if y.size == 0:
        raise ValueError("received an empty y_true")
    uniq = np.unique(y)
    if not np.all(np.isin(uniq, (0, 1))):
        raise ValueError(
            f"y_true must be binary 0/1 (or bool); got values {uniq[:8].tolist()}. "
            "Collapse the ordinal grade yourself, e.g. (y >= 2) for the usable decision."
        )
    return y.astype(np.int64)


def average_precision(y_true_bin, y_score) -> float:
    """AUC-PR (average precision) of one binary decision — the P6 companion to AUROC.

    ``average_precision_score``'s step-wise estimator, i.e. sum over thresholds of
    (R_n - R_{n-1}) * P_n. Deliberately not ``auc(recall, precision)``: trapezoidal
    interpolation of a PR curve is optimistic, and the two disagree by enough to
    matter at low prevalence.

    Why this is worth reporting alongside every AUROC: AUROC is invariant to class
    prevalence, AUC-PR is not, so a detector can look strong on AUROC and be
    unusable at the prevalence it will actually meet. The 2026-08-05 survey's
    example is LUNA — AUROC 0.921 with AUC-PR 0.528 on the *same* EEG test set —
    and FEMBA ~0.89 AUROC with ~0.51 AUPR. **Neither of those two numbers was
    re-verified here** (the survey is the only source); the asymmetry they
    illustrate is nonetheless a property of the metrics, and
    ``tests/test_metrics.py::test_auc_pr_exposes_what_auroc_hides_at_low_prevalence``
    reproduces it on synthetic data rather than on their authority.

    A bare AUC-PR is uninterpretable without its chance level, which is the
    positive prevalence, NOT 0.5 — see :func:`chance_average_precision`, and prefer
    :func:`usable_operating_points`, which returns both plus the ratio.

    Returns NaN on a single-class fold, matching :func:`usable_auroc`'s convention
    so callers can NaN-filter one way for both. (sklearn does not: it returns 1.0
    for an all-positive vector and 0.0 plus a warning for an all-negative one —
    two *plausible-looking* numbers that would silently enter a mean.)
    """
    y = _as_binary(y_true_bin)
    s = np.asarray(y_score).ravel()
    if s.size != y.size:
        raise ValueError(f"y_true ({y.size}) and y_score ({s.size}) must match")
    if y.min() == y.max():
        return float("nan")
    return float(average_precision_score(y, s))


def chance_average_precision(y_true_bin) -> float:
    """Positive prevalence — the AUC-PR a random ranker achieves on this fold.

    The "trivial-baseline column" the survey asks for beside every accuracy and
    every AUC-PR. An AUC-PR of 0.53 is excellent at 5% prevalence and worthless at
    60%, and published AUPRs are routinely quoted without the prevalence needed to
    tell those apart. Cheap to compute, so there is no excuse for omitting it.
    """
    y = _as_binary(y_true_bin)
    return float(y.mean())


def sensitivity_at_specificity(
    y_true_bin,
    y_score,
    specificities: Sequence[float] = SPEC_OPERATING_POINTS,
) -> dict[str, float]:
    """SEN at fixed SPE — the operating points the EEG literature reports (P6).

    For each target, the **maximum** sensitivity over all thresholds whose
    specificity is at least the target, plus the specificity actually achieved
    there and the score threshold that achieves it. Returns a flat dict so it can
    go straight into a result JSON and through :func:`summarize_runs`::

        {"sensitivity_at_spec_95": .., "specificity_at_spec_95": .., "threshold_at_spec_95": .., ...}

    Read the *achieved* specificity: an ROC is a step function, so a target is
    generally over-shot (95% requested, 96.2% delivered), and on a small fold it
    can be over-shot badly enough that the reported sensitivity is not the one a
    95%-specificity operating point would really give.

    Two implementation details that are the whole correctness of this function:

    * ``drop_intermediate=False``. sklearn's default *True* deletes ROC vertices
      it considers collinear, and a deleted vertex is exactly the one that may
      carry the best feasible sensitivity. Measured on a 10-vertex diagonal ROC
      (tied pos/neg score pairs): max TPR at FPR <= 0.30 reads **0.1** with the
      default and the correct **0.3** with it off — a 3x understatement. Pinned by
      ``test_sensitivity_at_specificity_survives_sklearn_drop_intermediate``.
    * take the max over feasible thresholds, not the first feasible one. TPR is
      non-decreasing along the returned curve, so the naive "first point past the
      target" reads far too low. Among ties the earliest index wins, which is the
      one with the smallest FPR, i.e. the highest achieved specificity.

    ``roc_curve`` always emits the (0, 0) vertex, so the feasible set is never
    empty and an unreachable target degrades to sensitivity 0.0 at threshold inf
    ("call nothing positive") rather than to NaN. NaN is reserved for the
    single-class fold, where the question is undefined.
    """
    y = _as_binary(y_true_bin)
    s = np.asarray(y_score).ravel()
    if s.size != y.size:
        raise ValueError(f"y_true ({y.size}) and y_score ({s.size}) must match")
    targets = [float(t) for t in specificities]
    if any(not (0.0 <= t <= 1.0) for t in targets):
        raise ValueError(f"specificities must lie in [0, 1]; got {targets}")

    out: dict[str, float] = {}
    if y.min() == y.max():
        for t in targets:
            tag = _spec_tag(t)
            out[f"sensitivity_at_spec_{tag}"] = float("nan")
            out[f"specificity_at_spec_{tag}"] = float("nan")
            out[f"threshold_at_spec_{tag}"] = float("nan")
        return out

    fpr, tpr, thr = roc_curve(y, s, drop_intermediate=False)
    for t in targets:
        # 1e-12 slack: fpr is an exact fp/n_neg ratio while 1 - t carries the
        # float error of the literal (1 - 0.97 == 0.030000000000000027), so an
        # exact-equality comparison would drop the vertex that *is* the target.
        feasible = fpr <= (1.0 - t) + 1e-12
        idx = int(np.argmax(np.where(feasible, tpr, -1.0)))
        tag = _spec_tag(t)
        out[f"sensitivity_at_spec_{tag}"] = float(tpr[idx])
        out[f"specificity_at_spec_{tag}"] = float(1.0 - fpr[idx])
        out[f"threshold_at_spec_{tag}"] = float(thr[idx])
    return out


def usable_operating_points(
    y_true,
    y_prob,
    specificities: Sequence[float] = SPEC_OPERATING_POINTS,
) -> dict:
    """The full P6 reporting row for the deployment decision (forward or discard).

    Same question and same collapse as :func:`usable_auroc` — score
    ``P(Q2) + P(Q3)`` against truth ``Q >= 2`` — scored on the axes AUROC hides:
    AUC-PR, its chance level, their ratio, and SEN at each target specificity.

    The AUROC entry is :func:`usable_auroc` itself, called first and unmodified,
    so this bundle can never disagree with the published usable-AUROC numbers and
    inherits its input contract exactly (including the degenerate paths: an empty
    fold raises, a Python-list ``y_true`` raises, a NaN probability raises). Pass
    an ndarray of integer grades and an (N, >=4) probability matrix.

    ``usable_auc_pr_over_chance`` is the number to read, not ``usable_auc_pr``: on
    a cohort that is 90% usable, an AUC-PR of 0.93 is *below* chance.
    """
    auroc = usable_auroc(y_true, y_prob)          # first: it defines the contract
    score = y_prob[:, 2] + y_prob[:, 3]           # same expression as usable_auroc
    yb = (y_true >= 2).astype(int)

    ap = average_precision(yb, score)
    chance = chance_average_precision(yb)
    out: dict = {
        "usable_auroc": auroc,
        "usable_auc_pr": ap,
        "usable_auc_pr_chance": chance,
        "usable_auc_pr_over_chance": float(ap / chance) if chance > 0 else float("nan"),
        "usable_prevalence": chance,
        "n": int(yb.size),
        "n_usable": int(yb.sum()),
    }
    for k, v in sensitivity_at_specificity(yb, score, specificities).items():
        out[f"usable_{k}"] = v
    return out


def overlap_accuracy(y_true, y_pred, *, tolerance: int = 1, labels: Sequence | None = None) -> float:
    """Ordinal overlap accuracy (OAc): a prediction within ``tolerance`` grades counts (P11).

    ``mean(|rank(y_pred) - rank(y_true)| <= tolerance)``. At ``tolerance=1`` an
    adjacent-grade call (Q2 graded Q3) is scored correct and only a two-or-more
    level error (Q0 graded Q2) is wrong.

    Why bother when we already report QWK: Li, Rajagopalan & Clifford (2014), "A
    machine learning approach to multi-level ECG signal quality classification",
    *Computer Methods and Programs in Biomedicine* 117(3), 435-447,
    doi:10.1016/j.cmpb.2014.09.002 — the only published *ordinal* ECG SQA line our
    Q0..Q3 grades could be cited against — report accuracy/overlap-accuracy pairs
    on a 5-level scale and no kappa at all. Without OAc our numbers and theirs
    share no metric, so the comparison cannot be made in either direction.
    ⚠ Their full text is paywalled and **their reported Ac/OAc values were not
    verified here**; the 2026-08-05 survey quotes 80.26/98.60 (simulated) and
    57.26/94.23 (unseen MITDB). Verify against the paper before any of those
    numbers enters the manuscript. What is verified is the bibliographic record
    above and that the metric is theirs, not ours.

    Denominator is ``len(y_true)`` — the same one ``accuracy_score`` uses, so
    ``(evaluate(...)["accuracy"], overlap_accuracy(...))`` is an internally
    consistent Ac/OAc pair. It therefore does NOT drop out-of-scale samples the
    way ``evaluate``'s confusion-matrix-derived metrics do; ``tolerance=0``
    reproduces ``evaluate(...)["accuracy"]`` exactly.

    ``labels`` is the ordered scale. Leave it None for Q0..Q3, where the class
    values already *are* the ordinal ranks. Pass it when the codes are not
    contiguous integers (e.g. ``labels=[1, 2, 3, 4, 5]`` for the Li et al. scale,
    or non-numeric grades) — ranks then come from position in ``labels``, and a
    value outside it raises rather than being silently ranked by its magnitude.
    """
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    if yt.shape != yp.shape:
        raise ValueError(f"y_true {yt.shape} and y_pred {yp.shape} must match")
    if yt.size == 0:
        raise ValueError("overlap_accuracy() received empty y_true/y_pred")
    if int(tolerance) != tolerance or tolerance < 0:
        raise ValueError(f"tolerance must be a non-negative integer; got {tolerance!r}")

    if labels is not None:
        order = list(np.asarray(labels).ravel().tolist())
        if len(set(order)) != len(order):
            raise ValueError(f"labels must be unique; got {order}")
        rank = {v: i for i, v in enumerate(order)}
        unknown = sorted({v for v in yt.tolist() + yp.tolist() if v not in rank}, key=repr)
        if unknown:
            raise ValueError(
                f"values {unknown[:8]} are outside labels={order}; overlap accuracy is "
                "undefined for a class with no position on the ordinal scale"
            )
        yt = np.array([rank[v] for v in yt.tolist()])
        yp = np.array([rank[v] for v in yp.tolist()])

    # float64 on purpose: an unsigned grade array (uint8 from a packed store)
    # would wrap on subtraction and turn a 3-level error into 253.
    diff = np.abs(yt.astype(np.float64) - yp.astype(np.float64))
    return float(np.mean(diff <= tolerance))


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
