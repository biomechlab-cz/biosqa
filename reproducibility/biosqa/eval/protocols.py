"""Frozen evaluation *protocols* (Plan 1 §10).

Split generators and statistical comparison helpers used by every experiment.
Like :mod:`biosqa.eval.metrics`, this is written once and not edited by
experiment code — the protocols define what "generalization" means for the paper:

* **LOSO** — leave-one-subject-out within a dataset (subject-level generalization).
* **LODO** — leave-one-dataset-out (the real cross-cohort test; a known weak
  point of CinC-trained ECG models).
* **label-fraction curves** — stratified subsampling of the *training* labels to
  10/25/50/100 % for the SSL claim (C2).
* **paired significance** — Wilcoxon signed-rank across folds/seeds for the
  shared-vs-specialist comparison (C1).

All generators operate on a *segment index* (a DataFrame-like table with at
least ``subject_id`` and ``dataset`` columns) and yield ``(train_idx, test_idx)``
integer-position arrays so they are backend-agnostic (numpy/pandas/polars rows).
"""
from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np

__all__ = [
    "loso_splits",
    "leave_one_dataset_out",
    "stratified_label_fractions",
    "paired_wilcoxon",
]


def _as_array(col) -> np.ndarray:
    """Accept pandas/polars Series, list, or ndarray -> 1D object/np array."""
    if hasattr(col, "to_numpy"):
        return col.to_numpy()
    return np.asarray(col)


def loso_splits(subject_ids: Sequence) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_idx, test_idx)`` leaving out one subject at a time.

    ``subject_ids`` is aligned with the segment rows; subject identity is
    whatever makes a subject unique (prefix with dataset if ids collide across
    datasets before calling).
    """
    sid = _as_array(subject_ids)
    uniq = _stable_unique(sid)
    all_idx = np.arange(len(sid))
    for s in uniq:
        test_mask = sid == s
        yield all_idx[~test_mask], all_idx[test_mask]


def leave_one_dataset_out(datasets: Sequence) -> Iterator[tuple[np.ndarray, np.ndarray, str]]:
    """Yield ``(train_idx, test_idx, held_out_dataset)`` leaving out one dataset.

    This is the cross-cohort generalization protocol (train on some datasets,
    test on a held-out cohort/device).
    """
    ds = _as_array(datasets)
    uniq = _stable_unique(ds)
    all_idx = np.arange(len(ds))
    for d in uniq:
        test_mask = ds == d
        yield all_idx[~test_mask], all_idx[test_mask], str(d)


def stratified_label_fractions(
    labels: Sequence[int],
    fractions: Sequence[float] = (0.10, 0.25, 0.50, 1.00),
    *,
    seed: int = 0,
    min_per_class: int = 1,
) -> dict[float, np.ndarray]:
    """Class-stratified nested subsamples of the labeled pool for SSL curves.

    Returns ``{fraction: selected_indices}``. Subsamples are **nested** (each
    larger fraction is a superset of the smaller) so the label-fraction curve
    isolates the effect of label *quantity*, not sample identity. At least
    ``min_per_class`` samples per class are kept at every fraction.
    """
    y = _as_array(labels).astype(int).ravel()
    rng = np.random.default_rng(seed)
    fractions = sorted(fractions)
    # Per-class shuffled index order; nesting = take prefixes of this order.
    order_by_class: dict[int, np.ndarray] = {}
    for c in np.unique(y):
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        order_by_class[int(c)] = idx_c

    out: dict[float, np.ndarray] = {}
    for f in fractions:
        chosen: list[np.ndarray] = []
        for c, idx_c in order_by_class.items():
            k = max(min_per_class, int(round(f * len(idx_c))))
            k = min(k, len(idx_c))
            chosen.append(idx_c[:k])
        sel = np.sort(np.concatenate(chosen)) if chosen else np.array([], dtype=int)
        out[float(f)] = sel
    return out


def paired_wilcoxon(a: Sequence[float], b: Sequence[float]) -> dict:
    """Paired Wilcoxon signed-rank test of ``a`` vs ``b`` across folds/seeds (C1).

    Returns ``{statistic, p_value, median_diff, n}``. Falls back to a paired
    t-test-free summary when SciPy is unavailable or n is too small.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired_wilcoxon requires equal-length paired samples")
    diff = a - b
    result = {"median_diff": float(np.median(diff)), "mean_diff": float(diff.mean()), "n": int(a.size)}
    try:
        from scipy.stats import wilcoxon

        # zero_method="wilcox" drops zero-diffs; guard the all-equal case.
        if np.allclose(diff, 0):
            result.update({"statistic": 0.0, "p_value": 1.0})
        else:
            stat, p = wilcoxon(a, b)
            result.update({"statistic": float(stat), "p_value": float(p)})
    except Exception:  # pragma: no cover - scipy always present here
        result.update({"statistic": float("nan"), "p_value": float("nan")})
    return result


def _stable_unique(arr: np.ndarray) -> np.ndarray:
    """Unique values preserving first-appearance order (deterministic folds)."""
    seen: dict = {}
    for v in arr:
        key = v.item() if hasattr(v, "item") else v
        if key not in seen:
            seen[key] = True
    return np.array(list(seen.keys()), dtype=arr.dtype if arr.dtype != object else object)
