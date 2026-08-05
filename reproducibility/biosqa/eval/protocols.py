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
  shared-vs-specialist comparison (C1), and Benjamini-Hochberg FDR control over
  the *family* of such comparisons.
* **nested cohort selection** — the honest version of "pick the best arm on the
  LODO folds, then report its LODO score": choose on g-1 cohorts, score on the
  held-out one.

All generators operate on a *segment index* (a DataFrame-like table with at
least ``subject_id`` and ``dataset`` columns) and yield ``(train_idx, test_idx)``
integer-position arrays so they are backend-agnostic (numpy/pandas/polars rows).
"""
from __future__ import annotations

import warnings
from typing import Iterator, Sequence

import numpy as np

__all__ = [
    "loso_splits",
    "leave_one_dataset_out",
    "stratified_label_fractions",
    "paired_wilcoxon",
    "benjamini_hochberg",
    "nested_cohort_selection",
    "cluster_bootstrap_ci",
    "cluster_bootstrap_ci_detail",
    "cluster_bootstrap_statistic",
    "MIN_CLUSTERS_FOR_PERCENTILE",
    "PAPER_CI_METHOD",
    "dump_raw_points",
]

# Below this many clusters the plain percentile cluster bootstrap is materially
# anti-conservative (measured coverage of a nominal 95 % interval: g=4 -> 81 %,
# g=5 -> 84 %, g=7 -> 88 %, g=20 -> 91 %; two independent 2000-replicate
# simulations agree to ~2 pp — see cluster_bootstrap_ci's docstring for the full
# table). LODO cohort counts sit right in that regime, so cluster_bootstrap_ci
# warns and offers method="t" (94-95 % there).
MIN_CLUSTERS_FOR_PERCENTILE = 10

# The paper-facing estimator. ``cluster_bootstrap_ci``'s *default* stays
# "percentile" because this module is the frozen harness and every historical call
# site must keep reproducing its logged interval; but no interval reported in the
# manuscript may use it at LODO cohort counts. Every publication-facing analysis
# (scripts/analyze_locked_results.py) passes ``method=PAPER_CI_METHOD`` explicitly
# and persists the legacy percentile interval beside it rather than replacing it.
PAPER_CI_METHOD = "t"


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


def benjamini_hochberg(pvals: Sequence[float], alpha: float = 0.05) -> dict:
    """Benjamini-Hochberg step-up FDR control over a *family* of p-values.

    Added (never a redefinition — no existing metric moves) because
    ``manuscript/results.tex:91`` reported the *outcome* of a BH correction
    ("No comparison remained significant after Benjamini-Hochberg correction")
    while no implementation existed anywhere in the repo: a repo-wide grep for
    ``benjamini|hochberg|fdr_bh|multipletests`` hit only the manuscript, the
    audit and site-packages. Any reported multiplicity correction has to come
    from here and be persisted next to the raw p-values it adjusts.

    Returns ``{p_values, q_values, rejected, n_rejected, min_q, m, alpha,
    method}`` with ``q_values`` in the **input order**. ``q_(i) = min_{k>=i}
    m*p_(k)/k`` clipped to 1 — the running minimum from the largest p downwards
    is what keeps q monotone in p, so ``rejected == (q <= alpha)`` is exactly the
    classical step-up rule (largest ``i`` with ``p_(i) <= i*alpha/m``).

    p-values must be finite and in ``[0, 1]``; a NaN (``paired_wilcoxon`` returns
    one when SciPy is missing) raises rather than silently shrinking the family.
    """
    p = np.asarray(pvals, dtype=float).ravel()
    if p.size and (not np.all(np.isfinite(p)) or p.min() < 0.0 or p.max() > 1.0):
        raise ValueError(
            f"benjamini_hochberg: p-values must be finite and in [0, 1]; got {p.tolist()}"
        )
    m = int(p.size)
    base = {"m": m, "alpha": float(alpha), "method": "benjamini-hochberg",
            "p_values": p.tolist()}
    if m == 0:
        return {**base, "q_values": [], "rejected": [], "n_rejected": 0,
                "min_q": float("nan")}
    order = np.argsort(p, kind="stable")
    scaled = p[order] * m / np.arange(1, m + 1)
    q_sorted = np.clip(np.minimum.accumulate(scaled[::-1])[::-1], 0.0, 1.0)
    q = np.empty(m, dtype=float)
    q[order] = q_sorted
    rejected = q <= float(alpha)
    return {**base, "q_values": q.tolist(), "rejected": rejected.tolist(),
            "n_rejected": int(rejected.sum()), "min_q": float(q.min())}


def nested_cohort_selection(
    scores: dict, *, higher_is_better: bool = True
) -> dict:
    """Nested leave-one-cohort-out arm selection, and the optimism it removes.

    ``scores`` is ``{arm: {cohort: value}}`` — one already-aggregated number per
    (arm, held-out cohort), e.g. QWK with seeds averaged within cohort. For each
    held-out cohort *h* the winning arm is chosen on the **other** g-1 cohorts
    and then scored on *h*; ``nested_mean`` is the mean of those honest scores.
    ``flat_mean`` is the reported quantity — the mean of the arm that wins on all
    g cohorts at once — and ``optimism = flat_mean - nested_mean`` (sign-flipped
    when lower is better) is what select-then-report buys.

    This exists because the promotion rule in ``experiments/lodo_cutpoint.py``
    (mean delta >= +0.02 over *all* folds) selects and reports on the identical
    folds, so the published LODO deltas *are* the selection statistic. Measured
    on the locked corpus the optimism is +0.0000 for ECG, PPG and EDA
    (``ordlogit_percohort`` wins on every inner subset) — the point of persisting
    it is that the number is now on the record instead of assumed.
    """
    arms = sorted(scores)
    if not arms:
        raise ValueError("nested_cohort_selection: no arms")
    cohorts = sorted(scores[arms[0]])
    for a in arms:
        if sorted(scores[a]) != cohorts:
            raise ValueError(
                f"nested_cohort_selection: arm {a!r} does not cover the same cohorts "
                f"as {arms[0]!r} (nested selection needs a complete arm x cohort grid)"
            )
    if len(cohorts) < 2:
        raise ValueError("nested_cohort_selection: needs >= 2 cohorts to hold one out")

    sign = 1.0 if higher_is_better else -1.0

    def _mean(arm: str, cs: list) -> float:
        return float(np.mean([scores[arm][c] for c in cs]))

    def _best(cs: list) -> str:
        # max() keeps the FIRST maximal element and `arms` is sorted, so ties
        # resolve deterministically rather than by dict insertion order.
        return max(arms, key=lambda a: sign * _mean(a, cs))

    per_held: dict = {}
    for h in cohorts:
        inner = [c for c in cohorts if c != h]
        pick = _best(inner)
        per_held[h] = {"selected_arm": pick, "inner_mean": _mean(pick, inner),
                       "outer_score": float(scores[pick][h])}
    nested_mean = float(np.mean([per_held[h]["outer_score"] for h in cohorts]))
    flat_arm = _best(cohorts)
    flat_mean = _mean(flat_arm, cohorts)
    return {
        "arms": arms,
        "cohorts": cohorts,
        "n_cohorts": len(cohorts),
        "flat_arm": flat_arm,
        "flat_mean": flat_mean,
        "nested_mean": nested_mean,
        "optimism": float(sign * (flat_mean - nested_mean)),
        "selected_arms": sorted({v["selected_arm"] for v in per_held.values()}),
        "per_held": per_held,
        "higher_is_better": bool(higher_is_better),
    }


def _stable_unique(arr: np.ndarray) -> np.ndarray:
    """Unique values preserving first-appearance order (deterministic folds)."""
    seen: dict = {}
    for v in arr:
        key = v.item() if hasattr(v, "item") else v
        if key not in seen:
            seen[key] = True
    return np.array(list(seen.keys()), dtype=arr.dtype if arr.dtype != object else object)


def _cluster_se(groups: list[np.ndarray]) -> float:
    """Cluster-robust (sandwich) standard error of the pooled mean over ``groups``.

    For equal-sized clusters this reduces exactly to the classical
    ``sd(cluster means) / sqrt(g)``; unequal sizes get the ratio-estimator form.
    """
    g = len(groups)
    n = int(sum(len(x) for x in groups))
    if g < 2 or n == 0:
        return float("nan")
    m = float(np.concatenate(groups).mean())
    e = np.array([float(x.sum()) - len(x) * m for x in groups], dtype=float)
    return float(np.sqrt(g / (g - 1.0) * float(np.sum(e ** 2))) / n)


def _norm_ppf(p: float) -> float:
    from scipy.stats import norm

    return float(norm.ppf(p))


def _norm_cdf(z: float) -> float:
    from scipy.stats import norm

    return float(norm.cdf(z))


def cluster_bootstrap_ci_detail(
    values: Sequence[float],
    clusters: Sequence | None = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    method: str = "percentile",
) -> dict:
    """Full result of :func:`cluster_bootstrap_ci` — the tuple plus its provenance.

    Returns ``{mean, lo, hi, n, n_clusters, method, alpha, se, n_boot}``. Use this
    (rather than the 3-tuple) whenever the CI is persisted, so ``n_clusters`` — the
    quantity that decides whether the interval is trustworthy — travels with it.
    """
    if method not in ("percentile", "t", "bca"):
        raise ValueError(f"cluster_bootstrap_ci: unknown method {method!r}")
    v = np.asarray(values, dtype=float).ravel()
    cl = None
    if clusters is not None:
        cl = _as_array(clusters).ravel()
        if cl.shape[0] != v.shape[0]:
            raise ValueError("cluster_bootstrap_ci: clusters must align with values")
    finite = ~np.isnan(v)
    v = v[finite]
    if cl is not None:
        cl = cl[finite]
    n = int(v.size)
    base = {"n": n, "method": method, "alpha": float(alpha), "n_boot": int(n_boot)}
    if n == 0:
        return {**base, "mean": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n_clusters": 0, "se": float("nan")}
    mean = float(v.mean())
    if cl is None:
        groups = [v[i:i + 1] for i in range(n)]   # i.i.d. == every point its own cluster
    else:
        groups = [v[cl == u] for u in _stable_unique(cl)]
    g = len(groups)
    if n == 1:
        return {**base, "mean": mean, "lo": mean, "hi": mean, "n_clusters": g,
                "se": float("nan")}

    se = _cluster_se(groups)
    if method == "percentile" and g < MIN_CLUSTERS_FOR_PERCENTILE:
        warnings.warn(
            f"cluster_bootstrap_ci: only {g} clusters — the plain percentile "
            f"bootstrap is anti-conservative here (a nominal {100 * (1 - alpha):.0f}% "
            "interval covers ~80/83/87% at g=4/5/7). Prefer method='t'.",
            UserWarning,
            stacklevel=3,
        )

    if method == "t":
        # Studentized (cluster t-percentile) interval: the small-g correction is the
        # Student-t critical value on g-1 cluster degrees of freedom, not a quantile
        # of the bootstrap distribution. No resampling needed.
        try:
            from scipy.stats import t as _t

            crit = float(_t.ppf(1.0 - alpha / 2.0, g - 1))
        except Exception:  # pragma: no cover - scipy always present here
            crit = _norm_ppf(1.0 - alpha / 2.0)
        if not np.isfinite(se):
            return {**base, "mean": mean, "lo": mean, "hi": mean, "n_clusters": g, "se": se}
        return {**base, "mean": mean, "lo": mean - crit * se, "hi": mean + crit * se,
                "n_clusters": g, "se": se}

    rng = np.random.default_rng(seed)
    boot = np.empty(int(n_boot), dtype=float)
    if cl is None:  # fast path — identical draws to the generic loop below
        for b in range(int(n_boot)):
            boot[b] = v[rng.integers(0, n, size=n)].mean()
    else:
        for b in range(int(n_boot)):
            pick = rng.integers(0, g, size=g)
            boot[b] = np.concatenate([groups[i] for i in pick]).mean()

    p_lo, p_hi = alpha / 2.0, 1.0 - alpha / 2.0
    if method == "bca":
        # bias-correction z0 from the bootstrap distribution, acceleration a from a
        # leave-one-CLUSTER-out jackknife (the resampling unit must match).
        frac = float(np.mean(boot < mean))
        frac = min(max(frac, 1.0 / (2 * n_boot)), 1.0 - 1.0 / (2 * n_boot))
        z0 = _norm_ppf(frac)
        jack = np.array(
            [np.concatenate([groups[j] for j in range(g) if j != i]).mean() for i in range(g)],
            dtype=float,
        )
        d = jack.mean() - jack
        denom = 6.0 * float(np.sum(d ** 2)) ** 1.5
        a = float(np.sum(d ** 3)) / denom if denom > 0 else 0.0
        def _adj(p: float) -> float:
            z = _norm_ppf(p)
            return float(np.clip(_norm_cdf(z0 + (z0 + z) / (1.0 - a * (z0 + z))), 1e-6, 1 - 1e-6))
        p_lo, p_hi = _adj(p_lo), _adj(p_hi)

    lo = float(np.percentile(boot, 100.0 * p_lo))
    hi = float(np.percentile(boot, 100.0 * p_hi))
    return {**base, "mean": mean, "lo": lo, "hi": hi, "n_clusters": g, "se": se}


def cluster_bootstrap_ci(
    values: Sequence[float],
    clusters: Sequence | None = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    method: str = "percentile",
) -> tuple[float, float, float]:
    """Bootstrap CI of the mean of ``values`` with optional subject/cohort clustering.

    Returns ``(mean, lo, hi)``. This is the CI reported alongside every headline
    number for the paper (CMPB reviewer §6). :func:`cluster_bootstrap_ci_detail`
    returns the same interval plus ``n_clusters``/``se`` and should be preferred
    whenever the CI is persisted.

    When ``clusters`` is given (one cluster id per value — e.g. subject or held-out
    cohort), the resample is a **cluster bootstrap**: whole clusters are drawn with
    replacement and all of their values pooled before averaging. That is the correct
    unit of resampling when the raw per-(seed, fold) points are *not* independent
    (windows of one subject / one cohort are correlated), so a naive i.i.d. bootstrap
    would understate the interval. With ``clusters=None`` an ordinary i.i.d. bootstrap
    of the values is used.

    ``method`` (audit fix — the *default* is unchanged so historical intervals stay
    reproducible from this frozen module; the alternatives are *added*, never
    substituted. Publication-facing callers must pass :data:`PAPER_CI_METHOD`
    explicitly — see ``scripts/analyze_locked_results.py``):

    * ``"percentile"`` (default) — the ``alpha/2`` / ``1-alpha/2`` percentiles of the
      bootstrap distribution of the mean. **Anti-conservative at few clusters, and
      therefore not fit to publish at LODO cohort counts.** Realized coverage of a
      nominal 95 % interval (2000 replicates, cohort effects ~N(0,1), 3 seeds/cohort,
      within-cohort noise 0.2, ``n_boot=2000``; re-measured 2026-08-04):
      g=4 -> 81.3 %, g=5 -> 84.0 %, g=7 -> 87.8 %, g=20 -> 91.4 %. The paper's LODO
      folds are PPG g=4, EDA g=5, ECG g=7, i.e. squarely in that regime — a warning
      is emitted below :data:`MIN_CLUSTERS_FOR_PERCENTILE` clusters.
    * ``"t"`` — cluster t-percentile: ``mean ± t_{g-1,1-alpha/2} * SE`` with a
      cluster-robust SE (no resampling; ``n_boot`` is ignored). Realized coverage on
      the same simulation: g=4 -> 95.2 %, g=5 -> 95.0 %, g=7 -> 94.8 %,
      g=20 -> 93.7 %. **This is :data:`PAPER_CI_METHOD` and the only estimator any
      reported LODO interval may use** — the small-g deficit is a *width* problem,
      and only studentization fixes it.
    * ``"bca"`` — bias-corrected & accelerated percentile bootstrap (cluster
      jackknife acceleration). Corrects skew/bias, NOT the small-g width deficit,
      so on the (symmetric) simulation above it is no better than plain percentile:
      g=4 -> 80.7 %, g=5 -> 85.8 %, g=7 -> 86.7 %, g=20 -> 92.0 %. Offered for
      visibly skewed bootstrap distributions only; it is not the LODO fix.

    NaNs are dropped (together with their clusters). Degenerate inputs return finite
    means with ``lo == hi == mean`` (0 or 1 point) or all-NaN (empty input).
    """
    r = cluster_bootstrap_ci_detail(
        values, clusters, n_boot=n_boot, alpha=alpha, seed=seed, method=method
    )
    return (r["mean"], r["lo"], r["hi"])



def cluster_bootstrap_statistic(
    statistic,
    clusters: Sequence,
    *arrays,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Cluster-bootstrap CI of an ARBITRARY statistic, not just a mean.

    :func:`cluster_bootstrap_ci` interval-estimates the *mean* of per-unit values, and
    its ``method="t"`` fix studentizes that mean with a cluster-robust SE. Neither
    applies to a statistic that is not an average of per-window numbers — expected
    calibration error is the motivating case: ECE bins the whole sample by confidence
    and sums ``|acc - conf|`` per bin, so there is no per-window quantity to average
    and no closed-form SE. The only honest interval for it is to resample whole
    clusters, recompute the statistic from scratch on each replicate, and take
    percentiles of that distribution.

    ``statistic(*arrays)`` is called once per replicate with each array indexed by the
    resampled row set, so it sees a real sample, not a re-weighted one. ``clusters``
    is one id per row (subject, record, cohort — whatever the independent unit is).

    Returns ``{point, lo, hi, n, n_clusters, n_boot, alpha, method, n_failed}``.

    ⚠ The interval is a PERCENTILE bootstrap and therefore inherits the small-``g``
    anti-conservatism this module measures for the mean case (g=4 -> 81 %, g=5 -> 84 %,
    g=7 -> 88 % realized coverage of a nominal 95 %); ``method="t"`` is not available
    here because there is no cluster-robust SE for a general statistic. ``n_clusters``
    is returned so a reader can apply that discount, and a warning fires below
    :data:`MIN_CLUSTERS_FOR_PERCENTILE`. Replicates on which ``statistic`` raises (a
    resample can be single-class) are dropped and counted in ``n_failed`` rather than
    silently poisoning the quantiles.
    """
    cl = _as_array(clusters)
    arrs = [np.asarray(a) for a in arrays]
    if any(len(a) != len(cl) for a in arrs):
        raise ValueError("cluster_bootstrap_statistic: arrays and clusters must align")
    uniq = _stable_unique(cl)
    g = len(uniq)
    member = {u: np.flatnonzero(cl == u) for u in uniq}
    point = float(statistic(*arrs))
    if g < MIN_CLUSTERS_FOR_PERCENTILE:
        warnings.warn(
            f"cluster_bootstrap_statistic: {g} clusters is below "
            f"{MIN_CLUSTERS_FOR_PERCENTILE}; the percentile interval is anti-conservative "
            "there (see the coverage table in cluster_bootstrap_ci). Report n_clusters.",
            UserWarning, stacklevel=2,
        )
    rng = np.random.default_rng(seed)
    reps, failed = [], 0
    for _ in range(int(n_boot)):
        pick = rng.integers(0, g, g)
        idx = np.concatenate([member[uniq[i]] for i in pick])
        try:
            reps.append(float(statistic(*[a[idx] for a in arrs])))
        except Exception:  # noqa: BLE001 -- a resample can be degenerate; that is data, not a bug
            failed += 1
    reps = np.asarray([r for r in reps if np.isfinite(r)], dtype=float)
    lo, hi = (float(np.quantile(reps, alpha / 2)), float(np.quantile(reps, 1 - alpha / 2)))         if reps.size else (float("nan"), float("nan"))
    return {"point": point, "lo": lo, "hi": hi, "n": int(len(cl)), "n_clusters": int(g),
            "n_boot": int(n_boot), "alpha": float(alpha), "method": "percentile",
            "n_failed": int(failed + (int(n_boot) - failed - reps.size))}

def dump_raw_points(path, rows, *, append: bool = False) -> str:
    """Persist raw per-(seed, fold/cohort/split) metric ``rows`` to a JSON list.

    ``rows`` is an iterable of flat JSON-able dicts (one per experiment point, e.g.
    ``{"seed": 0, "held": "cinc", "qwk": 0.42, ...}``). These are the atoms a later
    pass feeds to :func:`cluster_bootstrap_ci` (CIs) and :func:`paired_wilcoxon`
    (paired tests) *without re-running training*.

    By default the file at ``path`` is (over)written with exactly ``rows`` so
    re-running an experiment is idempotent. With ``append=True`` the rows are added
    to any existing JSON list at ``path`` (incremental writing across calls). Parent
    directories are created as needed. Returns the written path as a string.
    """
    import json
    from pathlib import Path

    rows = [dict(r) for r in rows]
    p = Path(path)
    if append and p.exists():
        try:
            prev = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            prev = []
        if isinstance(prev, list):
            rows = prev + rows
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return str(p)
