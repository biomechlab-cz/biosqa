"""Group-level Shapley feature attribution for the fused-input SQA models — the feature-level complement to
the spatial occlusion-saliency heatmap. Saliency answers *where* in time the model looks; this answers *which
signal-quality property* drives the grade ("this window is low-quality mainly because of Noise/HF + Spectral").

The PPG/EEG/EDA models fuse a hand-crafted SQI+dynamics vector (``combined_vector``) as their 2nd input and
their grade head reads it, so we can attribute the grade to that vector. (The ECG model uses spectral CHANNELS
and grade<-raw, so it has no such vector — ``runner.has_feature_attribution()`` gates it out.) We do NOT
attribute each of the ~20-29 raw features individually: many SQIs are correlated, and single-feature ablation
under-credits correlated features (ablating one leaves the other signalling the same corruption). Instead we
bin them into ~5-6 INTERPRETABLE GROUPS and compute EXACT Shapley values over the groups by exhaustively
enumerating all 2^G coalitions (G<=6 -> <=64 forward passes) on the segment's WORST window. Group-level exact
Shapley is correlation-correct and forward-only. Background = the training-mean vector shipped in the card's
``feature_attribution`` block.

Honest caveat (surfaced in the UI): this is a PERTURBATION estimate that deliberately breaks the (raw, feat)
coupling to probe the fused input's contribution — it measures the model's reliance on each quality property,
not ground-truth causation, and it is silent about anything the grade reads from the raw waveform directly.
"""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np

from biosqa.inference.preprocess import normalize_window

__all__ = ["grade_group_attribution", "has_feature_attribution"]

# ABSOLUTE scale for the UI bar length (P(unusable)-change units): a group whose Shapley value shifts the
# grade by ``_PHI_FULL`` draws a full-length bar. This mirrors the saliency lesson — scale by ABSOLUTE
# magnitude, not the relative share, so a confidently-CLEAN window (whose groups barely move the grade)
# renders faint bars instead of a full-length bar for its largest-of-tiny contribution.
_PHI_FULL = 0.3

# Interpretable bins by feature-name membership (exact name OR prefix). Anything unmatched falls into the
# catch-all "Complexity / dynamics" group (the shared advanced pack: spectral-kurtosis, recurrence-
# quantification, ordinal-pattern, dispersion-entropy, ...). Order here = display tie-break order.
_DYNAMICS = "Complexity / dynamics"
_GROUPS: dict[str, list[tuple[str, tuple[str, ...]]]] = {
    "ppg": [
        ("Noise / HF", ("ppg_hf_ratio",)),
        ("Pulse morphology", ("ppg_cardiac_ratio", "ppg_pulse_regularity")),
        ("Amplitude / shape", ("ppg_skew", "ppg_kurt")),
        ("Spectral", ("ppg_spec_entropy",)),
    ],
    "eeg": [
        ("Line noise / HF", ("eeg_line_index", "eeg_high_ratio", "eeg_gamma_ratio")),
        ("Spectral / aperiodic", ("eeg_low_ratio", "eeg_tilt", "eeg_aperiodic_slope", "eeg_flatness")),
        ("Amplitude / gradient", ("eeg_kurt", "eeg_max_grad", "eeg_zcr")),
        ("Flatline / Hjorth", ("eeg_flatline_frac", "eeg_hjorth_mob", "eeg_hjorth_comp")),
    ],
    "eda": [
        ("Motion (wavelet)", ("eda_haar",)),                       # prefix -> eda_haar1_std ... eda_haar4_eratio
        ("Detail / derivative", ("eda_detail_approx_ratio", "eda_dmax", "eda_drms")),
    ],
}


def has_feature_attribution(runner) -> bool:
    """Whether ``grade_group_attribution`` can run for this model (delegates to the runner's gate)."""
    return runner is not None and runner.card is not None and runner.has_feature_attribution()


def _resolve_groups(modality: str, feature_names: list[str]) -> list[tuple[str, list[int]]]:
    """Bin feature indices into named groups; unmatched features go to the ``_DYNAMICS`` catch-all."""
    assigned = [False] * len(feature_names)
    out: list[tuple[str, list[int]]] = []
    for gname, members in _GROUPS.get(modality, []):
        idxs = [j for j, fn in enumerate(feature_names)
                if not assigned[j] and any(fn == mem or fn.startswith(mem) for mem in members)]
        for j in idxs:
            assigned[j] = True
        if idxs:
            out.append((gname, idxs))
    rest = [j for j in range(len(feature_names)) if not assigned[j]]
    if rest:
        out.append((_DYNAMICS, rest))
    return out


def _worst_window(signal: np.ndarray, runner) -> "tuple[np.ndarray, int, float]":
    """Tile model-length windows over the segment and return the most-degraded one (max P(unusable)),
    its start sample, and that P(unusable) — the window whose grade is most worth explaining."""
    lm = int(runner.card.l_m)
    x = np.nan_to_num(np.asarray(signal, dtype=np.float64).reshape(-1))
    if x.size < lm:
        x = np.concatenate([x, np.full(lm - x.size, x[-1] if x.size else 0.0)])
    starts = list(range(0, x.size - lm + 1, lm)) or [0]
    wins = np.stack([normalize_window(x[s:s + lm].astype(np.float32), runner.card.normalization) for s in starts])
    probs = np.asarray(runner.predict_windows_multihead(wins).primary, dtype=np.float64)  # [n, C]
    unus = 1.0 - probs[:, -2:].sum(axis=1)                 # P(unusable) = P(Q0)+P(Q1) = 1 − (Q2+Q3)
    i = int(np.argmax(unus))
    return wins[i], int(starts[i]), float(unus[i])


def _group_shapley(runner, window_norm: np.ndarray, base_feat: np.ndarray, ref: np.ndarray,
                   group_idx: list[list[int]]) -> list[float]:
    """Exact Shapley values over feature GROUPS. Value function v(S) = P(unusable) when the groups IN S keep
    their actual (base) feature values and groups NOT in S are ablated to the reference mean; v(∅) = all
    ablated (baseline), v(all) = the true prediction. Enumerates every coalition (<=2^6) with caching."""
    G = len(group_idx)
    cache: dict[frozenset, float] = {}

    def v(S: frozenset) -> float:
        if S in cache:
            return cache[S]
        f = ref.astype(np.float32).copy()
        for g in S:
            for j in group_idx[g]:
                f[j] = base_feat[j]
        p = runner.grade_probs_with_feat(window_norm, f)
        val = float(1.0 - np.sum(p[-2:]))             # P(unusable) = 1 − (Q2+Q3), matches the saliency target
        cache[S] = val
        return val

    phi = [0.0] * G
    for g in range(G):
        others = [o for o in range(G) if o != g]
        for k in range(len(others) + 1):
            w = math.factorial(k) * math.factorial(G - k - 1) / math.factorial(G)
            for combo in combinations(others, k):
                phi[g] += w * (v(frozenset(combo + (g,))) - v(frozenset(combo)))
    return phi


def grade_group_attribution(signal: np.ndarray, runner, *, top_k: int = 6) -> "dict | None":
    """Attribute the grade of a SEGMENT to interpretable feature groups via exact group-Shapley on its worst
    window. Returns ``None`` when the model has no fused SQI vector (e.g. ECG) — callers should fall back to
    the spatial heatmap alone. Otherwise returns
    ``{"groups": [{"group", "phi", "share"}...], "base_unusable": float, "reference_unusable": float,
       "window_start": int}`` with groups sorted by |phi| (φ>0 pushes toward *unusable*, φ<0 toward *usable*)
    and ``share`` = |phi| / Σ|phi| (fraction of the total explained swing)."""
    if not has_feature_attribution(runner):
        return None
    fa = runner.card.feature_attribution
    names = list(fa["feature_names"])
    ref = np.asarray(fa["reference_mean"], dtype=np.float32)
    if ref.shape[0] != len(names):
        return None
    win, start, base_unus = _worst_window(signal, runner)
    base_feat, vnames = runner.combined_feature_vector(win)
    if list(vnames) != names or base_feat.shape[0] != ref.shape[0]:
        return None                                   # card/runtime feature-set mismatch -> fail safe
    groups = _resolve_groups(runner.modality, names)
    phi = _group_shapley(runner, win, base_feat, ref, [idx for _, idx in groups])
    ref_unus = float(1.0 - np.sum(runner.grade_probs_with_feat(win, ref)[-2:]))
    total = float(sum(abs(p) for p in phi)) or 1.0
    rows = [{"group": groups[i][0], "phi": float(phi[i]), "share": float(abs(phi[i]) / total),
             "scaled": float(min(1.0, abs(phi[i]) / _PHI_FULL))}     # bar length: absolute, not relative
            for i in range(len(groups))]
    rows.sort(key=lambda d: -abs(d["phi"]))
    return {"groups": rows[:top_k], "base_unusable": base_unus, "reference_unusable": ref_unus,
            "window_start": int(start)}
