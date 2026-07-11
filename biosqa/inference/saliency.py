"""Gradient-free occlusion SALIENCY over a 1D biosignal window (explainable-AI: "which part of the signal
is the model looking at?"). The deployed models run under ONNX Runtime with NO autograd, so gradient methods
(Grad-CAM, integrated gradients, LRP) are unavailable — but occlusion/perturbation saliency needs only
FORWARD passes: slide a smooth occluder across the window and measure how much each region changes the grade.

Faithfulness (validated): on an ECG with a localized noise burst importance concentrates ~100x inside the
burst (raw ΔP≈0.86) with its peak there, while a confidently-graded CLEAN window renders faint (raw ΔP small,
absolute-scaled so it isn't amplified into noise). Two design points matter: (1) the occluder is a smooth
ramp between the LOCAL BASELINES OUTSIDE the patch — it removes the artifact's energy, unlike a ramp between
the patch's own (noisy) endpoints; (2) the patch is artifact-SCALE (~0.5 s / 12% of the window) — occluding
a piece far smaller than the corruption leaves a redundant grade unchanged (≈0 ΔP). Importance is the change
in the predicted-class / P(unusable) probability. Pure numpy + forward passes; run on demand per segment.
"""
from __future__ import annotations

import numpy as np

__all__ = ["occlusion_saliency", "signal_saliency"]

# ABSOLUTE intensity scale (probability-change units): a raw |ΔP| of ``_SAL_FULL`` paints full intensity;
# below ``_SAL_FLOOR`` is treated as insensitive (0). This makes the heatmap reflect the TRUE grade-
# sensitivity — a confidently-graded window the model barely responds to renders FAINT rather than
# amplified noise — and makes intensities comparable across windows of a multi-window segment.
_SAL_FULL = 0.25
_SAL_FLOOR = 0.02


def _predict_grade(sig: np.ndarray, runner) -> np.ndarray:
    from biosqa.inference.preprocess import normalize_window
    b = normalize_window(sig.astype(np.float32), runner.card.normalization)[None, :]
    return runner.predict_windows_multihead(b).primary[0]


def occlusion_saliency(window: np.ndarray, runner, *, patch_frac: float = 0.12, target: str = "unusable",
                       normalize: bool = True) -> np.ndarray:
    """Per-sample importance for ONE model-length window. ``patch_frac``-sized overlapping occluder patches
    (floored at 0.5 s); ``target`` = ``"pred"`` (predicted-class prob — what supports THIS grade) or
    ``"unusable"`` (P(Q0)+P(Q1) = 1−(Q2+Q3) — what makes it look corrupt). ``normalize`` applies the ABSOLUTE intensity
    scale (``_SAL_FULL``, with a floor) so an insensitive window renders faint not amplified-noise and
    intensities compare across windows; pass False for the raw |ΔP|. Zeros on a degenerate window."""
    x = np.nan_to_num(np.asarray(window, dtype=np.float64).reshape(-1))
    n = x.size
    if n < 8 or float(np.std(x)) < 1e-9 or runner is None or runner.card is None:
        return np.zeros(n)
    base = _predict_grade(x, runner)
    cls = int(base.argmax())

    def tgt(p):
        # "unusable" = the app's real boundary P(Q0)+P(Q1) = 1 − (P(Q2)+P(Q3)) (Q2 acceptable and Q3
        # excellent are BOTH usable; see onnx_runner p_unusable) — NOT 1 − P(Q3), which is merely
        # P(not-excellent) and would flag a confidently-usable Q2 window as corrupt.
        return float(p[cls]) if target == "pred" else float(1.0 - np.sum(p[-2:]))

    base_t = tgt(base)
    fs = float(runner.card.fs_hz) or 1.0
    # Patch ~ the scale of a real artifact (>= 0.5 s, ~12% of the window): occluding a patch far SMALLER
    # than the corruption barely changes a redundant grade (a small piece of a long burst still reads
    # corrupt), so tiny patches give ~0 raw ΔP and a normalized map of pure noise.
    w = max(8, int(0.5 * fs), int(patch_frac * n))
    m = max(4, w // 4)                                    # context width for the CLEAN occluder baseline
    sal = np.zeros(n)
    for s in range(0, n, max(1, w // 2)):                 # 50%-overlap patches
        e = min(n, s + w)
        # occlude with a smooth ramp between the LOCAL BASELINES OUTSIDE the patch (means of neighbouring
        # samples) — this removes the local morphology/artifact cleanly, instead of a linspace between the
        # patch's own (possibly noisy) endpoints which would leave the artifact's energy in place.
        a = float(x[max(0, s - m):s].mean()) if s > 0 else float(x[s])
        z = float(x[e:min(n, e + m)].mean()) if e < n else float(x[e - 1])
        occ = x.copy()
        occ[s:e] = np.linspace(a, z, e - s)
        d = abs(base_t - tgt(_predict_grade(occ, runner)))
        sal[s:e] = np.maximum(sal[s:e], d)                # max over overlapping patches
    if not normalize:
        return sal                                        # raw |ΔP| magnitude
    out = np.clip(sal / _SAL_FULL, 0.0, 1.0)              # ABSOLUTE scale — reflects true sensitivity
    out[sal < _SAL_FLOOR] = 0.0                           # sub-threshold jitter → 0 (no spurious hotspots)
    return out


def signal_saliency(signal: np.ndarray, runner, *, patch_frac: float = 0.12, max_windows: int = 8,
                    target: str = "unusable") -> np.ndarray:
    """Occlusion saliency over a selected SEGMENT by tiling model-length windows. Each window is
    ABSOLUTE-scaled (not per-window max), so intensities are comparable across the segment's windows and an
    insensitive window stays faint. Capped at ``max_windows`` (centered) to stay responsive; a longer
    segment's tail beyond the cap gets 0. Returns a per-sample importance map (0..1)."""
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    n = x.size
    if runner is None or runner.card is None:
        return np.zeros(n)
    lm = int(runner.card.l_m)
    if n <= lm:
        pad = np.concatenate([x, np.full(lm - n, x[-1] if n else 0.0)]) if n < lm else x
        return occlusion_saliency(pad, runner, patch_frac=patch_frac, target=target)[:n]
    starts = list(range(0, n - lm + 1, lm))
    if len(starts) > max_windows:                          # keep the CENTRED windows
        off = (len(starts) - max_windows) // 2
        starts = starts[off:off + max_windows]
    sal = np.zeros(n)
    for s in starts:
        sal[s:s + lm] = occlusion_saliency(x[s:s + lm], runner, patch_frac=patch_frac, target=target)
    return sal
