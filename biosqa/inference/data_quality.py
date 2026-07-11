"""Record-level DATA-QUALITY report — completeness / validity / stability checks (research2 idea).

Orthogonal to the per-window SIGNAL-quality model: the model rates whether a clean-looking window is a
good biosignal, but a real recording can also be broken at the *data* level — missing samples, NaNs,
sensor saturation/clipping, flat leads, or dropout gaps — which the model never sees (windows are
normalised and it assumes a valid trace). This computes the standard domain-agnostic data-quality
dimensions (completeness, validity, statistical stability) with plain numpy so the app can warn the user
BEFORE trusting per-window quality, and feed the summary to the LLM audit.

Pure numpy, microseconds; no model, no dependency.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["record_quality", "RecordQuality"]

_FLATLINE_TOL = 1e-6      # |diff| below this = flat
_CLIP_FRAC = 0.999        # sample within this fraction of the channel's min/max range = saturated
_MIN_GAP_S = 0.5          # a zero/NaN run at least this long counts as a dropout gap


@dataclass
class RecordQuality:
    n_samples: int
    duration_s: float
    missing_frac: float          # NaN / inf fraction
    flatline_frac: float         # fraction of samples on a flat run (longest-run aware)
    clipping_frac: float         # fraction saturated at the channel rail
    n_dropout_gaps: int          # count of zero/NaN runs >= _MIN_GAP_S
    longest_gap_s: float
    completeness: float          # 1 - (missing + gap) coverage, 0..1
    flags: list = field(default_factory=list)

    @property
    def usable(self) -> bool:
        """Coarse data-level gate: enough valid, non-flat, non-clipped signal to bother scoring."""
        return self.completeness >= 0.9 and self.flatline_frac < 0.5 and self.clipping_frac < 0.5


def _channel_flags(x: np.ndarray, fs: float):
    n = len(x)
    finite = np.isfinite(x)
    missing = 1.0 - finite.mean()
    xf = np.where(finite, x, 0.0)
    dz = np.abs(np.diff(xf))
    flat = np.concatenate([[False], dz < _FLATLINE_TOL])
    flat_frac = float(flat.mean())
    lo, hi = np.nanmin(x), np.nanmax(x)
    rng = (hi - lo) + 1e-12
    clip = (x >= lo + _CLIP_FRAC * rng) | (x <= hi - _CLIP_FRAC * rng)
    clip_frac = float(np.mean(clip & finite))
    # dropout gaps = SUSTAINED runs (>= min_run) of exact-zero OR non-finite samples. Count the
    # TOTAL dropout duration (not just the longest run — many separate gaps must all reduce
    # completeness) plus any isolated non-finite sample.
    bad = (~finite) | (xf == 0.0)
    min_run = max(1, int(_MIN_GAP_S * fs))
    sustained = np.zeros(n, dtype=bool)
    gaps, longest, run, run_start = 0, 0, 0, 0
    for i, b in enumerate(bad):
        if b:
            if run == 0:
                run_start = i
            run += 1
        else:
            if run >= min_run:
                gaps += 1
                sustained[run_start:i] = True
            longest = max(longest, run)
            run = 0
    if run >= min_run:
        gaps += 1
        sustained[run_start:n] = True
    longest = max(longest, run)
    gap_frac = float((sustained | ~finite).mean())     # total unusable fraction (all gaps + NaN)
    return missing, flat_frac, clip_frac, gaps, longest / fs, gap_frac


def record_quality(signal: np.ndarray, fs: float) -> RecordQuality:
    """Data-quality report for a full recording. ``signal`` is ``[C, L]`` or ``[L]`` (channels reduced by
    worst-case, since one broken lead is a quality problem)."""
    x = np.asarray(signal, dtype=np.float64)
    x = x[None, :] if x.ndim == 1 else x
    n = x.shape[-1]
    per = [_channel_flags(x[c], fs) for c in range(x.shape[0])]
    missing = max(p[0] for p in per)
    flat = max(p[1] for p in per)
    clip = max(p[2] for p in per)
    gaps = max(p[3] for p in per)
    longest = max(p[4] for p in per)
    gap_frac = max(p[5] for p in per)
    completeness = float(max(0.0, 1.0 - gap_frac))     # total dropout+NaN fraction, not just the longest gap
    flags = []
    if missing > 0.01:
        flags.append(f"{missing:.0%} missing/NaN samples")
    if flat > 0.2:
        flags.append(f"{flat:.0%} flatline (dead/disconnected sensor?)")
    if clip > 0.05:
        flags.append(f"{clip:.0%} clipped/saturated (gain too high?)")
    if gaps > 0:
        flags.append(f"{gaps} dropout gap(s), longest {longest:.1f}s")
    return RecordQuality(n_samples=n, duration_s=n / fs, missing_frac=missing, flatline_frac=flat,
                         clipping_frac=clip, n_dropout_gaps=gaps, longest_gap_s=longest,
                         completeness=completeness, flags=flags)
