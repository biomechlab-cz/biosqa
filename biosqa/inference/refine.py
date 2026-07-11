"""Boundary refinement: localize a poor segment to the ACTUAL artifact.

The models have a fixed receptive field (ECG/PPG 10 s, EEG 5 s, EDA 60 s), so a short burst poisons
the whole window — and, worse, *more overlap makes the smear wider* (every overlapping window that
touches the burst is graded poor, so a 1 s burst can span ~2× the window). Overlap therefore can't
localize; a finer analysis must.

This module keeps the model's GRADE but refines the EXTENT: within each poor run it computes a cheap
per-bin "badness" (robust short-time-energy spikes, flatline dropout, clipping) and trims the poor
segment to the bad core, handing the clean flanks back to the adjacent good grade. It only ever
SHRINKS poor regions, and if it can't localize a bad core it leaves the segment untouched (trust the
model) — so it never invents cleanliness the model didn't see the *level* of.
"""
from __future__ import annotations

import numpy as np

from biosqa.inference.segmenter import QualityInterval

_POOR = ("Q0", "Q1")
_GOOD = ("Q2", "Q3")

#: fine-analysis bin per modality (localization resolution). EDA is slow → coarser bins.
_BIN_SEC = {"ecg": 1.0, "ppg": 1.0, "eeg": 1.0, "eda": 5.0}


def fine_badness(signal, fs: float, bin_sec: float, amp_k: float = 3.0):
    """Per-bin boolean 'this ``bin_sec`` slice looks artefactual' over the whole signal.

    Keys on **amplitude excursion** — a real burst (motion / EMG / electrode pop) exceeds the
    signal's normal peak, whereas benign beat-to-beat energy wobble does not; that separates the
    two far better than an energy z-score (whose tiny MAD makes normal peaks blow up). A bin is bad
    when its peak deviation exceeds ``amp_k`` × the robust normal peak (99th pct of |x−median|), OR
    it flatlines (dropout), OR it hard-clips. Returns ``(bad[bool array], bin_samples)``.
    """
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    fs = float(fs) or 1.0
    bs = max(1, int(round(bin_sec * fs)))
    nb = x.shape[0] // bs
    if nb < 1:
        return np.zeros(0, dtype=bool), bs
    seg = x[:nb * bs].reshape(nb, bs)
    dev = np.abs(seg - np.median(x))
    scale = float(np.percentile(np.abs(x - np.median(x)), 99.0)) or 1e-9   # ~ normal peak (e.g. QRS)
    bin_peak = dev.max(axis=1)
    bin_var = seg.var(axis=1)
    med_var = float(np.median(bin_var)) + 1e-12
    xmax = float(np.max(np.abs(x))) or 1e-9
    clip_frac = (np.abs(seg) > 0.98 * xmax).mean(axis=1)
    bad = (bin_peak > amp_k * scale) | (bin_var < 1e-3 * med_var) | (clip_frac > 0.30)
    return bad, bs


_RANK = {"Q0": 0, "Q1": 1, "Q2": 2, "Q3": 3}


def _covering(intervals, t: float):
    """The WORST-grade interval covering time ``t`` (overlapping windows at overlap>0 resolve to the
    worst grade — conservative), or ``None``."""
    best = None
    for iv in intervals:
        if iv.start_sec <= t < iv.end_sec:
            if best is None or _RANK.get(iv.tier, 3) < _RANK.get(best.tier, 3):
                best = iv
    return best


def refine_intervals(intervals, signal, fs: float, modality: str, *,
                     amp_k: float = 3.0, margin_bins: int = 1) -> list:
    """Refine poor-segment boundaries against a fine per-bin badness score (see module docstring).

    Works on a per-bin grid (robust to the overlapping intervals RLE produces at overlap>0) and
    returns clean, non-overlapping, bin-resolution intervals: each maximal poor run is shrunk to
    its artefact core (bins the fine detector flags), with the clean flanks handed to the adjacent
    good grade. A poor run with no localizable core is left as-is (trust the model). ``signal`` is
    the model-rate signal that was scored.
    """
    intervals = list(intervals)
    if len(intervals) < 1 or not any(iv.tier in _POOR for iv in intervals):
        return intervals
    bin_sec = _BIN_SEC.get((modality or "").lower(), 1.0)
    bad, bs = fine_badness(signal, fs, bin_sec, amp_k)
    if bad.size == 0:
        return intervals
    nb = len(bad)

    # per-bin grade + metadata from the covering (worst) interval
    tiers = ["Q2"] * nb
    conf = np.full(nb, 0.6)
    arts: list[tuple] = [()] * nb
    rec = [False] * nb
    rtier = [""] * nb
    unc = np.zeros(nb)
    ru = [False] * nb
    hrb = np.zeros(nb)
    cset: list[tuple] = [()] * nb
    last_covered = -1
    for b in range(nb):
        iv = _covering(intervals, (b + 0.5) * bin_sec)
        if iv is not None:
            last_covered = b
            tiers[b], conf[b], arts[b] = iv.tier, iv.confidence, tuple(iv.artifacts)
            rec[b], rtier[b] = iv.recoverable, iv.recovered_tier
            unc[b] = getattr(iv, "uncertainty", 0.0)
            ru[b], hrb[b] = getattr(iv, "rate_usable", False), getattr(iv, "hr_bpm", 0.0)
            cset[b] = tuple(getattr(iv, "conformal_set", ()))

    # `fine_badness` bins the WHOLE model-rate signal, but the RLE intervals stop at the last full window
    # (`make_windows` drops the trailing partial), so bins past the analyzed span kept the default "Q2" and
    # would be RLE'd into a fabricated 'acceptable' segment over signal the model never scored. Truncate the
    # per-bin grid to what was actually analyzed; the erosion/RLE below only ever reads bins in [0, nb).
    nb = last_covered + 1
    if nb < 1:
        return intervals

    # erode each maximal poor run to its bad core; hand the clean flanks to the nearest good grade
    b = 0
    while b < nb:
        if tiers[b] not in _POOR:
            b += 1
            continue
        j = b
        while j < nb and tiers[j] in _POOR:
            j += 1
        bad_idx = [k for k in range(b, j) if bad[k]]
        if bad_idx:
            c0 = max(b, min(bad_idx) - margin_bins)
            c1 = min(j - 1, max(bad_idx) + margin_bins)
            left_good = tiers[b - 1] if b > 0 and tiers[b - 1] in _GOOD else "Q2"
            right_good = tiers[j] if j < nb and tiers[j] in _GOOD else "Q2"
            for k in range(b, c0):           # clean leading flank (now good morphology)
                tiers[k], conf[k], rec[k], rtier[k], ru[k] = left_good, 0.6, False, "", False
                arts[k], unc[k], cset[k] = (), 0.0, ()  # artifact/uncertainty/set belong to the core
            for k in range(c1 + 1, j):       # clean trailing flank
                tiers[k], conf[k], rec[k], rtier[k], ru[k] = right_good, 0.6, False, "", False
                arts[k], unc[k], cset[k] = (), 0.0, ()
        b = j

    # RLE the refined per-bin grade back into non-overlapping intervals
    out: list[QualityInterval] = []
    b = 0
    while b < nb:
        j = b
        while j < nb and tiers[j] == tiers[b]:
            j += 1
        union: list[str] = []
        for k in range(b, j):
            for a in arts[k]:
                if a not in union:
                    union.append(a)
        rk = next((k for k in range(b, j) if rec[k]), None)
        ruk = [k for k in range(b, j) if ru[k]]
        rate_usable = len(ruk) >= (j - b) / 2.0 and len(ruk) > 0
        hrs = [hrb[k] for k in ruk if hrb[k] > 0]
        cs = next((cset[k] for k in range(b, j) if cset[k]), ())
        out.append(QualityInterval(
            start_sec=b * bin_sec, end_sec=j * bin_sec, tier=tiers[b],
            confidence=float(np.mean(conf[b:j])), artifacts=tuple(union),
            recoverable=bool(rk is not None), recovered_tier=(rtier[rk] if rk is not None else ""),
            uncertainty=float(np.mean(unc[b:j])),
            rate_usable=bool(rate_usable), hr_bpm=float(np.median(hrs)) if hrs else 0.0,
            conformal_set=cs,
        ))
        b = j
    return out
