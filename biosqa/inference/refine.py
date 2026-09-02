"""Boundary refinement: tighten a poor segment towards the ACTUAL artifact — without ever showing a
grade better than the model's own.

The models have a fixed receptive field (ECG/PPG 10 s, EEG 5 s, EDA 60 s), so a short burst poisons the
whole window — and, worse, *more overlap makes the smear wider*: every overlapping window that touches
the burst is graded poor, so a 1 s burst can smear over ~2x the window.

Exactly ONE rule bounds what this module may do: **the model's opinion about a time is the set of grades
it gave the WINDOWS covering that time, and refinement may never assert better than the BEST of those.**

Refinement decides a grade per BIN, and a bin is a SPAN, so every question it asks must be asked over
the whole span, never at one instant of it. A bin may be promoted only to the best grade among the
windows that COVER THE WHOLE BIN (a window that merely starts or ends inside the bin said nothing about
the rest of it), and it is otherwise shown the WORST of the displayed grades it intersects. Sampling the
bin CENTRE instead is precisely how this module once displayed *excellent* over a half-bin the model had
only ever graded *poor*: at the shipped 50% overlap every other window start lands exactly on a bin
centre, so the centre sample saw a good window that began there and painted its grade over the bin's
left half, which that window does not cover at all.

That set is why refinement reads the PER-WINDOW model output (``model_windows``, from
:func:`segmenter.window_intervals`) and NOT the RLE intervals it is refining. ``run_length_encode``
deliberately collapses the set away: it hands a multiply-covered time to exactly one run (boundary at the
midpoint of the ambiguous zone) and its output is therefore strictly NON-OVERLAPPING — every time has
exactly one covering interval, so a ceiling computed from it can never exceed what is already displayed
and refinement would be an unconditional no-op. The overlap information lives in the windows; read it
there. Concretely:

* Where windows OVERLAP (the app default is 50%), a bin can sit inside both a poor window and a good
  one; RLE displayed the poor grade. If the fine per-bin detector (amplitude excursion / dropout /
  clipping) finds no artifact in that bin, refinement relaxes it to the good grade the model ITSELF
  produced for a window covering it, carrying THAT window's confidence/uncertainty/artifacts, and
  records the conservative grade in ``model_tier`` so the change is auditable through the export.
* Where windows do NOT overlap, a bin has exactly one covering window: the model made exactly ONE
  statement about that time, so there is nothing to refine and nothing may be promoted. Refinement is
  CORRECTLY a no-op there. That is the honest answer, not a bug — do not "fix" it by handing the clean
  flank of a poor window to the neighbouring good window. The model never graded that flank, and a
  fabricated clean grade in an SQA tool is worse than an honest smear.
  CAREFUL: "overlap = 0" does NOT mean "no windows overlap". ``preprocess.make_windows`` END-ANCHORS the
  final window (start = n - L_m) so the trailing partial window is graded rather than silently dropped,
  so whenever the record is not a whole number of windows the LAST TWO windows genuinely overlap even at
  stride == L_m. In that tail zone refinement can and does promote — honestly, because a real window
  really did grade it. Refinement is a strict no-op at overlap 0 only for an exact-multiple record.
* A poor run with no localizable core, and a bin no window covers, are left alone (trust the model /
  show nothing) — never defaulted to a plausible grade or confidence.

So the artifact core can shrink to the fine grid, but a poor region can never shrink below the span of
the windows the model unambiguously graded poor. That is the model's true resolution.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from biosqa.inference.segmenter import QualityInterval

_POOR = ("Q0", "Q1")

#: fine-analysis bin per modality (localization resolution). EDA is slow → coarser bins.
_BIN_SEC = {"ecg": 1.0, "ppg": 1.0, "eeg": 1.0, "eda": 5.0}


def fine_badness(signal, fs: float, bin_sec: float, amp_k: float = 3.0):
    """Per-bin boolean 'this ``bin_sec`` slice looks artefactual' over the whole signal.

    Keys on **amplitude excursion** — a real burst (motion / EMG / electrode pop) exceeds the
    signal's normal peak, whereas benign beat-to-beat energy wobble does not; that separates the
    two far better than an energy z-score (whose tiny MAD makes normal peaks blow up). A bin is bad
    when its peak deviation exceeds ``amp_k`` × the robust normal peak (99th pct of |x−median|), OR
    it flatlines (dropout), OR it hard-clips. Returns ``(bad[bool array], bin_samples)``.

    This is a LOCALIZER, not a grader: it says where an artifact sits, never how good the signal is.
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
_EPS = 1e-9


def _worst_over(intervals, lo: float, hi: float):
    """The WORST-grade displayed interval that INTERSECTS the bin ``[lo, hi)``, or ``None``.

    A bin is a SPAN, and the grade decided for it is painted over the WHOLE span, so it must be
    decided over the whole span. Sampling one instant (e.g. the bin centre) instead lets a run
    boundary that falls INSIDE the bin hand the entire bin the grade of just one of the two runs it
    straddles; RLE boundaries sit at the midpoint of an ambiguous zone and land wherever the window
    grid puts them, so they routinely fall inside a bin. Taking the worst of everything the bin
    touches keeps the bin no better than the worst part of it actually is.
    """
    best = None
    for iv in intervals:
        if iv.start_sec < hi - _EPS and iv.end_sec > lo + _EPS:
            if best is None or _RANK.get(iv.tier, 3) < _RANK.get(best.tier, 3):
                best = iv
    return best


def _ceiling_over(model_windows, lo: float, hi: float):
    """The BEST-grade MODEL WINDOW that COVERS THE WHOLE bin ``[lo, hi)``: the hard ceiling on what
    refinement may display over that bin. ``None`` when NO single window spans the bin, and then the
    bin may not be promoted at all.

    Covering the *whole* bin is the entire point. A window that covers only PART of the bin (one that
    starts or ends inside it) made no statement about the rest of the bin, so its grade may not be
    painted over the rest: the model never ran over that part. Sampling the ceiling at an instant
    (e.g. the bin centre) does exactly that, and it is how a bin whose left half was covered ONLY by
    poor windows came to be displayed excellent: the centre sample saw a good window that STARTED at
    the centre. At the shipped 50% overlap every other window start lands exactly on a bin centre, so
    that is not a corner case; it is the common case.
    """
    best = None
    for iv in model_windows:
        if iv.start_sec <= lo + _EPS and iv.end_sec >= hi - _EPS:
            if best is None or _RANK.get(iv.tier, 0) > _RANK.get(best.tier, 0):
                best = iv
    return best


def refine_intervals(intervals, signal, fs: float, modality: str, *,
                     model_windows=None, amp_k: float = 3.0, margin_bins: int = 1) -> list:
    """Refine poor-segment boundaries against a fine per-bin badness score (see module docstring).

    Args:
        intervals: the RLE (displayed, non-overlapping) intervals to refine.
        signal: the model-rate signal that was scored.
        model_windows: the model's PER-WINDOW grades as overlapping intervals
            (:func:`segmenter.window_intervals`) — the source of the per-bin CEILING. Without them the
            only grades on record are the RLE ones, which cover each time exactly once, so the ceiling
            equals the displayed grade everywhere and refinement is a no-op (it returns ``intervals``
            untouched) — never a promotion the model did not license.

    Within each maximal poor run the artefact core is located, and a clean flank bin is relaxed ONLY to
    the best grade the model itself gave a WINDOW covering that bin — with that window's own confidence,
    and the conservative grade preserved in ``model_tier``.
    """
    intervals = list(intervals)
    if len(intervals) < 1 or not any(iv.tier in _POOR for iv in intervals):
        return intervals
    ceilings = list(model_windows) if model_windows is not None else intervals
    bin_sec = _BIN_SEC.get((modality or "").lower(), 1.0)
    bad, bs = fine_badness(signal, fs, bin_sec, amp_k)
    if bad.size == 0:
        return intervals
    nb = len(bad)

    # Per-bin grade + metadata from the covering (displayed) interval, plus the per-bin CEILING window
    # (the best grade the model gave any WINDOW over that bin). A bin NO interval covers stays unknown
    # (tier None, NaN confidence) and is dropped from the output — never defaulted to a plausible
    # "Q2"/0.6: `fine_badness` bins the WHOLE model-rate signal, but the intervals may not span all of
    # it (a caller may pass a partial list, or a gap), so those bins are signal the model never scored.
    tiers: list[str | None] = [None] * nb
    conf = np.full(nb, np.nan)
    arts: list[tuple] = [()] * nb
    rec = [False] * nb
    rtier = [""] * nb
    unc = np.zeros(nb)
    ru = [False] * nb
    hrb = np.zeros(nb)
    cset: list[tuple] = [()] * nb
    mtier = [""] * nb            # the model's conservative grade, set only where we relax the display
    ceil_iv: list = [None] * nb
    for b in range(nb):
        lo, hi = b * bin_sec, (b + 1) * bin_sec       # the bin is a SPAN -- sample it as one
        iv = _worst_over(intervals, lo, hi)
        if iv is None:
            continue
        tiers[b], conf[b], arts[b] = iv.tier, iv.confidence, tuple(iv.artifacts)
        rec[b], rtier[b] = iv.recoverable, iv.recovered_tier
        unc[b] = getattr(iv, "uncertainty", 0.0)
        ru[b], hrb[b] = getattr(iv, "rate_usable", False), getattr(iv, "hr_bpm", 0.0)
        cset[b] = tuple(getattr(iv, "conformal_set", ()))
        ceil_iv[b] = _ceiling_over(ceilings, lo, hi)

    def _relax(k: int) -> bool:
        """Show bin ``k`` the BEST grade the model gave a WINDOW spanning the WHOLE bin, keeping that
        window's own confidence/uncertainty/artifacts and preserving the conservative grade in
        ``model_tier``. False when no window spans the bin, or every window that does graded it no
        better: then the bin keeps what the model said."""
        top = ceil_iv[k]
        if top is None or _RANK.get(top.tier, 0) <= _RANK.get(tiers[k], 0):
            return False
        mtier[k] = tiers[k]
        tiers[k], conf[k], arts[k] = top.tier, top.confidence, tuple(top.artifacts)
        rec[k], rtier[k] = top.recoverable, top.recovered_tier
        unc[k] = getattr(top, "uncertainty", 0.0)
        ru[k], hrb[k] = getattr(top, "rate_usable", False), getattr(top, "hr_bpm", 0.0)
        cset[k] = tuple(getattr(top, "conformal_set", ()))
        return True

    # Locate each maximal poor run's artefact core, then erode the run's two BOUNDARIES inward over the
    # clean flanks (all bad bins are in the core). Eroding inward — rather than relaxing every relaxable
    # flank bin — stops at the first bin the model has no better grade for, so refinement can only ever
    # move a boundary; it can never punch a clean island into the middle of a poor run.
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
            for k in range(b, c0):                  # leading flank, outward -> inward
                if not _relax(k):
                    break
            for k in range(j - 1, c1, -1):          # trailing flank, outward -> inward
                if not _relax(k):
                    break
        b = j

    if not any(mtier):                    # nothing could be relaxed within the model's own grades
        return intervals

    # RLE the refined per-bin grade back into non-overlapping intervals. The run key carries model_tier
    # too, so a relaxed span never merges into a genuinely-good neighbour and loses its provenance.
    out: list[QualityInterval] = []
    b = 0
    while b < nb:
        if tiers[b] is None:              # signal no window covered — emit nothing, invent nothing
            b += 1
            continue
        j = b
        while j < nb and tiers[j] == tiers[b] and mtier[j] == mtier[b]:
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
            model_tier=mtier[b],
            conformal_set=cs,
        ))
        b = j

    # The bin grid stops at the last WHOLE bin, but the model's analyzed span runs to the end of the
    # end-anchored final window — up to one bin (1 s; 5 s for EDA) of graded signal lies past
    # `nb * bin_sec`. Re-emit it exactly as the segmenter graded it rather than letting refinement
    # silently truncate the record: the residual was never binned, so it is never relaxed. Every
    # displayed interval overlapping the residual is CLIPPED into it: the residual is a span, and a
    # run boundary can fall inside it (it is a whole 5 s for EDA), so picking one interval by sampling
    # an instant of it would paint that instant's grade over the rest of the residual.
    grid_end = nb * bin_sec
    analyzed_end = max(iv.end_sec for iv in intervals)
    if analyzed_end > grid_end + _EPS:
        for iv in sorted(intervals, key=lambda x: x.start_sec):
            lo = max(float(iv.start_sec), grid_end)
            hi = min(float(iv.end_sec), float(analyzed_end))
            if hi <= lo + _EPS:
                continue
            if (out and out[-1].tier == iv.tier and not out[-1].model_tier
                    and abs(out[-1].end_sec - lo) < _EPS):
                out[-1] = replace(out[-1], end_sec=hi)      # same grade, no provenance to lose -> extend
            else:
                out.append(replace(iv, start_sec=lo, end_sec=hi, model_tier=""))
    return out
