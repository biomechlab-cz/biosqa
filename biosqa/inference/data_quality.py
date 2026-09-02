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

_FLAT_WIN_S = 0.5         # local-activity window: "does the trace move at all over half a second?"
_FLAT_MIN_WIN_N = 64      # ... but NEVER fewer than this many samples (see _flat_mask). A seconds-only
                          # horizon is not bandwidth-aware: at EDA's 8 Hz model rate it is a 4-SAMPLE
                          # rolling std, which measures quantisation noise rather than motion, and a
                          # healthy sub-0.5 Hz tonic trace fails it for most of its length (measured on
                          # 10 real EDABE records: flatline_frac 0.42-0.82, 9 of 10 flagged
                          # "dead/disconnected sensor?" and 8 of 10 usable=False -- on data that is fine).
                          # A sampling rate is chosen for its modality's bandwidth, so a fixed SAMPLE
                          # count is the bandwidth-relative horizon a fixed second count only looks like.
                          # 64 samples is 0.26 s at ECG/EEG rates (below the 0.5 s floor -> no change),
                          # 1 s at PPG's 64 Hz and 8 s at EDA's 8 Hz.
_FLAT_REL = 1e-3          # local excursion below this fraction of the channel's OWN range = not moving
_FLAT_MIN_RUN_S = 1.0     # ... and it must last this long: a slow trace is briefly still at a turning point
_DEAD_MOBILITY = 1.3      # std(diff)/std: >= this = white noise only, no band-limited biosignal.
                          # This is the LOCAL gate, judged on a rolling _DEAD_WIN_S estimate, so it is a
                          # noisy statistic and must stay permissive. It can afford to: _dead_stretch_mask
                          # ANDs it with two further conditions (anomalously still for this channel, and
                          # still in absolute terms), which are what keep a white-but-LIVE burst (EMG,
                          # motion) out.
_DEAD_MOBILITY_WHOLE = 1.38   # ... and this is the WHOLE-CHANNEL gate, which must sit tighter.
                          # _is_dead_channel averages over the entire record, so the estimate is
                          # low-variance and can be held close to the sqrt(2) ~ 1.414 of pure white noise
                          # -- and it MUST be, because it is an all-or-nothing verdict ("100% flatline,
                          # dead/disconnected sensor?") guarded only by the CV stationarity test. The
                          # margin above real signal is thin, not wide: a genuinely noisy but LIVE trace
                          # (10 uV alpha under 20 uV of white sensor noise) measures 1.33 and ECG under
                          # heavy EMG measures 1.25. At the local gate's 1.30 the noisy EEG was condemned
                          # end-to-end -- a confident lie about a connected electrode.
_DEAD_CV = 0.3            # ... and that noise is stationary (not an artifact burst on a live channel)
_DEAD_WIN_S = 2.0         # window the PER-STRETCH shape test is judged over (see _dead_stretch_mask)
_DEAD_ACT_PCT = 95.0      # "what this channel's activity looks like when it is ALIVE" = this percentile
_DEAD_ACT_REL = 0.15      # a dead stretch is this still relative to that -- anomalously still FOR IT
_DEAD_RANGE_REL = 0.15    # ... and never more than this fraction of the channel's range (not a burst)
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


def _rolling_std(y: np.ndarray, w: int) -> np.ndarray:
    """Std of ``y`` over a centred ``w``-sample window, O(n) via cumulative sums (edges use the clipped
    window). ``y`` must already be centred so the sum-of-squares stays well conditioned."""
    n = y.size
    w = int(min(max(w, 1), n))
    c1 = np.concatenate(([0.0], np.cumsum(y)))
    c2 = np.concatenate(([0.0], np.cumsum(y * y)))
    lo = np.clip(np.arange(n) - w // 2, 0, n)      # i - w//2 <= n-1 always, so every window has >= 1 sample
    hi = np.clip(lo + w, 0, n)
    cnt = (hi - lo).astype(np.float64)
    mean = (c1[hi] - c1[lo]) / cnt
    var = np.maximum((c2[hi] - c2[lo]) / cnt - mean * mean, 0.0)
    return np.sqrt(var)


def _sustained(mask: np.ndarray, min_run: int) -> np.ndarray:
    """Keep only the runs of ``mask`` that last at least ``min_run`` samples."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    out = np.zeros_like(mask)
    for s, e in zip(edges[0::2], edges[1::2]):
        if e - s >= min_run:
            out[s:e] = True
    return out


def _rolling_mobility(xc: np.ndarray, w: int) -> np.ndarray:
    """Hjorth mobility std(diff x)/std(x) evaluated LOCALLY, over a centred ``w``-sample window.

    This is the same scale-free shape statistic :func:`_is_dead_channel` uses on the whole channel,
    just measured per window, so a dead STRETCH is judged by the criterion that already catches a dead
    CHANNEL. Windows where the trace is exactly constant give 0 (no structure to measure); they are a
    flatline by amplitude anyway and :func:`_flat_mask` catches them by the strict test."""
    sdx = _rolling_std(xc, w)
    d = np.diff(xc, prepend=xc[0])
    sdd = _rolling_std(d - float(np.mean(d)), w)
    return np.divide(sdd, sdx, out=np.zeros_like(sdx), where=sdx > 0.0)


def _dead_stretch_mask(xc: np.ndarray, act: np.ndarray, scale: float, fs: float) -> np.ndarray:
    """A stretch that carries no signal, even though the rest of the channel does: a PARTIALLY dead
    sensor (an electrode that falls off mid-recording, a lead that goes open).

    Amplitude alone provably cannot find this. A floating electrode dithers at the AMPLIFIER NOISE
    FLOOR (~1 uV RMS), which is a few percent of a healthy channel's dynamic range -- ~30x above the
    0.1%-of-range gate the strict test in :func:`_flat_mask` uses, so that test reports 0.000 and a
    half-dead sensor passes as perfect. Widening the gate is not the fix: real signal is genuinely
    quiet at times, and a gate loose enough to see a noise floor also swallows quiet real signal.

    So ask what actually makes a stretch dead -- it has no STRUCTURE -- and require three things, all
    ratios, so the verdict is identical in volts and in microvolts:
      * structureless: local mobility >= _DEAD_MOBILITY. White noise reaches sqrt(2) ~ 1.41; any real
        biosignal is band-limited well below Nyquist and sits far under 1 (real ECG/PPG/EDA/EEG
        measure 0.25-0.55). Judged over _DEAD_WIN_S, long enough that a slow trace has room to show
        its structure -- a 0.1 Hz wave is nearly still across 0.5 s, but it plainly moves across 2 s.
      * anomalously still FOR THIS CHANNEL: below _DEAD_ACT_REL of the channel's own _DEAD_ACT_PCT
        local activity. This is the load-bearing gate. Locally, a dead lead and a LIVE-but-resting
        sensor are the same thing -- a constant plus white noise (a resting EDA channel really is
        that) -- so whiteness alone would condemn every quiet stretch of a slow signal. What separates
        them is the rest of the channel: a dead stretch is far stiller than this channel is when it is
        alive, while a resting sensor is quiet everywhere and so is not anomalous for itself. A high
        percentile (not the median, not mean |diff|) is what makes the reference survive a channel that
        is dead for most of its length -- the statistic must not be contaminated by the thing it measures.
      * still in absolute terms: never more than _DEAD_RANGE_REL of the channel's dynamic range. A
        broadband artifact burst (EMG, motion) is white too, and this is what keeps it out: a stretch
        swinging through a real part of the channel's range is never a dead sensor, however white.

    KNOWN LIMIT -- do not claim this case is solved. Detection is RELATIVE to the live part, so it
    degrades as the live part gets quieter: it is reliable when the dead stretch is well under
    _DEAD_ACT_REL of the channel's live activity, and it MISSES a dead stretch whose noise floor
    approaches that of a quiet live signal (e.g. a 1 uV floor against a 5 uV EEG montage). It also
    under-reports EXTENT near that boundary -- a channel that is 50% dead can score ~0.27. It is a
    detector, not a measurement. There is no scale-free way around this: locally, a dead lead and a
    quiet live one are the same trace, and only the rest of the channel distinguishes them."""
    ref = float(np.percentile(act, _DEAD_ACT_PCT))
    if ref <= 0.0:                                       # >=95% of the channel is exactly constant
        return np.zeros(xc.size, dtype=bool)             # ... which is a flatline by amplitude already
    still = (act <= _DEAD_ACT_REL * ref) & (act <= _DEAD_RANGE_REL * scale)
    mobility = _rolling_mobility(xc, max(2, int(round(_DEAD_WIN_S * fs))))
    return still & (mobility >= _DEAD_MOBILITY)


def _is_dead_channel(xv: np.ndarray, act: np.ndarray) -> bool:
    """True when the channel carries NO band-limited signal ANYWHERE -- only white measurement noise,
    which is what a disconnected input looks like once it dithers by an ADC code or two.

    This is the one case the relative test in :func:`_flat_mask` cannot see: a channel that is dead end
    to end has no live stretch left to set its scale, and 'dead + dither' is, up to a scale factor, just
    a low-amplitude noise trace -- amplitude alone cannot separate them, so the SHAPE has to. Two
    conditions, both scale-free (ratios), so volts and microvolts agree:
      * mobility std(diff x)/std(x) >= _DEAD_MOBILITY_WHOLE -- consecutive samples are uncorrelated.
        White noise reaches sqrt(2) ~ 1.41; any real biosignal is band-limited far below Nyquist, so it
        is smooth from sample to sample and sits well under 1 (a 10 Hz alpha at 250 Hz is ~0.6). This
        gate is TIGHTER than the local _DEAD_MOBILITY used per-stretch: the whole-channel estimate is
        low-variance, and the verdict here is all-or-nothing, so it must not fire on a merely NOISY live
        channel (10 uV alpha under 20 uV of white sensor noise measures 1.33).
      * its local activity is stationary (CV of the rolling std < _DEAD_CV) -- this keeps a LIVE but
        broadband channel (EMG, motion) from ever being called a dead sensor: real artifact arrives in
        bursts, amplifier/quantisation noise does not.
    A channel that is dead only in PART keeps the live part's std, so its mobility stays low and this
    never fires. That case belongs to :func:`_dead_stretch_mask`, which applies this same shape test
    per window. (This docstring used to claim the partial case was handled by the amplitude test in
    :func:`_flat_mask`. It was not: that test only sees a dead stretch quieter than 0.1% of the
    channel's range, and a real floating electrode dithers ~30x louder than that, so a half-dead
    sensor scored flatline_frac 0.000 and passed as perfect.)"""
    if xv.size < 3:
        return False
    sd = float(np.std(xv))
    if sd <= 0.0:
        return True
    if float(np.std(np.diff(xv))) / sd < _DEAD_MOBILITY_WHOLE:
        return False
    mean_act = float(np.mean(act))
    return mean_act > 0.0 and float(np.std(act)) / mean_act < _DEAD_CV


def _flat_mask(xf: np.ndarray, finite: np.ndarray, fs: float) -> np.ndarray:
    """Flatline = the trace does not MOVE. Judged scale-free, as a LOCAL-ACTIVITY ratio:

        (excursion over a ~0.5 s window)  /  (the channel's own 1-99 percentile dynamic range)

    i.e. "in half a second this trace travelled less than _FLAT_REL of its own working range". Both
    terms are amplitudes, so the ratio is identical in volts and in microvolts: the unit the loader
    happens to use cannot change the verdict (MNE returns SI volts -- a healthy 5 uV EEG arrives as
    5e-6, and an ABSOLUTE tolerance would call it a dead sensor).

    Why local activity and not the sample-to-sample |diff| this used to test:
      * |diff| shrinks with the sampling rate, so a heavily oversampled slow trace (0.1 Hz EDA at 1 kHz)
        has genuinely tiny per-sample steps and reads as flat -- yet over half a second it clearly moves.
      * a |diff| tolerance scaled by the channel's own median |diff| is CONTAMINATED: on a half-dead
        channel the dead part dominates that median, so the tolerance collapses to the dither level and
        the dead half stops looking flat -- the statistic destroys the very thing it must measure. The
        1-99 percentile range is taken over the whole channel and survives a channel that is dead for
        most of its length (it only needs ~1% live samples on each side of the distribution).
    Why the horizon has a floor in SAMPLES (_FLAT_MIN_WIN_N) and not only in seconds: the horizon has to
    be long enough that a live trace of THIS modality plainly moves across it, and each modality's
    sampling rate is already chosen for its bandwidth -- so a sample count travels with the bandwidth
    where a second count does not. Half a second is 125 samples of ECG but 4 samples of EDA, and a
    4-sample rolling std of an 8 Hz tonic trace is measuring the recorder's quantisation step, not
    motion: every real EDA record then reads as a mostly-dead sensor. Holding the horizon at >= 64
    samples leaves ECG/EEG untouched and gives EDA the several seconds its physiology needs. It also
    repairs a MISS in the same direction: at a 4-sample horizon the rolling-activity estimate is so
    noisy that its CV fails :func:`_is_dead_channel`'s stationarity test, so an 8 Hz channel that was
    dead end to end scored flatline_frac 0.000.
    A genuinely still stretch also has to LAST (_FLAT_MIN_RUN_S, and never less than the horizon
    itself -- a run shorter than the window that measured it is not independent evidence): every smooth
    signal is momentarily motionless at a turning point, and a clipped plateau is short -- neither is a
    dead sensor.

    Amplitude is necessary but NOT sufficient: it only sees a stretch stiller than _FLAT_REL of the
    range, and a floating electrode dithers far louder than that. So a stretch is flat if it fails to
    move (this test) OR if it carries no structure while the channel does (:func:`_dead_stretch_mask`)."""
    n = xf.size
    xv = xf[finite]
    if xv.size == 0:
        return np.ones(n, dtype=bool)                    # nothing valid at all: dead
    lo, hi = np.percentile(xv, [1.0, 99.0])
    scale = float(hi) - float(lo)
    if scale <= 0.0:
        return np.ones(n, dtype=bool)                    # zero dynamic range: constant / all-zero = dead
    xc = xf - float(np.median(xv))
    win = max(int(round(_FLAT_WIN_S * fs)), _FLAT_MIN_WIN_N)
    act = _rolling_std(xc, win)
    if _is_dead_channel(xv, act):
        return np.ones(n, dtype=bool)                    # structureless end to end: disconnected input
    flat = (act <= _FLAT_REL * scale) | _dead_stretch_mask(xc, act, scale, fs)
    min_run = max(int(round(_FLAT_MIN_RUN_S * fs)), win)
    return _sustained(flat, max(1, min(min_run, n)))


def _channel_flags(x: np.ndarray, fs: float):
    n = len(x)
    finite = np.isfinite(x)
    missing = 1.0 - finite.mean()
    xf = np.where(finite, x, 0.0)
    flat_frac = float(_flat_mask(xf, finite, fs).mean())
    xv = x[finite]
    lo = float(xv.min()) if xv.size else 0.0
    hi = float(xv.max()) if xv.size else 0.0
    rng = hi - lo
    if rng > 0.0:                                  # rails are a fraction of the channel's own range = scale free
        clip = (x >= lo + _CLIP_FRAC * rng) | (x <= hi - _CLIP_FRAC * rng)
        clip_frac = float(np.mean(clip & finite))
    else:
        clip_frac = 0.0                            # constant channel: that is flatline, not saturation
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
