"""Filter-robust INTEGRITY guard — the tier-2 fix for the false-clean failure.

A raw-trained quality model keys on spectral / high-frequency cues that filtering removes, so a
pre-filtered but still-corrupted signal can be scored clean. The guard consults voters that are
INVARIANT to filtering (they live in beat morphology / detector-agreement / predictability, not in
out-of-band energy) and, ONLY when the input looks pre-filtered (:mod:`prefilter`), overrides a
confident-clean model verdict when those voters say the signal is corrupt.

Key voter: **bSQI** — agreement of two independent QRS detectors. An in-band artifact injects spurious
detections that lower agreement, and bSQI is empirically filter-invariant (unchanged by a 0.5-40 Hz
bandpass), so it still fires exactly when the spectral cues have gone silent. Gating the override on the
pre-filter detector means RAW-input behaviour is unchanged (no regression) — the guard only adds signal
on the pre-filtered path, which is the real deployment exposure.

Pure numpy; a few hundred µs/window. Reuses the two-detector bSQI from the training rich-SQI bank.
"""
from __future__ import annotations

import numpy as np

__all__ = ["bsqi", "tsqi", "rr_plausibility", "autocorr_regularity", "IntegrityVerdict", "integrity_guard"]

_EPS = 1e-8
# empirical bSQI operating point: multi-lead mean bSQI, clean ~0.84 vs in-band-artifacted ~0.70 on CinC.
# At 0.72 the regime-C guard recovers ~59% of filtered-artifacted while false-flagging ~2% of
# filtered-clean (target <10%). Calibrate on FILTERED validation data per deployment.
_BSQI_CORRUPT = 0.72
_REG_CORRUPT = 0.30
# tSQI (beat-to-template correlation): clean beats correlate ~0.9+, corrupt morphology drops below ~0.6.
# 0.66 is a conservative dissent threshold (only lowers a CONFIDENT-CLEAN, PRE-FILTERED verdict).
_TSQI_CORRUPT = 0.66
# beat/pulse detection band (Hz): ECG QRS = 5-15 Hz; PPG pulse = 0.5-8 Hz. Used by tSQI + plausibility.
_BEAT_BAND = {"ecg": (5.0, 15.0), "ppg": (0.5, 8.0)}


def _bandpass_fft(x, fs, lo, hi):
    L = len(x); X = np.fft.rfft(x); f = np.fft.rfftfreq(L, d=1.0 / fs)
    X[(f < lo) | (f > hi)] = 0.0
    return np.fft.irfft(X, n=L)


def _peaks(x, thr, dist):
    """numpy local maxima >= thr, greedy min-distance thinning (scipy-free)."""
    cand = np.where((x[1:-1] >= x[:-2]) & (x[1:-1] > x[2:]) & (x[1:-1] >= thr))[0] + 1
    if len(cand) == 0:
        return cand
    order = cand[np.argsort(-x[cand])]
    kept = []
    for p in order:
        if all(abs(p - k) >= dist for k in kept):
            kept.append(p)
    return np.sort(np.array(kept, dtype=int))


def bsqi(x, fs, tol=0.15):
    """Agreement of two independent QRS detectors on ``x`` (1-D). 0..1; both-empty -> 0."""
    x = np.asarray(x, dtype=np.float64)
    a = _bandpass_fft(x, fs, 5.0, 15.0)
    da = np.diff(a, prepend=a[:1]); mwi = np.convolve(da * da, np.ones(max(1, int(0.15 * fs))) / max(1, int(0.15 * fs)), mode="same")
    pa = _peaks(mwi, 0.3 * mwi.max() if mwi.max() > 0 else np.inf, max(1, int(0.25 * fs)))
    e = np.abs(_bandpass_fft(x, fs, 8.0, 20.0))
    pb = _peaks(e, e.mean() + 0.5 * e.std(), max(1, int(0.25 * fs)))
    if len(pa) == 0 or len(pb) == 0:
        return 0.0
    toln = tol * fs
    matched = sum(1 for p in pa if np.abs(pb - p).min() <= toln)
    return matched / max(len(pa), len(pb))


def autocorr_regularity(x, fs, f_lo=0.7, f_hi=3.0):
    """Peak normalized autocorrelation in the plausible beat-lag band (rhythm regularity, 0..1)."""
    z = (np.asarray(x, float) - np.mean(x)) / (np.std(x) + _EPS)
    L = len(z); lo = max(1, int(fs / f_hi)); hi = min(L - 1, int(fs / f_lo))
    if hi <= lo:
        return 0.0
    denom = (z * z).sum() + _EPS
    return float(max((z[:L - lag] * z[lag:]).sum() / denom for lag in range(lo, hi + 1)))


def _detect_beats(x, fs, modality):
    """Beat/pulse peak indices via a modality-band bandpass + energy-envelope peak picker (scipy-free).
    ECG uses the QRS band (5-15 Hz), PPG the pulse band (0.5-8 Hz)."""
    lo, hi = _BEAT_BAND.get((modality or "").lower(), (5.0, 15.0))
    a = _bandpass_fft(x, fs, lo, hi)
    da = np.diff(a, prepend=a[:1])
    w = max(1, int(0.15 * fs))
    e = np.convolve(da * da, np.ones(w) / w, mode="same")
    thr = 0.3 * e.max() if e.max() > 0 else np.inf
    return _peaks(e, thr, max(1, int(0.25 * fs)))


def tsqi(x, fs, modality="ecg"):
    """Template-matching beat-correlation SQI (Orphanidou 2015). Ensemble-average the detected beats into
    a template, return the mean beat-to-template Pearson correlation (0..1). Because it lives in beat
    MORPHOLOGY it is invariant to the band-pass/notch filtering that silences the model's spectral cues —
    an ideal 2nd filter-robust guard voter alongside bSQI, and the pulse-band voter PPG previously lacked.
    Returns 0.0 when fewer than 3 beats are found (too few to judge — callers treat 0 as 'abstain')."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    fs = float(fs) or 1.0
    pk = _detect_beats(x, fs, modality)
    if len(pk) < 3:
        return 0.0
    ibi = int(np.median(np.diff(pk)))
    if ibi < 4:
        return 0.0
    half = max(4, min(int(0.4 * ibi), int(0.5 * fs)))       # beat half-width, capped physiologically
    beats = [x[p - half:p + half] for p in pk if p - half >= 0 and p + half < len(x)]
    if len(beats) < 3:
        return 0.0
    B = np.vstack(beats)
    ts = B.mean(axis=0) - B.mean()                          # zero-mean template
    tden = np.sqrt((ts * ts).sum()) + _EPS
    corrs = [float(((b - b.mean()) * ts).sum() / ((np.sqrt(((b - b.mean()) ** 2).sum()) + _EPS) * tden))
             for b in B]
    return float(np.clip(np.mean(corrs), 0.0, 1.0))


def rr_plausibility(x, fs, modality="ecg"):
    """Physiological-plausibility of the detected beat timing (Orphanidou rule bank). Returns
    ``{plausible, hr_bpm, reasons}``. A filter-INVARIANT corruption cue: even when the spectrum looks
    clean, an impossible beat train betrays corruption. Implausible when HR is outside 40-180 bpm, a gap
    exceeds 3 s (max RR > 3 s), or the RR-interval ratio exceeds 2.2 (abrupt beat-to-beat change). With
    fewer than 3 beats it ABSTAINS (plausible=True) — too little evidence to dissent."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    fs = float(fs) or 1.0
    pk = _detect_beats(x, fs, modality)
    if len(pk) < 3:
        return {"plausible": True, "hr_bpm": 0.0, "reasons": []}
    rr = np.diff(pk) / fs
    rr = rr[rr > 0]
    if rr.size < 2:
        return {"plausible": True, "hr_bpm": 0.0, "reasons": []}
    hr = float(60.0 / np.median(rr))
    reasons = []
    if not (40.0 <= hr <= 180.0):
        reasons.append(f"HR {hr:.0f} bpm outside 40-180")
    if float(rr.max()) > 3.0:
        reasons.append(f"max RR {rr.max():.1f} s > 3 s")
    ratio = float(rr.max() / (rr.min() + _EPS))
    if ratio > 2.2:
        reasons.append(f"RR ratio {ratio:.1f} > 2.2")
    return {"plausible": not reasons, "hr_bpm": hr, "reasons": reasons}


class IntegrityVerdict:
    def __init__(self, corrupt_override: bool, reasons: list[str], voters: dict):
        self.corrupt_override = corrupt_override   # True -> downgrade/abstain the model's clean verdict
        self.reasons = reasons
        self.voters = voters

    def as_dict(self):
        return {"corrupt_override": self.corrupt_override, "reasons": self.reasons,
                "voters": {k: round(v, 3) for k, v in self.voters.items()}}


def integrity_guard(window, fs, modality, model_p_unusable, prefilter_verdict,
                    bsqi_corrupt=_BSQI_CORRUPT, reg_corrupt=_REG_CORRUPT, tsqi_corrupt=_TSQI_CORRUPT):
    """Override a confident-clean verdict when the input looks pre-filtered AND a filter-robust voter
    says corrupt. ``window`` [L] or [C,L]; ``model_p_unusable`` in 0..1; ``prefilter_verdict`` from
    :func:`prefilter.detect_prefiltering`. Returns :class:`IntegrityVerdict`.

    Voters (all filter-invariant): ECG uses bSQI (calibrated QRS-band detector agreement) PLUS tSQI
    (beat-template correlation) and beat-timing plausibility. PPG uses tSQI + plausibility on its pulse
    band — the "combination of SQIs beats any single index" finding (Li 2014), and the pulse-band voter
    that finally gives PPG a false-clean guard (bSQI stays ECG-only; it mis-scores device-filtered PPG)."""
    xw = np.asarray(window, float)
    if xw.ndim == 2:                                  # multi-lead: mean bSQI over leads (more robust)
        x = xw[0]
        b_ml = float(np.mean([bsqi(xw[c], fs) for c in range(xw.shape[0])]))
    else:
        x = xw; b_ml = None
    m = (modality or "").lower()
    voters = {}
    corrupt_votes, reasons = [], []
    if m == "ecg":
        # bSQI (multi-lead two-detector agreement) is the calibrated, filter-invariant trigger.
        b = b_ml if b_ml is not None else bsqi(x, fs); voters["bsqi"] = b
        if b < bsqi_corrupt:
            corrupt_votes.append(True); reasons.append(f"bSQI {b:.2f} < {bsqi_corrupt} (QRS detectors disagree)")
        # rhythm regularity is reported for context only — it does NOT trigger the override.
        voters["autocorr_reg"] = autocorr_regularity(x, fs)
    if m in ("ecg", "ppg"):
        # tSQI (beat-to-template correlation) — filter-invariant morphology voter; gives PPG its first
        # false-clean voter. tSQI==0 means "too few beats to judge" → abstain (do not dissent).
        t = tsqi(x, fs, m); voters["tsqi"] = t
        if 0.0 < t < tsqi_corrupt:
            corrupt_votes.append(True); reasons.append(f"tSQI {t:.2f} < {tsqi_corrupt} (beat morphology inconsistent)")
        # physiological-plausibility of the detected beat timing (abstains on <3 beats).
        pl = rr_plausibility(x, fs, m); voters["hr_bpm"] = pl["hr_bpm"]
        if not pl["plausible"]:
            corrupt_votes.append(True); reasons.append("; ".join(pl["reasons"]))
    model_clean = model_p_unusable < 0.5
    # gate: only override on a pre-filtered input where the model reads clean but a robust voter dissents
    override = bool(model_clean and prefilter_verdict.prefiltered and corrupt_votes)
    if override:
        reasons = ["input appears pre-filtered; filter-robust voters dissent from the clean score"] + reasons
    return IntegrityVerdict(override, reasons if override else [], voters)
