"""Task/metric-relative usability (research3.md trend D + arXiv:2602.12478: "good enough for R-peak
detection may be unusable for AF classification").

A window graded poor for MORPHOLOGY (Q0/Q1) may still be usable for RATE (heart/pulse rate) if beats
are reliably detectable. This asks a downstream-specific question the single ordinal grade can't, and
it is DISTINCT from recoverability: recoverability asks whether a FILTER would restore the signal;
rate-usability asks whether a rate estimate is ALREADY trustworthy despite poor morphology. Validated
(workflow experiment): baseline-wander / 50 Hz-powerline windows the model grades Q0 have bSQI≈1.0 and
a correct HR — a category the recoverability pass leaves Q0→Q0 (misses entirely); gate bSQI≥0.85 with
a physiological rate separates true cases from noise traps with no false positives.

The "usable for WHAT" question is answered per modality: ECG/PPG → RATE-usability (:func:`rate_usability`);
EEG → PER-BAND usability (:func:`band_usability`: which of δ/θ/α/β/γ carry real content above the 1/f floor,
so broadband muscle that kills β/γ doesn't discard usable δ/θ); EDA → TONIC/PHASIC usability
(:func:`eda_component_usability`: SCL vs SCR, which have different corruption sensitivities). Pure numpy.
"""
from __future__ import annotations

import numpy as np

from biosqa.inference.integrity import _bandpass_fft, _peaks, autocorr_regularity, bsqi

_EPS = 1e-12

#: plausible rate band (bpm) per modality. ECG uses the QRS-band bSQI+MWI detector; PPG's pulse lives
#: ~0.5-8 Hz, so it uses a pulse-band rate + pulse-regularity gate instead of the (ECG-calibrated) bSQI.
_RATE_BAND = {"ecg": (30.0, 220.0), "ppg": (30.0, 220.0)}


def estimate_hr(x, fs: float):
    """Median-IBI heart/pulse rate (bpm) from a band-pass + moving-window-integrator peak detector,
    or ``None`` if fewer than two beats are found."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    fs = float(fs) or 1.0
    a = _bandpass_fft(x, fs, 5.0, 15.0)
    da = np.diff(a, prepend=a[:1])
    w = max(1, int(0.15 * fs))
    mwi = np.convolve(da * da, np.ones(w) / w, mode="same")
    thr = 0.3 * mwi.max() if mwi.max() > 0 else np.inf
    pk = _peaks(mwi, thr, max(1, int(0.25 * fs)))
    if len(pk) < 2:
        return None
    ibi = np.diff(pk) / fs
    ibi = ibi[ibi > 0]
    if not ibi.size:
        return None
    return float(60.0 / np.median(ibi))


def estimate_pulse_rate(x, fs: float):
    """Pulse rate (bpm) for PPG from the dominant cardiac-band (30-220 bpm) autocorrelation lag, or
    ``None``. Autocorrelation is robust to the smooth, non-peaky PPG pulse the QRS detector mis-handles."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    fs = float(fs) or 1.0
    z = (x - x.mean()) / (x.std() + 1e-8)
    L = len(z)
    lo = max(1, int(fs / (220.0 / 60.0)))     # shortest plausible pulse interval (220 bpm)
    hi = min(L - 1, int(fs / (30.0 / 60.0)))  # longest (30 bpm)
    if hi <= lo:
        return None
    denom = (z * z).sum() + 1e-8
    ac = [(z[:L - lag] * z[lag:]).sum() / denom for lag in range(lo, hi + 1)]
    lag = lo + int(np.argmax(ac))
    return float(60.0 * fs / lag)


def rate_usability(window, fs: float, modality: str, *, bsqi_val: float | None = None,
                   usable_thr: float = 0.85, reg_thr: float = 0.5) -> dict:
    """``{rate_usable, bsqi, hr_bpm, reason}`` for one window. ``rate_usable`` is True only when the
    beat/pulse train is reliably periodic AND the estimated rate is physiological — a conservative gate.
    ECG uses bSQI ≥ ``usable_thr``; PPG uses pulse-band autocorrelation regularity ≥ ``reg_thr`` (bSQI is
    QRS-band-calibrated and mis-scores PPG). ``bsqi`` in the result is the modality's quality proxy."""
    m = (modality or "").lower()
    band = _RATE_BAND.get(m)
    if band is None:
        return {"rate_usable": False, "bsqi": 0.0, "hr_bpm": 0.0, "reason": ""}
    x = np.asarray(window, dtype=np.float64).reshape(-1)
    if m == "ppg":
        reg = float(autocorr_regularity(x, fs, f_lo=0.5, f_hi=3.7))   # pulse-band regularity = quality proxy
        hr = estimate_pulse_rate(x, fs)
        ok = (reg >= reg_thr) and (hr is not None) and (band[0] <= hr <= band[1])
        if ok:
            reason = f"pulse reliably periodic (reg {reg:.2f}); rate {hr:.0f} bpm"
        elif reg < reg_thr:
            reason = f"pulse regularity {reg:.2f} < {reg_thr:.2f}"
        else:
            reason = "rate not physiological"
        return {"rate_usable": bool(ok), "bsqi": reg, "hr_bpm": float(hr) if hr else 0.0, "reason": reason}
    b = float(bsqi_val) if bsqi_val is not None else float(bsqi(x, fs))
    hr = estimate_hr(x, fs)
    ok = (b >= usable_thr) and (hr is not None) and (band[0] <= hr <= band[1])
    if ok:
        reason = f"beats reliably detected (bSQI {b:.2f}); rate {hr:.0f} bpm"
    elif b < usable_thr:
        reason = f"bSQI {b:.2f} < {usable_thr:.2f}"
    else:
        reason = "rate not physiological"
    return {"rate_usable": bool(ok), "bsqi": b, "hr_bpm": float(hr) if hr else 0.0, "reason": reason}


# ---- EEG per-band usability (FOOOF-lite: content above the 1/f aperiodic floor) -------------------

_EEG_BANDS = [("delta", 0.5, 4.0), ("theta", 4.0, 8.0), ("alpha", 8.0, 13.0),
              ("beta", 13.0, 30.0), ("gamma", 30.0, 45.0)]


def band_usability(window, fs: float = 256.0) -> list[dict]:
    """Per-band EEG usability (research3 task-relative quality; Donoghue 2020 aperiodic decomposition).
    Returns EXACTLY 5 dicts (δ/θ/α/β/γ, fixed order) ``{band, lo_hz, hi_hz, usable, snr_db, detail}``. A
    band is usable when its in-band power stands ≥3 dB above the fitted 1/f aperiodic floor AND is not
    dominated by broadband muscle (flat 1/f) or 50/60 Hz line leakage. Broadband muscle kills β/γ while
    δ/θ stay valid — this recovers the low-band content a single global grade would discard. Pure numpy."""
    from biosqa.inference.sqi_breakdown import _line_noise_index
    x = np.nan_to_num(np.asarray(window, dtype=np.float64).reshape(-1))
    fs = float(fs) or 256.0
    n = len(x); nyq = fs / 2.0

    def _all_false(detail):
        return [{"band": nm, "lo_hz": lo, "hi_hz": hi, "usable": False, "snr_db": 0.0, "detail": detail}
                for (nm, lo, hi) in _EEG_BANDS]

    if n < 32 or float(np.std(x)) < 1e-9:
        return _all_false("flat/zero or too-short window")
    P = np.abs(np.fft.rfft(x - x.mean())) ** 2
    f = np.fft.rfftfreq(n, 1.0 / fs)
    # fit the aperiodic 1/f floor in log-log (line-excluded), then a robust 2nd pass under the peaks
    fit_hi = min(45.0, 0.95 * nyq)
    mask = (f >= 1.0) & (f <= fit_hi) & (P > 0)
    for f0 in (50.0, 60.0):
        mask &= ~((f >= f0 - 2.0) & (f <= f0 + 2.0))
    if int(mask.sum()) < 4:
        b, a, chi = 0.0, float(np.log10(np.median(P[P > 0]) + _EPS)), 0.0     # degenerate → flat floor
    else:
        lf, lp = np.log10(f[mask]), np.log10(P[mask] + _EPS)
        b, a = np.polyfit(lf, lp, 1)
        keep = (lp - (b * lf + a)) <= 0.0
        if int(keep.sum()) >= 4:
            b, a = np.polyfit(lf[keep], lp[keep], 1)
        chi = -float(b)

    def _floor(fi):
        return 10.0 ** (b * np.log10(np.maximum(fi, 1e-6)) + a)

    # low-frequency (1-13 Hz) aperiodic exponent: real EEG keeps a clear 1/f slope here even when muscle
    # flattens the HIGH region, whereas pure broadband noise is flat everywhere. A flat low region
    # (χ_low < 0.5) means the δ/θ/α power is noise, not content — used to stop the low bands passing for
    # a pure-noise window (their power sits above the lower-envelope floor regardless).
    lo_mask = (f >= 1.0) & (f <= 13.0) & (P > 0)
    chi_low = -float(np.polyfit(np.log10(f[lo_mask]), np.log10(P[lo_mask] + _EPS), 1)[0]) \
        if int(lo_mask.sum()) >= 4 else chi

    # broadband-muscle regime: a flat aperiodic exponent OR excess power in 30-45 Hz. The HF indicator is
    # BOUNDED to 30-45 Hz (below the 50/60 line) so a narrowband mains line does NOT masquerade as muscle.
    hi_musc = min(45.0, 0.95 * nyq)
    musc_frac = float(P[(f >= 30.0) & (f <= hi_musc)].sum() / (float(P.sum()) + _EPS))
    emg_regime = (chi < 0.5) or (musc_frac > 0.20)
    line_idx = _line_noise_index(x, fs)
    rows = []
    for (nm, lo, hi) in _EEG_BANDS:
        if lo >= 0.95 * nyq:
            rows.append({"band": nm, "lo_hz": lo, "hi_hz": hi, "usable": False, "snr_db": 0.0,
                         "detail": f"above Nyquist (fs/2={nyq:.0f} Hz)"}); continue
        clipped = hi > 0.95 * nyq
        hi_eff = min(hi, 0.95 * nyq)
        need_n = max(int(np.ceil(2 * fs / lo)), int(np.ceil(3 * fs / max(hi_eff - lo, _EPS))))
        if n < need_n:
            rows.append({"band": nm, "lo_hz": lo, "hi_hz": hi_eff, "usable": False, "snr_db": 0.0,
                         "detail": f"too short: need ≥ {need_n / fs:.1f}s to resolve {lo:g} Hz"}); continue
        sel = (f >= lo) & (f < hi_eff)
        if not sel.any():
            rows.append({"band": nm, "lo_hz": lo, "hi_hz": hi_eff, "usable": False, "snr_db": 0.0,
                         "detail": "no spectral bins in band"}); continue
        band_pow = float(P[sel].sum()); floor_pow = float(_floor(f[sel]).sum())
        snr_db = 10.0 * np.log10((band_pow + _EPS) / (floor_pow + _EPS))
        ratio = band_pow / (floor_pow + _EPS)
        center = (lo + hi_eff) / 2.0
        # High bands (β/γ) under a CONFIRMED broadband-muscle regime are genuinely swamped → flag them.
        # Low bands (δ/θ/α) survive muscle IF the low region still has 1/f structure, but are flagged when
        # the WHOLE low region is flat (pure noise). (A per-bin "strong peak" bypass was rejected: broadband
        # noise produces single-bin spikes > 6 dB that would spuriously rescue a muscle-contaminated band.)
        emg_bad = emg_regime and (center >= 13.0)
        low_noise = (center < 13.0) and (chi_low < 0.5)
        line_bad = (line_idx > 0.10) and (hi_eff >= 40.0)
        usable = (snr_db >= 3.0) and (not emg_bad) and (not low_noise) and (not line_bad)
        if line_bad:
            detail = "50/60 Hz line leakage"
        elif emg_bad:
            detail = f"broadband muscle (flat 1/f, χ={chi:.1f})"
        elif low_noise:
            detail = f"flat spectrum — no 1/f structure (χ_low={chi_low:.1f})"
        elif snr_db < 3.0:
            detail = "at 1/f floor (no separable rhythm)"
        else:
            detail = f"clear content {ratio:.1f}× over 1/f floor"
        if clipped:
            detail += " (clipped to Nyquist)"
        rows.append({"band": nm, "lo_hz": lo, "hi_hz": hi_eff, "usable": bool(usable),
                     "snr_db": round(float(np.nan_to_num(snr_db)), 2), "detail": detail})
    return rows


# ---- EDA tonic (SCL) / phasic (SCR) usability -----------------------------------------------------

def _lowpass_fft(x, fs, cut):
    L = len(x); X = np.fft.rfft(x); f = np.fft.rfftfreq(L, 1.0 / fs)
    X[f > cut] = 0.0
    return np.fft.irfft(X, n=L)


def _longest_run(mask) -> int:
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return 0
    d = np.diff(np.concatenate(([0], m.astype(np.int8), [0])))
    return int((np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).max())


def eda_component_usability(window, fs: float = 32.0) -> list[dict]:
    """EDA tonic (SCL) + phasic (SCR) usability (Benedek & Kaernbach 2010; Boucsein SPR guidelines). Returns
    EXACTLY 2 dicts (tonic, phasic) ``{component, usable, value, detail}``. Tonic-usable = no railing /
    flatline-dropout, SCL in 0.05-60 µS, slow drift ≤ 3 µS/s. Phasic-usable = SCR-band energy dominates the
    out-of-band (motion) energy and isn't isolated sharp bipolar spikes. Assumes RAW µS amplitudes (do NOT
    feed instance-normalized EDA — the absolute-µS gates would misfire). Pure numpy."""
    x = np.nan_to_num(np.asarray(window, dtype=np.float64).reshape(-1))
    fs = float(fs) or 32.0
    n = len(x); nyq = fs / 2.0

    def _out(tu, tv, td, pu, pv, pd):
        return [{"component": "tonic", "usable": bool(tu), "value": round(float(tv), 3), "detail": td},
                {"component": "phasic", "usable": bool(pu), "value": round(float(pv), 3), "detail": pd}]

    if n < 0.5 * fs:
        return _out(False, 0.0, "window too short (<0.5 s)", False, 0.0, "window too short (<0.5 s)")
    rng = float(np.ptp(x)); tol = max(1e-3, 1e-4 * rng)
    t = np.arange(n) / fs
    slope, intercept = np.polyfit(t, x, 1)
    trend = slope * t + intercept
    tonic = trend + _lowpass_fft(x - trend, fs, 0.05)

    # tonic detectors, evaluated in order flat → rail → range → drift
    flat = (float(np.std(x)) < 0.02) or (_longest_run(np.abs(np.diff(x)) <= tol) >= max(0.5 * fs, 0.25 * n))
    rail_frac = max(float((np.abs(x - x.max()) <= tol).mean()), float((np.abs(x - x.min()) <= tol).mean()))
    railed = (rng > tol) and (rail_frac > 0.05)
    scl = float(np.median(tonic))
    drift = abs(float(slope))          # slope is already µS/s (t is in seconds) — do NOT multiply by fs
    if flat:
        tu, td = False, "flatline / dropout"
    elif railed:
        tu, td = False, "railing / saturation"
    elif not (0.05 <= scl <= 60.0):
        tu, td = False, f"SCL {scl:.2f} µS out of 0.05-60"
    elif drift > 3.0:
        tu, td = False, f"baseline drift {drift:.1f} µS/s"
    else:
        tu, td = True, f"SCL {scl:.2f} µS, stable"

    # phasic: gated on a USABLE tonic (a flat/railed/out-of-range/drifting baseline makes the phasic
    # band untrustworthy — a drift ramp otherwise leaks into the 0.05-0.6 Hz band as spurious SCRs)
    if not tu:
        pu, pv, pd = False, 0.0, "no valid tonic baseline"
    elif n < 2 * fs:
        pu, pv, pd = False, 0.0, "too short to resolve SCRs (<2 s)"
    else:
        # derive the phasic band from hp = x − tonic (the fast residual). Because the tonic already
        # absorbs the trend + slow drift, hp ≈ 0 for a pure drift ramp → no spurious SCRs / motion.
        hp = x - tonic
        scr = _lowpass_fft(hp, fs, 0.6)                       # SCR band (hp already excludes <0.05 Hz tonic)
        X = np.abs(np.fft.rfft(hp)) ** 2
        ff = np.fft.rfftfreq(n, 1.0 / fs)
        p_scr = float(X[(ff >= 0.05) & (ff < 0.6)].sum())
        p_mot = float(X[(ff >= 0.6) & (ff <= nyq)].sum())
        motion_ratio = p_mot / (p_scr + _EPS)
        dph = np.diff(hp) * fs
        spikiness = float(np.percentile(np.abs(dph), 99) / (np.median(np.abs(dph)) + _EPS))
        phasic_rms = float(np.sqrt(np.mean(scr ** 2)))
        if motion_ratio > 0.5:
            pu, pv, pd = False, motion_ratio, f"motion-dominated (HF/SCR {motion_ratio:.1f})"
        elif spikiness > 10.0:
            pu, pv, pd = False, spikiness, f"sharp transients (crest {spikiness:.0f})"
        elif phasic_rms < 0.01:
            pu, pv, pd = True, 0.0, "quiet — no SCRs, no motion"
        else:
            n_scr = int(len(_peaks(scr, 0.05, max(1, int(fs)))))
            pu, pv, pd = True, float(n_scr), f"{n_scr} SCR(s), rms {phasic_rms:.2f} µS"
    return _out(tu, scl, td, pu, pv, pd)


def usability_verdicts(window, fs: float, modality: str) -> list[dict]:
    """Dispatch the per-modality "usable for what" verdicts shown in the inspector: EEG → per-band,
    EDA → tonic/phasic. ECG/PPG return ``[]`` here (their RATE verdict is surfaced separately as the
    rate-usable card). Each verdict dict carries ``{label, usable, detail}`` (+ modality-specific extras)."""
    m = (modality or "").lower()
    if m == "eeg":
        return [{"label": r["band"].capitalize() + f" ({r['lo_hz']:g}-{r['hi_hz']:g} Hz)",
                 "usable": r["usable"], "detail": r["detail"]} for r in band_usability(window, fs)]
    if m == "eda":
        return [{"label": ("Tonic (SCL)" if r["component"] == "tonic" else "Phasic (SCR)"),
                 "usable": r["usable"], "detail": r["detail"]} for r in eda_component_usability(window, fs)]
    return []
