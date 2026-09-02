"""Per-modality hand-crafted SQA feature packs (deep-research slate, 2026-07-06).

Pure-numpy, deploy-reproducible (numpy.fft only — NO scipy) scalar feature vectors
that bring EDA/PPG/EEG toward ECG depth, fused into the shared trunk exactly like
the ECG spectral/nonlinear branches (host-precomputed -> 2nd ONNX input).

**Scale-invariance is the load-bearing constraint.** Store windows are already
z-scored per channel by the loaders, and the deployed model bakes a per-window
instance-norm, so any feature that depends on absolute amplitude or the DC term
(e.g. PPG perfusion index) is unrecoverable here and is NOT included — every
feature below is invariant to per-window affine rescaling, so its value measured
on the store is exactly its value at deploy. (Perfusion index remains a legitimate
*deploy-only* extra computable from the app's pre-instance-norm raw window; it is
intentionally out of scope for these store-validated packs.)

Packs (slate ranks):
- ``ppg_sqi_vector``  — amplitude moments + cardiac-band ratio + spectral entropy
  + autocorrelation pulse-regularity + HF-noise ratio (ranks 1/10, DC-free subset).
- ``eda_haar_vector`` — multiscale Haar-DWT detail-energy statistics + derivative
  transient extrema (rank 2), the canonical EDA motion-transient discriminators.
- ``eeg_spectral_vector`` — spectral-tilt band ratios + EMG spectral-flatness
  + 50/60 Hz line index + 1/f aperiodic slope + time-domain EQI (ranks 3/9).

Each ``*_vector(X, fs)`` takes ``X`` ``[N, L]`` or ``[N, 1, L]`` and returns
``(feat [N, D] float32, names [D])``. Rows are processed independently; features
are finite (NaN/Inf -> 0). Cost is a few hundred microseconds/window.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "ppg_sqi_vector", "eda_haar_vector", "eeg_spectral_vector",
    "advanced_feature_vector", "MODALITY_VECTOR",
]

_EPS = 1e-8


def _as_rows(X) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 3:
        X = X[:, 0, :]
    if X.ndim == 1:
        X = X[None, :]
    # Scrub non-finite samples ONCE, here: a single NaN/Inf anywhere used to abort
    # the whole batch (np.clip(nan, 1, c).astype(int64) -> INT64_MIN in
    # _dispersion_entropy, whose pattern index then blows up np.bincount), so one
    # invalid sample in a user's file graded zero windows for the entire record.
    # A no-op for finite input, so every store-validated feature value is unchanged.
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def _zrows(X: np.ndarray) -> np.ndarray:
    """Per-row z-score (scale-invariance: features on z-scored rows == on raw)."""
    mu = X.mean(1, keepdims=True)
    sd = X.std(1, keepdims=True)
    return (X - mu) / (sd + _EPS)


def _psd(X: np.ndarray, fs: float):
    """Hann-windowed one-sided power spectrum per row -> (freqs, psd[N, F])."""
    N, L = X.shape
    w = np.hanning(L)
    F = np.fft.rfft(X * w, axis=1)
    psd = (F.real ** 2 + F.imag ** 2)
    freqs = np.fft.rfftfreq(L, d=1.0 / fs)
    return freqs, psd


def _bandpower(freqs, psd, lo, hi):
    m = (freqs >= lo) & (freqs < hi)
    if not m.any():
        return np.zeros(psd.shape[0])
    return psd[:, m].sum(1)


def _finite(a: np.ndarray) -> np.ndarray:
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


def _stack(cols: dict[str, np.ndarray]):
    names = list(cols)
    feat = np.stack([_finite(np.asarray(cols[n], dtype=np.float64)) for n in names], axis=1)
    return feat.astype(np.float32), names


# ----------------------------------------------------------------------------- PPG
def ppg_sqi_vector(X, fs: float):
    """PPG signal-quality pack (scale-invariant subset of slate ranks 1 & 10).

    skew/kurt of the amplitude distribution (a clean pulse train is right-skewed),
    cardiac-band power ratio + spectral entropy (periodicity concentration),
    normalized-autocorrelation pulse-regularity in the 0.7-2.25 Hz cardiac lag band
    (the strongest DC-free morphology proxy, no peak detector needed), and the
    high-frequency noise fraction.
    """
    X = _as_rows(X)
    z = _zrows(X)
    skew = (z ** 3).mean(1)
    kurt = (z ** 4).mean(1) - 3.0
    freqs, psd = _psd(z, fs)
    total = _bandpower(freqs, psd, 0.1, min(8.0, fs / 2)) + _EPS
    cardiac = _bandpower(freqs, psd, 1.0, 2.25) / total
    hf = _bandpower(freqs, psd, 8.0, fs / 2) / (_bandpower(freqs, psd, 0.1, fs / 2) + _EPS)
    # spectral entropy over 0-8 Hz (low => concentrated => clean periodic pulse)
    band_m = (freqs >= 0.1) & (freqs < min(8.0, fs / 2))
    p = psd[:, band_m]
    p = p / (p.sum(1, keepdims=True) + _EPS)
    sent = -(p * np.log(p + _EPS)).sum(1) / np.log(max(2, band_m.sum()))
    # normalized autocorrelation peak in the cardiac lag range (pulse regularity)
    reg = _autocorr_regularity(z, fs, f_lo=0.7, f_hi=2.25)
    return _stack({
        "ppg_skew": skew, "ppg_kurt": kurt, "ppg_cardiac_ratio": cardiac,
        "ppg_hf_ratio": hf, "ppg_spec_entropy": sent, "ppg_pulse_regularity": reg,
    })


def _autocorr_regularity(z: np.ndarray, fs: float, f_lo: float, f_hi: float) -> np.ndarray:
    """Peak of the unbiased normalized autocorrelation within the [1/f_hi, 1/f_lo]
    s lag band — high for a regular pulse/rhythm, low for motion/noise."""
    N, L = z.shape
    lag_lo = max(1, int(np.floor(fs / f_hi)))
    lag_hi = min(L - 1, int(np.ceil(fs / f_lo)))
    if lag_hi <= lag_lo:
        return np.zeros(N)
    out = np.zeros(N)
    denom = (z ** 2).sum(1) + _EPS
    for lag in range(lag_lo, lag_hi + 1):
        c = (z[:, :L - lag] * z[:, lag:]).sum(1) / denom
        out = np.maximum(out, c)
    return out


# ----------------------------------------------------------------------------- EDA
def _haar_levels(z: np.ndarray, n_levels: int):
    """Non-recursive Haar-DWT: yield the detail coefficients at each of ``n_levels``
    dyadic scales (exact sum/difference cascade, pure numpy)."""
    a = z
    for _ in range(n_levels):
        if a.shape[1] < 2:
            break
        if a.shape[1] % 2:
            a = a[:, :-1]
        even, odd = a[:, 0::2], a[:, 1::2]
        detail = (even - odd) / np.sqrt(2.0)
        a = (even + odd) / np.sqrt(2.0)
        yield detail, a


def eda_haar_vector(X, fs: float, n_levels: int = 4):
    """EDA multiscale Haar detail-energy pack (slate rank 2).

    Fine-scale Haar detail std/max localize the sharp motion transients that
    dominate EDA corruption but that slow tonic SCRs do not; per-scale energy
    ratios are amplitude-invariant. Plus the derivative transient extrema.
    """
    X = _as_rows(X)
    z = _zrows(X)
    cols: dict[str, np.ndarray] = {}
    total_energy = (z ** 2).sum(1) + _EPS
    last_approx = z
    for j, (detail, approx) in enumerate(_haar_levels(z, n_levels), start=1):
        e = (detail ** 2).sum(1)
        cols[f"eda_haar{j}_std"] = detail.std(1)
        cols[f"eda_haar{j}_max"] = np.abs(detail).max(1)
        cols[f"eda_haar{j}_eratio"] = e / total_energy
        last_approx = approx
    # detail-to-approx energy ratio at the finest split (transient vs baseline)
    d1 = (z[:, 0::2][:, : (z.shape[1] // 2)] - z[:, 1::2][:, : (z.shape[1] // 2)]) / np.sqrt(2.0)
    a1 = (z[:, 0::2][:, : (z.shape[1] // 2)] + z[:, 1::2][:, : (z.shape[1] // 2)]) / np.sqrt(2.0)
    cols["eda_detail_approx_ratio"] = ((d1 ** 2).sum(1) + _EPS) / ((a1 ** 2).sum(1) + _EPS)
    # derivative transient extrema (motion spikes)
    dz = np.diff(z, axis=1)
    cols["eda_dmax"] = np.abs(dz).max(1)
    cols["eda_drms"] = np.sqrt((dz ** 2).mean(1))
    return _stack(cols)


# ----------------------------------------------------------------------------- EEG
def eeg_spectral_vector(X, fs: float):
    """EEG spectral-shape + time-domain EQI pack (slate ranks 3 & 9).

    Artifact TYPES separate by spectral tilt (eyem: low-freq heavy; musc/chew:
    broadband >20 Hz shelf; elec: narrowband mains spike). Spectral flatness is
    the cue separating broadband EMG from peaked neural gamma. Time-domain EQI
    (ZCR, max gradient, flatline fraction, Hjorth) catches electrode-pop / lead-off
    / clipping that spectral tilt misses. All amplitude-scale-invariant.
    """
    X = _as_rows(X)
    z = _zrows(X)
    nyq = fs / 2
    freqs, psd = _psd(z, fs)
    total = _bandpower(freqs, psd, 0.5, min(100.0, nyq)) + _EPS
    low = _bandpower(freqs, psd, 0.5, 8.0) / total          # delta+theta
    high = _bandpower(freqs, psd, 20.0, min(100.0, nyq)) / total
    gamma = _bandpower(freqs, psd, 30.0, min(60.0, nyq)) / total
    tilt = high / (low + _EPS)
    # EMG spectral flatness (Wiener entropy) over 20-100 Hz
    fm = (freqs >= 20.0) & (freqs < min(100.0, nyq))
    flat = _spectral_flatness(psd, fm)
    # 50 & 60 Hz mains index: peak power near line freq vs local floor
    line = _line_noise_index(freqs, psd, nyq)
    # 1/f aperiodic slope over 7-45 Hz (log-log linear fit)
    slope = _aperiodic_slope(freqs, psd, 7.0, min(45.0, nyq))
    # time-domain EQI
    dz = np.diff(z, axis=1)
    zcr = (np.abs(np.diff(np.sign(z), axis=1)) > 0).mean(1)
    kurt = (z ** 4).mean(1) - 3.0
    max_grad = np.abs(dz).max(1)
    flat_frac = (np.abs(dz) < 1e-3).mean(1)
    mob = np.sqrt((dz.var(1) + _EPS) / (z.var(1) + _EPS))           # Hjorth mobility
    ddz = np.diff(dz, axis=1)
    mob2 = np.sqrt((ddz.var(1) + _EPS) / (dz.var(1) + _EPS))
    comp = mob2 / (mob + _EPS)                                       # Hjorth complexity
    return _stack({
        "eeg_low_ratio": low, "eeg_high_ratio": high, "eeg_gamma_ratio": gamma,
        "eeg_tilt": tilt, "eeg_flatness": flat, "eeg_line_index": line,
        "eeg_aperiodic_slope": slope, "eeg_zcr": zcr, "eeg_kurt": kurt,
        "eeg_max_grad": max_grad, "eeg_flatline_frac": flat_frac,
        "eeg_hjorth_mob": mob, "eeg_hjorth_comp": comp,
    })


def _spectral_flatness(psd, mask):
    if not mask.any():
        return np.zeros(psd.shape[0])
    p = psd[:, mask] + _EPS
    gmean = np.exp(np.log(p).mean(1))
    amean = p.mean(1)
    return gmean / (amean + _EPS)


def _line_noise_index(freqs, psd, nyq):
    out = np.zeros(psd.shape[0])
    floor = np.median(psd, axis=1) + _EPS
    for f0 in (50.0, 60.0):
        if f0 >= nyq:
            continue
        m = (freqs >= f0 - 1.5) & (freqs <= f0 + 1.5)
        if m.any():
            out = np.maximum(out, psd[:, m].max(1) / floor)
    return out


def _aperiodic_slope(freqs, psd, lo, hi):
    m = (freqs >= lo) & (freqs < hi)
    if m.sum() < 3:
        return np.zeros(psd.shape[0])
    lf = np.log(freqs[m] + _EPS)
    lp = np.log(psd[:, m] + _EPS)
    lf0 = lf - lf.mean()
    denom = (lf0 ** 2).sum() + _EPS
    return (lp - lp.mean(1, keepdims=True)) @ lf0 / denom   # slope per row


# ------------------------------------------------------- ADVANCED (round-2 slate)
# Orthogonal feature axes the existing banks miss (deep-research round-2, 2026-07-06):
# 4th-order impulsiveness (spectral kurtosis), amplitude+order symbol dynamics
# (dispersion entropy), trajectory determinism (scalar RQA), discontinuous-jump
# isolation (bipower jump ratio), sequential ordinal structure (transition stats),
# and spectral periodicity (cepstral peak prominence). All pure-numpy, scale-invariant,
# deploy-reproducible. Cross-modal — screened as an add-on to the per-modality packs.

def _stft_power(z, fs, frame_s=0.25, hop_s=0.125):
    """Hann-windowed STFT power per row -> [N, n_frames, n_bins], or None if too short."""
    N, L = z.shape
    frame = max(8, int(round(frame_s * fs))); hop = max(1, int(round(hop_s * fs)))
    n_fr = 1 + (L - frame) // hop
    if n_fr < 3:
        return None
    idx = np.arange(frame)[None, :] + hop * np.arange(n_fr)[:, None]     # [n_fr, frame]
    frames = z[:, idx] * np.hanning(frame)                              # [N, n_fr, frame]
    S = np.fft.rfft(frames, axis=2)
    return S.real ** 2 + S.imag ** 2                                     # [N, n_fr, n_bins]


def _spectral_kurtosis(z, fs):
    """Per-bin spectral kurtosis SK(f)=<|S|^4>/<|S|^2>^2 - 2 over frames; SK~0 for
    stationary Gaussian, large for in-band transients (pops/spikes) at unchanged power."""
    P = _stft_power(z, fs)
    if P is None:
        return {"sk_max": np.zeros(len(z)), "sk_mean": np.zeros(len(z)), "sk_flat": np.zeros(len(z))}
    m2 = P.mean(1); m4 = (P ** 2).mean(1)                               # [N, n_bins]
    sk = m4 / (m2 ** 2 + _EPS) - 2.0
    sk_pos = np.clip(sk, 0, None) + _EPS
    flat = np.exp(np.log(sk_pos).mean(1)) / (sk_pos.mean(1) + _EPS)
    return {"sk_max": sk.max(1), "sk_mean": sk.mean(1), "sk_flat": flat}


def _dispersion_entropy(z, c=6, m=2):
    """Dispersion entropy: map each z-scored sample to one of c classes via a logistic
    NCDF (numpy-only), form m-length dispersion patterns, normalized Shannon entropy.
    Encodes amplitude LEVEL and order together (perm-entropy is order-only)."""
    N, L = z.shape
    # np.clip does NOT tame a NaN, and the int64 cast then yields INT64_MIN, whose
    # pattern index makes np.bincount raise for the whole batch. _as_rows already
    # scrubs, but keep the cast itself total for direct callers of this helper.
    lvl = np.ceil(c / (1.0 + np.exp(-1.702 * z)))
    cls = np.clip(np.where(np.isfinite(lvl), lvl, 1.0), 1, c).astype(np.int64) - 1  # 0..c-1
    pat = np.zeros((N, L - m + 1), dtype=np.int64)
    for k in range(m):
        pat += cls[:, k:L - m + 1 + k] * (c ** k)
    out = np.zeros(N); K = c ** m
    for i in range(N):
        cnt = np.bincount(pat[i], minlength=K).astype(float); p = cnt / (cnt.sum() + _EPS)
        nz = p[p > 0]; out[i] = -(nz * np.log(nz)).sum() / np.log(K)
    return out


def _rqa_determinism(z, m=3, tau=1, L_ds=128, rr=0.12, lmin=2):
    """Scale-invariant scalar recurrence-quantification (NOT the killed full-native RQA):
    downsample to L_ds, m-dim delay embed, threshold distances at the rr-percentile
    (fixed recurrence rate -> auto scale-invariant), report %determinism + mean/longest
    diagonal-line length (trajectory repeatability; motion destroys it).

    CONVENTION (audit 2026-08): ``rqa_det`` INCLUDES the line of identity (k=0) in
    both the diagonal-point count and ``rec``, unlike the textbook %DET. Because
    ``rr`` fixes the recurrence rate and ``L_ds``/``m``/``tau`` fix M=126 for every
    real window, this is an exact affine map of the LOI-excluded value
    (``shipped = 0.93389*standard + 0.06611``, residual <1e-15), fully absorbed by
    the baked per-feature standardization — so it changes no model output, but the
    raw value is NOT comparable to published %DET. Switching to the standard
    definition would change a shipped feature value and force a re-export."""
    N = len(z)
    det = np.zeros(N); mdl = np.zeros(N); ldl = np.zeros(N)
    for i in range(N):
        row = z[i]
        if len(row) > L_ds:                                             # decimate
            row = row[np.linspace(0, len(row) - 1, L_ds).astype(int)]
        M = len(row) - (m - 1) * tau
        if M < 8:
            continue
        emb = np.stack([row[k * tau:k * tau + M] for k in range(m)], axis=1)  # [M, m]
        D = np.sqrt(((emb[:, None, :] - emb[None, :, :]) ** 2).sum(-1))
        eps = np.percentile(D, rr * 100)
        R = (D <= eps)
        rec = R.sum()
        if rec == 0:
            continue
        diag_pts = 0; lengths = []
        for k in range(-(M - lmin), M - lmin + 1):                     # each diagonal
            d = np.diag(R, k).astype(np.int8)
            if d.sum() == 0:
                continue
            # run-lengths of 1s
            run = 0
            for v in d:
                if v:
                    run += 1
                else:
                    if run >= lmin:
                        diag_pts += run; lengths.append(run)
                    run = 0
            if run >= lmin:
                diag_pts += run; lengths.append(run)
        det[i] = diag_pts / (rec + _EPS)
        if lengths:
            mdl[i] = np.mean(lengths); ldl[i] = np.max(lengths) / M
    return {"rqa_det": det, "rqa_mean_diag": mdl, "rqa_long_diag": ldl}


def _jump_ratio(z):
    """Bipower jump-vs-diffusion: RV=sum(dz^2), BV=(pi/2)*sum(|dz_i||dz_{i+1}|); the
    jump ratio J=max(RV-BV,0)/RV isolates DISCONTINUOUS spike energy (electrode pops)
    that TKEO/kurtosis fire on for smooth high-freq oscillation too."""
    dz = np.diff(z, axis=1); adz = np.abs(dz)
    rv = (dz ** 2).sum(1) + _EPS
    bv = (np.pi / 2.0) * (adz[:, :-1] * adz[:, 1:]).sum(1)
    j = np.clip(rv - bv, 0, None) / rv
    return {"jump_ratio": j, "bv_rv_ratio": bv / rv}


def _ordinal_transition(z, m=3, tau=1):
    """Ordinal-pattern symbol statistics: forbidden-pattern fraction and
    self-transition probability (both genuinely sequential, discarded by
    permutation entropy) plus ``op_trans_ent``.

    NAMING DEBT (audit 2026-08): ``op_trans_ent`` is NOT a transition-matrix
    entropy — no transition matrix is built. ``tc`` below is a bincount of the
    MARGINAL pattern codes, so the column is exactly the normalized Bandt-Pompe
    permutation entropy at m=3, tau=1 (identical to
    ``nonlinear_features._perm_entropy(z, 3)`` to <4e-12). The column name is
    load-bearing: it is baked into the ppg/eeg/eda fusion cards' ``feat_names``
    and pinned by the app's feature contract, so renaming it (or computing the
    real m!xm! transition entropy) requires re-exporting all three models. Read
    it as permutation entropy until then."""
    from math import factorial
    N, L = z.shape; nfac = factorial(m)
    # rank-order code per m-window -> pattern id via Lehmer-ish encoding
    fwd = np.zeros(N); frac = np.zeros(N); tent = np.zeros(N)
    weights = np.array([factorial(m - 1 - k) for k in range(m)])
    for i in range(N):
        w = np.lib.stride_tricks.sliding_window_view(z[i], m)[::tau]   # [P, m]
        order = np.argsort(w, axis=1)
        # inversion-count code -> 0..m!-1
        codes = np.zeros(len(order), dtype=np.int64)
        for k in range(m):
            smaller = (order[:, k:k + 1] > order[:, k + 1:]).sum(1)
            codes += smaller * weights[k]
        seen = len(np.unique(codes))
        frac[i] = 1.0 - seen / nfac
        if len(codes) > 1:
            fwd[i] = float((codes[:-1] == codes[1:]).mean())
            tc = np.bincount(codes, minlength=nfac).astype(float); p = tc / (tc.sum() + _EPS)
            nz = p[p > 0]; tent[i] = -(nz * np.log(nz)).sum() / np.log(nfac)
    return {"op_forbidden": frac, "op_self_trans": fwd, "op_trans_ent": tent}


def _cepstral_cqi(z, fs, f_lo=0.5, f_hi=4.0):
    """Cepstral peak prominence: real cepstrum = irfft(log|rfft|); the peak in the
    plausible-period quefrency band measures the harmonic-comb regularity of the
    SPECTRUM (robust to a missing fundamental) — an axis time-domain features miss."""
    N, L = z.shape
    mag = np.abs(np.fft.rfft(z * np.hanning(L), axis=1)) + _EPS
    cep = np.fft.irfft(np.log(mag), n=L, axis=1)
    q_lo = max(1, int(fs / f_hi)); q_hi = min(L // 2, int(fs / f_lo))
    if q_hi <= q_lo:
        return {"cpp": np.zeros(N), "cqi": np.zeros(N)}
    band = np.abs(cep[:, q_lo:q_hi])
    peak = band.max(1); mean = band.mean(1) + _EPS
    return {"cpp": peak, "cqi": peak / mean}


def advanced_feature_vector(X, fs: float, modality: str = "ecg"):
    """Round-2 orthogonal feature bank (spectral-kurtosis + dispersion-entropy + scalar
    RQA + jump-ratio + ordinal-transition + cepstral). ``[N,L]`` -> ``(feat [N,D], names)``.
    Screened as an add-on to the per-modality packs; ~0.3-0.8 ms/window."""
    X = _as_rows(X); z = _zrows(X)
    cols: dict[str, np.ndarray] = {}
    cols.update(_spectral_kurtosis(z, fs))
    cols["disp_ent"] = _dispersion_entropy(z)
    cols.update(_rqa_determinism(z))
    cols.update(_jump_ratio(z))
    cols.update(_ordinal_transition(z))
    cols.update(_cepstral_cqi(z, fs))
    return _stack(cols)


# map modality -> its vector fn (for generic experiment code)
MODALITY_VECTOR = {
    "ppg": ppg_sqi_vector,
    "eda": eda_haar_vector,
    "eeg": eeg_spectral_vector,
}


def combined_vector(X, fs: float, modality: str):
    """Per-modality SQI pack ++ the round-2 advanced dynamics/HOS bank (the promoted
    deployable for EDA/PPG/EEG). ``[N,L]`` -> ``(feat [N,D], names)``. Deterministic
    column order (pack first, then advanced) so the app port and the ONNX feature-
    standardization align exactly."""
    Vp, np_names = MODALITY_VECTOR[modality](X, fs)
    Va, na_names = advanced_feature_vector(X, fs, modality)
    return np.concatenate([Vp, Va], axis=1).astype(np.float32), list(np_names) + list(na_names)
