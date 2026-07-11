"""Synthetic artifact-TYPE injection for ECG (level-3 supervision, calibrated SNR).

Native ECG artifact-type labels are sparse (only PTB-XL flags + NSTDB 'em'), so
the primary route to a balanced, learnable artifact-type dataset is to inject the
MIT-BIH Noise Stress Test Database pure-noise templates — ``bw`` (baseline
wander), ``ma`` (muscle/EMG), ``em`` (electrode motion) — plus a synthesized
powerline sinusoid, into CLEAN ECG carriers at a controlled SNR. Every injected
window gets an exact multi-hot :data:`harmonize.ARTIFACT_TYPES` label, and
1-2 types may co-occur (multi-label, matching real recordings).

The noise templates are the physical MIT-BIH recordings (real bw/ma/em), not
parametric models, so morphology is realistic; only their amplitude (SNR) and
mixing are controlled. Output windows are per-window z-scored to match the
deployment graph.
"""
from __future__ import annotations

import numpy as np

from .datasets import nstdb
from .harmonize import ARTIFACT_TYPE_INDEX, ARTIFACT_TYPES

__all__ = ["NOISE_TYPE_MAP", "synth_ecg_artifacts", "load_resampled_templates", "powerline_wave",
           "PROCEDURAL_TYPES"]

# NSTDB pure-noise record -> canonical artifact-type class.
NOISE_TYPE_MAP = {"bw": "baseline_wander", "ma": "muscle_emg", "em": "electrode_leadoff"}
# artifact classes reachable by ECG synthetic injection (+ 'clean' negative, + 'powerline').
SYNTH_TYPES = ("clean", "baseline_wander", "muscle_emg", "electrode_leadoff", "powerline")
# PROCEDURAL types (transforms of the clean signal, not additive-SNR noise). These are
# the rare, under-labeled types (burst/clipping/dropout/motion ~0 F1 in the ceiling probe)
# that drag the macro-F1 — generating them procedurally gives the type head positives to
# learn from. The macro-F1 ceiling was DATA-limited, so this is the lever.
PROCEDURAL_TYPES = ("burst_transient", "clipping_flatline", "dropout", "motion")


def _burst(sig, fs, rng):
    """1-3 sharp high-amplitude transient deflections (motion spikes / electrode taps)."""
    L = len(sig); out = sig.copy(); sd = sig.std() + 1e-6
    t = np.arange(L)
    for _ in range(int(rng.integers(1, 4))):
        c = int(rng.integers(0, L)); w = max(1, int(rng.uniform(0.008, 0.04) * fs))
        amp = rng.uniform(3.0, 9.0) * sd * (1.0 if rng.random() < 0.5 else -1.0)
        out = out + amp * np.exp(-0.5 * ((t - c) / w) ** 2).astype(np.float32)
    return out


def _clip(sig, rng):
    """Hard amplitude saturation (ADC/lead clipping) -> flat tops."""
    thr = rng.uniform(0.35, 0.8) * float(np.max(np.abs(sig)) + 1e-6)
    return np.clip(sig, -thr, thr)


def _dropout(sig, fs, rng):
    """Flatline a contiguous segment (signal loss / electrode lead-off gap)."""
    L = len(sig); out = sig.copy()
    w = int(rng.uniform(0.08, 0.45) * L); s = int(rng.integers(0, max(1, L - w)))
    out[s:s + w] = rng.normal(0.0, 0.01 * (sig.std() + 1e-6), size=w).astype(np.float32)
    return out


def _motion(sig, fs, rng):
    """Broadband motion: large slow baseline swing + a transient burst."""
    L = len(sig); t = np.arange(L) / fs; sd = sig.std() + 1e-6
    swing = rng.uniform(2.0, 5.0) * sd * np.sin(2 * np.pi * rng.uniform(0.1, 0.6) * t + rng.uniform(0, 2 * np.pi))
    return _burst(sig + swing.astype(np.float32), fs, rng)


_PROCEDURAL_FN = {
    "burst_transient": lambda s, fs, rng: _burst(s, fs, rng),
    "clipping_flatline": lambda s, fs, rng: _clip(s, rng),
    "dropout": lambda s, fs, rng: _dropout(s, fs, rng),
    "motion": lambda s, fs, rng: _motion(s, fs, rng),
}


def _znorm(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-6)


def load_resampled_templates(fs_out: float) -> dict[str, np.ndarray]:
    """NSTDB bw/ma/em templates as single-channel 1-D arrays resampled to ``fs_out``."""
    from fractions import Fraction

    from scipy.signal import resample_poly

    fs_in = 360.0
    frac = Fraction(fs_out / fs_in).limit_denominator(1000)
    out = {}
    for tok, sig in nstdb.load_noise_templates().items():
        x = np.asarray(sig, dtype=np.float32)
        x = x[0] if x.ndim == 2 else x           # take channel 0
        out[tok] = resample_poly(x, frac.numerator, frac.denominator).astype(np.float32)
    return out


def powerline_wave(length: int, fs: float, freq: float, rng, harmonics=(1, 2, 3)) -> np.ndarray:
    """Mains-interference waveform: sum of ``freq`` harmonics, random phase."""
    t = np.arange(length) / fs
    w = np.zeros(length, dtype=np.float32)
    for h in harmonics:
        w += (1.0 / h) * np.sin(2 * np.pi * freq * h * t + rng.uniform(0, 2 * np.pi))
    return w.astype(np.float32)


def _inject(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Add ``noise`` to ``clean`` scaled to the target per-window ``snr_db``."""
    sp = float(np.mean(clean ** 2)) + 1e-12
    npow = float(np.mean(noise ** 2)) + 1e-12
    scale = np.sqrt(sp / (npow * 10 ** (snr_db / 10.0)))
    return clean + scale * noise


def synth_ecg_artifacts(
    clean_X: np.ndarray, fs: float, *, snr_db_range=(-2.0, 12.0), powerline_freq=(50.0, 60.0),
    p_second_type: float = 0.25, clean_frac: float = 0.25, seed: int = 0,
    include_procedural: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Inject typed noise into clean ECG carriers.

    Args:
        clean_X: ``[N, 1, L]`` clean (z-scored) ECG windows to use as carriers.
        fs: sampling rate of the carriers (canonical ECG rate).
        snr_db_range: uniform SNR range per injection (lower = more corrupt).
        p_second_type: probability of adding a co-occurring second artifact type
            (multi-label). clean windows never get a second type.
        clean_frac: fraction of windows left clean (negative class).

    Returns:
        ``(X [N,1,L] float32 z-scored, Y [N,K] multi-hot, ARTIFACT_TYPES list)``.
    """
    rng = np.random.default_rng(seed)
    templates = load_resampled_templates(fs)
    K = len(ARTIFACT_TYPES)
    N, _, L = clean_X.shape
    X = np.empty((N, 1, L), dtype=np.float32)
    Y = np.zeros((N, K), dtype=np.float32)
    additive = ["baseline_wander", "muscle_emg", "electrode_leadoff", "powerline"]
    procedural = list(PROCEDURAL_TYPES) if include_procedural else []
    injectable = additive + procedural
    tok_by_type = {v: k for k, v in NOISE_TYPE_MAP.items()}

    def draw_noise(tp: str) -> np.ndarray:
        if tp == "powerline":
            return powerline_wave(L, fs, float(rng.choice(powerline_freq)), rng)
        t = templates[tok_by_type[tp]]
        s = int(rng.integers(0, max(1, len(t) - L)))
        return t[s:s + L]

    def apply_type(sig, tp):
        if tp in _PROCEDURAL_FN:
            return _PROCEDURAL_FN[tp](sig, fs, rng).astype(np.float32)
        return _inject(sig, draw_noise(tp), float(rng.uniform(*snr_db_range)))

    for i in range(N):
        clean = clean_X[i, 0].astype(np.float32)
        if rng.random() < clean_frac:
            X[i, 0] = _znorm(clean[None, :])[0]
            Y[i, ARTIFACT_TYPE_INDEX["clean"]] = 1.0
            continue
        types = [str(rng.choice(injectable))]
        if rng.random() < p_second_type:
            other = str(rng.choice([t for t in injectable if t != types[0]]))
            types.append(other)
        sig = clean.copy()
        for tp in types:
            sig = apply_type(sig, tp)
            Y[i, ARTIFACT_TYPE_INDEX[tp]] = 1.0
        X[i, 0] = _znorm(sig[None, :])[0]
    return X, Y, list(ARTIFACT_TYPES)


def mint_unacceptable_quality(X_accept: np.ndarray, fs: float, n: int, *,
                              snr_db_range=(-6.0, 6.0), seed: int = 0):
    """Mint ``n`` synthetic UNACCEPTABLE (quality Q0) 12-lead ECG records from acceptable ones.

    The CinC-2011 SOTA balanced its 773/225 split exactly this way. ``X_accept`` ``[M, C, L]``
    (z-scored). Returns ``(X [n, C, L] float32, y[n]=0)``. Each synth = a random acceptable record
    corrupted across ALL leads by one of: additive NSTDB bw/ma/em at a random (per-lead-jittered)
    SNR, procedural motion, hard clipping, or a dropout gap. Output is per-lead z-scored to match
    the deployment graph. Train-time only (never test) -> export-neutral.
    """
    rng = np.random.default_rng(seed)
    templates = load_resampled_templates(fs)          # {bw, ma, em} -> 1-D
    toks = list(templates)
    M, C, L = X_accept.shape
    out = np.empty((n, C, L), dtype=np.float32)
    modes = ["additive", "additive", "additive", "motion", "clip", "dropout"]
    for i in range(n):
        base = X_accept[rng.integers(M)].astype(np.float32).copy()   # [C, L]
        mode = modes[rng.integers(len(modes))]
        if mode == "additive":
            t = templates[toks[rng.integers(len(toks))]]
            s = int(rng.integers(0, max(1, len(t) - L)))
            noise = t[s:s + L]
            snr = float(rng.uniform(*snr_db_range))
            for lead in range(C):
                base[lead] = _inject(base[lead], noise, snr + float(rng.uniform(-3.0, 3.0)))
        else:
            fn = {"motion": _motion, "clip": lambda s, f, r: _clip(s, r), "dropout": _dropout}[mode]
            for lead in range(C):
                base[lead] = fn(base[lead], fs, rng).astype(np.float32)
        out[i] = _znorm(base)
    return out, np.zeros(n, dtype=np.int64)
