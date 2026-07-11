"""Nonlinear / complexity per-window feature VECTOR for biosignal SQA (pure numpy).

Beyond frequency: SCALE-INVARIANT complexity descriptors that classic SQA/HRV
literature uses for signal quality, and — because they are amplitude-normalized —
plausibly transfer cross-cohort (the LODO lever) where raw-amplitude features fail.
All numpy-only (numpy.fft not even needed) -> deploy-reproducible in the app; fed as
a per-window vector to the fusion/MLP branch, no in-graph op.

Features (all cheap, O(L) or O(L log L) unless noted):
  * permutation entropy (order 3 & 5) — ordinal-pattern complexity, scale-invariant.
  * Higuchi & Petrosian fractal dimension — waveform complexity.
  * DFA alpha (detrended fluctuation) / Hurst — long-range correlation.
  * Poincare SD1/SD2/ratio on the 1st difference — short-vs-long-term variability.
  * Lempel-Ziv complexity (median-binarized) — algorithmic complexity.
  * Hjorth mobility & complexity; time-domain: ZCR, waveform length, slope-sign
    changes, RMS-of-diff, kurtosis, |skew|, and downsampled sample entropy.
"""
from __future__ import annotations

import numpy as np

__all__ = ["nonlinear_feature_vector", "nonlinear_feature_names"]

_EPS = 1e-8


def nonlinear_feature_names():
    return ["perm_entropy_o3", "perm_entropy_o5", "higuchi_fd", "petrosian_fd", "dfa_alpha",
            "poincare_sd1", "poincare_sd2", "poincare_ratio", "lziv", "hjorth_mobility",
            "hjorth_complexity", "zcr", "waveform_length", "slope_sign_changes", "rms_diff",
            "kurtosis", "abs_skew", "sample_entropy"]


def _znorm_rows(x):
    return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + _EPS)


def _perm_entropy(x, order, delay=1):
    """Vectorized permutation entropy over rows of x [N, L] (Bandt & Pompe 2002)."""
    n, L = x.shape
    m = L - (order - 1) * delay
    if m <= 1:
        return np.zeros(n)
    # embed: [N, m, order]
    emb = np.stack([x[:, i * delay: i * delay + m] for i in range(order)], axis=2)
    perms = np.argsort(emb, axis=2)                        # ordinal pattern per position
    # encode each length-`order` permutation as an integer code
    weights = (order ** np.arange(order))[None, None, :]
    codes = (perms * weights).sum(2)                       # [N, m]
    out = np.empty(n)
    for i in range(n):
        _, cnt = np.unique(codes[i], return_counts=True)
        p = cnt / cnt.sum()
        out[i] = -(p * np.log(p)).sum() / np.log(math_factorial(order))
    return out


def math_factorial(k):
    f = 1.0
    for i in range(2, k + 1):
        f *= i
    return f


def _higuchi_fd(x, kmax=10):
    """Higuchi fractal dimension per row [N, L]."""
    n, L = x.shape
    ks = np.arange(1, kmax + 1)
    lnL = np.empty((n, len(ks)))
    for j, k in enumerate(ks):
        lm = np.zeros(n)
        for mstart in range(k):
            idx = np.arange(mstart, L, k)
            if len(idx) < 2:
                continue
            seg = x[:, idx]
            length = np.abs(np.diff(seg, axis=1)).sum(1) * (L - 1) / (((len(idx) - 1)) * k)
            lm += length
        lm /= k
        lnL[:, j] = np.log(lm + _EPS)
    lnk = np.log(1.0 / ks)
    # slope of lnL vs lnk per row
    A = np.vstack([lnk, np.ones_like(lnk)]).T
    coef, *_ = np.linalg.lstsq(A, lnL.T, rcond=None)
    return coef[0]


def _petrosian_fd(x):
    d = np.diff(x, axis=1)
    ndelta = (np.diff(np.sign(d), axis=1) != 0).sum(1).astype(float)  # derivative sign changes
    L = x.shape[1]
    log10L = np.log10(L)
    return log10L / (log10L + np.log10(L / (L + 0.4 * ndelta)))


def _dfa(x, scales=(4, 8, 16, 32, 64, 128)):
    """DFA-1 scaling exponent alpha per row [N, L]."""
    n, L = x.shape
    y = np.cumsum(x - x.mean(1, keepdims=True), axis=1)
    scales = [s for s in scales if s < L // 2]
    F = np.empty((n, len(scales)))
    for j, s in enumerate(scales):
        nseg = L // s
        yy = y[:, : nseg * s].reshape(n, nseg, s)
        t = np.arange(s)
        A = np.vstack([t, np.ones_like(t)]).T
        # per-segment linear detrend (closed form), RMS over all segments
        pinv = np.linalg.pinv(A)                           # [2, s]
        coef = yy @ pinv.T                                 # [n, nseg, 2]
        trend = coef @ A.T                                 # [n, nseg, s]
        F[:, j] = np.sqrt(((yy - trend) ** 2).mean(axis=(1, 2)))
    ls = np.log(np.array(scales)); lf = np.log(F + _EPS)
    A2 = np.vstack([ls, np.ones_like(ls)]).T
    coef, *_ = np.linalg.lstsq(A2, lf.T, rcond=None)
    return coef[0]


def _poincare(x):
    d = np.diff(x, axis=1)                                 # 1st difference (detrends)
    a, b = d[:, :-1], d[:, 1:]
    sd1 = np.std((a - b) / np.sqrt(2), axis=1)
    sd2 = np.std((a + b) / np.sqrt(2), axis=1)
    return sd1, sd2, sd1 / (sd2 + _EPS)


def _lziv(x, ds=400):
    """Normalized Lempel-Ziv complexity of the median-binarized row (LZ76).

    LZ76 is a pure-Python O(L^2) scan, so the signal is downsampled to ``ds`` before
    binarizing (complexity ordering is preserved; keeps it a few ms not ~85 ms/window)."""
    n, L = x.shape
    if L > ds:
        x = x[:, np.linspace(0, L - 1, ds).astype(int)]
    Ld = x.shape[1]
    b = (x > np.median(x, axis=1, keepdims=True)).astype(np.uint8)
    out = np.array([_lz76(b[i]) for i in range(n)], dtype=np.float64)
    norm = Ld / np.log2(Ld)
    return out / (norm + _EPS)


def _lz76(s):
    """Kaspar-Schuster Lempel-Ziv (1976) complexity of a binary sequence."""
    n = len(s)
    c, ell, i, k, kmax = 1, 1, 0, 1, 1
    while ell + k <= n:
        if s[i + k - 1] == s[ell + k - 1]:
            k += 1
            if ell + k > n:
                c += 1
                break
        else:
            if k > kmax:
                kmax = k
            i += 1
            if i == ell:
                c += 1
                ell += kmax
                i, k, kmax = 0, 1, 1
            else:
                k = 1
    return float(c)


def _sample_entropy(x, m=2, r=0.2, ds=200):
    """SampEn on a downsampled, z-normed row (O(ds^2) per window)."""
    n = x.shape[0]
    out = np.empty(n)
    for i in range(n):
        s = x[i]
        if len(s) > ds:
            s = s[np.linspace(0, len(s) - 1, ds).astype(int)]
        s = (s - s.mean()) / (s.std() + _EPS)
        N = len(s); tol = r
        def phi(mm):
            emb = np.stack([s[a:a + mm] for a in range(N - mm + 1)])
            d = np.abs(emb[:, None, :] - emb[None, :, :]).max(2)
            np.fill_diagonal(d, np.inf)
            return (d <= tol).sum()
        A = phi(m + 1); B = phi(m)
        out[i] = -np.log((A + _EPS) / (B + _EPS))
    return out


def nonlinear_feature_vector(X: np.ndarray, fs: float, modality: str = "ecg",
                             with_sampen: bool = True) -> np.ndarray:
    """``X`` ``[N,1,L]`` (or ``[N,L]``) -> ``[N, F]`` scale-invariant nonlinear features."""
    if X.ndim == 3:
        X = X[:, 0, :]
    x = _znorm_rows(np.asarray(X, dtype=np.float64))       # scale-invariance up front
    d = np.diff(x, axis=1)
    pe3 = _perm_entropy(x, 3); pe5 = _perm_entropy(x, 5)
    hfd = _higuchi_fd(x); pfd = _petrosian_fd(x); dfa = _dfa(x)
    sd1, sd2, sdr = _poincare(x); lz = _lziv(x)
    v0 = x.var(1) + _EPS; v1 = d.var(1) + _EPS; v2 = np.diff(d, axis=1).var(1) + _EPS
    mob = np.sqrt(v1 / v0); comp = np.sqrt(v2 / v1) / (mob + _EPS)
    zcr = (np.abs(np.diff(np.sign(x), axis=1)) > 0).mean(1)
    wl = np.abs(d).sum(1)
    ssc = (np.diff(np.sign(d), axis=1) != 0).sum(1) / x.shape[1]
    rmsd = np.sqrt((d ** 2).mean(1))
    mu = x.mean(1, keepdims=True); sd = x.std(1, keepdims=True) + _EPS
    kurt = (((x - mu) / sd) ** 4).mean(1) - 3.0
    skew = np.abs((((x - mu) / sd) ** 3).mean(1))
    sampen = _sample_entropy(x) if with_sampen else np.zeros(len(x))
    feats = np.stack([pe3, pe5, hfd, pfd, dfa, sd1, sd2, sdr, lz, mob, comp, zcr, wl, ssc,
                      rmsd, kurt, skew, sampen], axis=1)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
