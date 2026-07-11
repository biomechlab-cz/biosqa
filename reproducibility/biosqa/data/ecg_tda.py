"""Topological (persistent-homology) ECG features — the privileged teacher signal for KD.

The published CinC-2011 SOTA (persistent-homology barcodes -> GoogLeNet, ~0.98) works because a
CLEAN ECG's Takens delay-embedding forms a persistent 1-cycle (H1 loop) from the quasi-periodic
beat structure, while artifact/noise destroys that topology. So H1 persistence (max/total lifetime,
count, entropy) + the persistence IMAGE is a quality signal a 1-D CNN cannot compute from raw samples.

This is a RESEARCH-SIDE feature (ripser/persim, NOT exportable) — used only to build a stronger
non-deployable TEACHER whose knowledge is then distilled into the exportable student.

``tda_features(X[N,12,L], fs) -> (feat[N,D] float32, names)``.
"""
from __future__ import annotations

import numpy as np

__all__ = ["tda_features", "TDA_LEADS"]

TDA_LEADS = (1, 7, 10)   # II, V2, V5 — limb + precordial coverage
_EPS = 1e-8


def _takens(x: np.ndarray, d: int, tau: int, npts: int):
    """z-scored delay embedding -> point cloud [<=npts, d], or None if too short."""
    x = (x - x.mean()) / (x.std() + _EPS)
    n = len(x) - (d - 1) * tau
    if n <= 8:
        return None
    step = max(1, n // npts)
    idx = np.arange(0, n, step)[:npts]
    return np.stack([x[idx + k * tau] for k in range(d)], axis=1).astype(np.float64)


def _pers_entropy(lifetimes: np.ndarray) -> float:
    p = lifetimes / (lifetimes.sum() + _EPS)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum()) if len(p) else 0.0


def tda_features(X: np.ndarray, fs: float, *, leads=TDA_LEADS, d: int = 3, tau: int = 8,
                 npts: int = 350, img_px: float = 0.5, img_range=(0.0, 3.0)):
    """Persistent-homology features per record. For each selected lead: H1 scalar stats
    (max/total persistence, count, entropy, max death) + a fixed-grid H1 persistence image.
    Returns ``(feat[N,D] float32, names)`` with a fixed D."""
    from persim import PersistenceImager
    from ripser import ripser

    imgr = PersistenceImager(pixel_size=img_px)
    imgr.birth_range = img_range
    imgr.pers_range = img_range
    # infer image size from a probe diagram
    probe = imgr.transform(np.array([[0.5, 1.5]]))
    img_dim = int(np.asarray(probe).size)
    n_scalar = 5

    N = len(X)
    D = len(leads) * (n_scalar + img_dim)
    out = np.zeros((N, D), dtype=np.float32)
    for i in range(N):
        col = 0
        for lead in leads:
            pc = _takens(X[i, lead], d, tau, npts)
            if pc is None:
                col += n_scalar + img_dim
                continue
            h1 = ripser(pc, maxdim=1)["dgms"][1]
            h1 = h1[np.isfinite(h1).all(1)] if len(h1) else h1
            if len(h1):
                life = h1[:, 1] - h1[:, 0]
                out[i, col:col + n_scalar] = [life.max(), life.sum(), len(h1),
                                              _pers_entropy(life), h1[:, 1].max()]
                img = np.asarray(imgr.transform(h1)).ravel()
            else:
                img = np.zeros(img_dim)
            col += n_scalar
            out[i, col:col + img_dim] = img[:img_dim]
            col += img_dim
    names = [f"tda_L{lead}_{k}" for lead in leads
             for k in (["h1_maxpers", "h1_totpers", "h1_count", "h1_entropy", "h1_maxdeath"]
                       + [f"pi{j}" for j in range(img_dim)])]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), names
