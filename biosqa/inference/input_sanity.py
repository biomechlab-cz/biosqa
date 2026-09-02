"""Record-level ACQUISITION-REGIME sanity check + Domain-Shift Index (research3's central finding:
in-domain 98-99% collapses to ~80% cross-dataset — Rahman et al. 2025, arXiv:2502.14522).

Orthogonal to :mod:`data_quality` (which flags a BROKEN trace — missing samples, clipping, dropout) and to
the per-window false-clean/pre-filter guard (:mod:`integrity`, :mod:`prefilter`). This asks: was the
recording sampled at a rate consistent with the model's? A signal digitised BELOW Nyquist for its content —
or sampled/resampled wrong — folds high-frequency energy back down as ALIASING, which piles power up against
Nyquist. That is a distribution shift the raw-trained model never saw. The check is deliberately restricted
to the aliasing signature because it is the one regime deviation that is robust across modalities: clean
ECG/PPG/EEG/EDA all concentrate power well below Nyquist and carry ~0 near-Nyquist energy, so this does NOT
false-positive on the genuinely narrow-band modalities (EEG 1/f, PPG pulse, EDA tonic). The complementary
"over-filtered / band-limited" regime shift is caught separately by the pre-filter detector.

Reports a 0..1 Domain-Shift Index (0 = in-regime .. 1 = far out) + an explanatory flag. Pure numpy (one
rfft per channel, worst-case reduced). Runs on the in-memory inference path; streamed (out-of-core) records
skip it, like the other whole-signal passes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["input_sanity", "RegimeReport"]


@dataclass
class RegimeReport:
    dsi: float                       # Domain-Shift Index, 0 (in-regime) .. 1 (far out of regime)
    f_edge_hz: float                 # effective bandwidth: freq below which 99% of power lies (informational)
    band_ratio: float                # f_edge / Nyquist (informational — NOT penalised; natural spectra vary)
    near_nyquist_frac: float         # power fraction in the top 10% of the band — the aliasing signature
    flags: list = field(default_factory=list)

    @property
    def in_regime(self) -> bool:
        return self.dsi < 0.3


def input_sanity(signal: np.ndarray, fs: float, nyq_hz: float | None = None) -> RegimeReport:
    """Regime report for a full recording (``[C, L]`` or ``[L]``; channels reduced WORST-CASE — one
    out-of-regime lead is a problem). ``fs`` is the model's sampling rate (the signal is already resampled
    to it), so Nyquist is ``fs/2`` unless ``nyq_hz`` overrides. The Domain-Shift Index is driven by excess
    NEAR-NYQUIST energy (the aliasing / sampling-mismatch signature); ``f_edge``/``band_ratio`` are reported
    for context but are NOT penalised (a low value is normal for EEG/PPG/EDA)."""
    x = np.asarray(signal, dtype=np.float64)
    x = x if x.ndim == 2 else x[None, :]
    fs = float(fs) or 1.0
    nyq = float(nyq_hz) if nyq_hz else fs / 2.0
    n = x.shape[-1]
    if n < 16:
        return RegimeReport(dsi=0.0, f_edge_hz=0.0, band_ratio=1.0, near_nyquist_frac=0.0, flags=[])

    dsi = 0.0
    f_edge = 0.0
    band_ratio = 1.0
    near_nyq = 0.0
    for c in range(x.shape[0]):                    # worst-case over channels
        xc = np.nan_to_num(x[c] - np.nanmean(x[c]))
        if float(np.std(xc)) < 1e-9:
            continue
        P = np.abs(np.fft.rfft(xc)) ** 2
        f = np.fft.rfftfreq(n, 1.0 / fs)
        tot = float(P.sum()) + 1e-12
        csum = np.cumsum(P) / tot
        fe = float(f[min(int(np.searchsorted(csum, 0.99)), len(f) - 1)])
        nn = float(P[f >= 0.9 * nyq].sum() / tot)
        # A perfectly FLAT spectrum already puts ~10% of power in the top 10% of the band, so the ramp
        # starts ABOVE that baseline (0.15) to avoid flagging broadband/white/EMG-heavy signals as
        # aliased; real folded aliasing piles far more energy at Nyquist (a full alarm by ~35%).
        d = float(np.clip((nn - 0.15) / 0.20, 0.0, 1.0))
        if d >= dsi:
            dsi, f_edge, band_ratio, near_nyq = d, fe, fe / (nyq + 1e-9), nn

    flags = []
    if dsi > 0.2:
        flags.append(f"{near_nyq:.0%} of power sits near Nyquist ({0.9 * nyq:.0f}-{nyq:.0f} Hz) — the signal "
                     f"looks aliased or sampled at the wrong rate; the model's scores may not transfer.")
    return RegimeReport(dsi=float(dsi), f_edge_hz=float(f_edge), band_ratio=float(band_ratio),
                        near_nyquist_frac=float(near_nyq), flags=flags)
