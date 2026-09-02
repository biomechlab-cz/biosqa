"""Signal-specific learnable FRONTENDS — drop-in replacements for PatchAdapter
(campaign round-2, 2026-07-06).

The 3-campaign finding was FEATURES > ARCHITECTURE: generic trunk swaps (transformer,
Mamba, KAN, ...) merely tie a tuned dilated CNN. The hypothesis these frontends test is
narrower and signal-specific: replace the trunk's FIRST layer (a free strided conv,
``PatchAdapter``) with a layer whose inductive bias is a bank of BAND-PASS filters, so the
network learns a modality-tuned filterbank INSIDE the graph — potentially subsuming the
hand-crafted spectral 2nd-input branch. Each frontend outputs the same token tensor
``[B, n_tokens, d_model]`` PatchAdapter does, so it drops into BioSQAModel/fusion trunks
unchanged, and every op (conv, sin, elementwise) traces to legacy ONNX.

- ``SincFrontend`` — a SincNet learnable band-pass filterbank (Ravanelli & Bengio 2018:
  each filter is parameterized by (low-cutoff, bandwidth), Hamming-windowed) followed by a
  strided conv tokenizer. The spectral prior a plain conv lacks; ~2N params for the bank.
- ``MultiScaleFrontend`` — parallel strided convs at several kernel sizes (multi-resolution
  receptive field), concatenated then projected. The multi-timescale prior.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SincConv1d", "SincFrontend", "MultiScaleFrontend"]


class SincConv1d(nn.Module):
    """Learnable band-pass filterbank (SincNet). ``n_filt`` kernels, each a band-pass
    parameterized by a learnable low cutoff + bandwidth (Hz), Hamming-windowed. Traces to
    ONNX (the kernels are recomputed from the two parameter vectors with sin ops each
    forward; freezing the params at export collapses to a plain Conv1d)."""

    def __init__(self, n_filt: int, kernel_size: int, fs: float, stride: int = 1,
                 min_hz: float = 0.5, min_band: float = 1.0):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1  # odd -> symmetric, zero-phase
        self.n_filt = n_filt
        self.kernel_size = kernel_size
        self.stride = stride
        self.fs = float(fs)
        self.min_hz = float(min_hz)
        self.min_band = float(min_band)
        # init band edges linearly from min_hz to Nyquist (kept in Hz; abs() in forward
        # guarantees positivity, so no gradient sign games).
        nyq = self.fs / 2
        low = torch.linspace(min_hz, nyq - (min_band + min_hz), n_filt)
        self.low_hz_ = nn.Parameter(low)
        self.band_hz_ = nn.Parameter(torch.full((n_filt,), (nyq - min_hz) / n_filt))
        # time index n/fs for the symmetric sinc, and a Hamming window (both buffers)
        half = (kernel_size - 1) // 2
        n = torch.arange(-half, half + 1).float() / self.fs      # [K] seconds
        self.register_buffer("t_", n)
        self.register_buffer("window_", torch.hamming_window(kernel_size))
        self.register_buffer("center_", torch.zeros(1))          # placeholder (DC handled below)

    def _filters(self) -> torch.Tensor:
        low = self.min_hz + self.low_hz_.abs()                                   # [F]
        high = torch.clamp(low + self.min_band + self.band_hz_.abs(), max=self.fs / 2)
        t = self.t_.unsqueeze(0)                                                  # [1,K]
        # band-pass = difference of two low-pass sinc filters (2*f*sinc(2*pi*f*t))
        f_low = low.unsqueeze(1); f_high = high.unsqueeze(1)                      # [F,1]
        lp_high = 2 * f_high * _sinc(2 * math.pi * f_high * t)
        lp_low = 2 * f_low * _sinc(2 * math.pi * f_low * t)
        band = (lp_high - lp_low) * self.window_.unsqueeze(0)                     # [F,K]
        band = band / (band.abs().amax(dim=1, keepdim=True) + 1e-8)              # scale-invariant kernels
        return band.unsqueeze(1)                                                  # [F,1,K]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B,1,L] -> [B, n_filt, L']`` band-decomposed signal."""
        return F.conv1d(x, self._filters(), stride=self.stride, padding=0)


def _sinc(x: torch.Tensor) -> torch.Tensor:
    """sin(x)/x with the removable singularity at 0 handled (=1).

    The denominator uses a SAFE x (1 where x≈0) so the division never evaluates 0/0 —
    ``torch.where`` computes both branches, so a raw ``sin(x)/x`` would poison the backward
    pass with NaN gradients at the sinc center even though the forward value is masked.
    """
    safe = torch.where(x.abs() < 1e-8, torch.ones_like(x), x)
    return torch.where(x.abs() < 1e-8, torch.ones_like(x), torch.sin(x) / safe)


class SincFrontend(nn.Module):
    """SincNet band-pass bank -> per-band abs + strided-conv tokenizer -> ``[B,T,d_model]``.

    Drop-in for :class:`biosqa.models.adapters.PatchAdapter`. The sinc bank gives the
    spectral (bandpass) prior; the ``abs`` makes each band an envelope (SQA cares about
    band ENERGY, not phase); the strided conv tokenizes to the trunk's token grid.
    """

    def __init__(self, patch_len: int, stride: int, d_model: int = 128, *, c_in: int = 1,
                 max_tokens: int = 128, fs: float = 100.0, n_filt: int = 32,
                 sinc_kernel: int | None = None, dropout: float = 0.0):
        super().__init__()
        self.sinc = SincConv1d(n_filt, sinc_kernel or (patch_len | 1), fs, stride=1)
        self.bn = nn.BatchNorm1d(n_filt)
        self.proj = nn.Conv1d(n_filt, d_model, kernel_size=patch_len, stride=stride)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        z = self.sinc(x).abs()               # [B, n_filt, L'] band envelopes
        z = self.bn(z)
        z = self.proj(z).transpose(1, 2)     # [B, T, d_model]
        z = z + self.pos_embed[:, : z.size(1), :]
        return self.dropout(z)


class MultiScaleFrontend(nn.Module):
    """Parallel strided convs at several kernel sizes (multi-resolution receptive field),
    concatenated then projected to ``d_model``. Drop-in for PatchAdapter; the inductive
    bias is that quality cues live at MULTIPLE timescales simultaneously (fast artifact
    spikes + slow baseline drift), which a single-kernel patch conv cannot see at once."""

    def __init__(self, patch_len: int, stride: int, d_model: int = 128, *, c_in: int = 1,
                 max_tokens: int = 128, scales=(0.5, 1.0, 2.0), dropout: float = 0.0):
        super().__init__()
        kernels = [max(3, int(round(patch_len * s)) | 1) for s in scales]
        per = d_model // len(kernels)
        widths = [per] * len(kernels)
        widths[-1] += d_model - per * len(kernels)  # absorb rounding into the last branch
        self.branches = nn.ModuleList([
            nn.Conv1d(c_in, w, kernel_size=k, stride=stride, padding=k // 2)
            for k, w in zip(kernels, widths)
        ])
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        outs = [b(x) for b in self.branches]
        t = min(o.size(-1) for o in outs)
        z = torch.cat([o[..., :t] for o in outs], dim=1)   # [B, d_model, T]
        z = z.transpose(1, 2)                               # [B, T, d_model]
        z = z + self.pos_embed[:, : z.size(1), :]
        return self.dropout(z)
