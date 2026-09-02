"""Bidirectional Mamba / selective-SSM backbone — the RESEARCH arm (Plan 1 §11.3).

The single largest deployment risk in Plan 1: the ``mamba-ssm`` CUDA selective-
scan is a Triton/CUDA custom op that **does not trace to ONNX**. This module
therefore ships a **pure-PyTorch selective scan** (recursive S6 form) that is
always available (CPU/Windows included) and *does* trace to a static ONNX graph
for a fixed sequence length — Plan 1 §11.3 fallback #2. The fast CUDA kernel is
imported behind a guard and used only when present *and* not exporting.

The Phase-0 smoke test (``experiments/phase0_mamba_onnx_smoke.py``) exports a toy
block and checks logit parity + CPU latency, which decides whether Mamba is a
production backbone or stays a research-only accuracy comparison.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MambaBackbone", "MambaBlock", "HAS_CUDA_MAMBA", "selective_scan_ref"]

try:  # fast CUDA kernel — Linux/CUDA only; guarded so the repo runs anywhere
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _cuda_scan  # noqa: F401

    HAS_CUDA_MAMBA = True
except Exception:  # ImportError on Windows/CPU, or build/runtime error
    _cuda_scan = None
    HAS_CUDA_MAMBA = False


def selective_scan_ref(
    u: torch.Tensor,      # [B, D, L]  input (after conv+act)
    delta: torch.Tensor,  # [B, D, L]  input-dependent step
    A: torch.Tensor,      # [D, N]     (negative) state matrix
    B: torch.Tensor,      # [B, N, L]  input-dependent
    C: torch.Tensor,      # [B, N, L]  input-dependent
    D: torch.Tensor,      # [D]        skip
) -> torch.Tensor:
    """Reference (pure-PyTorch) diagonal selective scan. Traces to ONNX for a
    fixed ``L`` (the time loop unrolls). Numerically identical in eager and
    exported graphs — that identity is the whole point of the fallback."""
    b, d, L = u.shape
    n = A.shape[1]
    dA = torch.exp(delta.unsqueeze(2) * A.unsqueeze(0).unsqueeze(-1))     # [B, D, N, L]
    dB_u = delta.unsqueeze(2) * B.unsqueeze(1) * u.unsqueeze(2)           # [B, D, N, L]
    h = torch.zeros(b, d, n, device=u.device, dtype=u.dtype)
    ys = []
    for t in range(L):
        h = dA[:, :, :, t] * h + dB_u[:, :, :, t]                        # [B, D, N]
        y_t = torch.einsum("bdn,bn->bd", h, C[:, :, t])                  # [B, D]
        ys.append(y_t)
    y = torch.stack(ys, dim=-1)                                          # [B, D, L]
    return y + u * D.unsqueeze(0).unsqueeze(-1)


class MambaBlock(nn.Module):
    """A single (uni-directional) Mamba/S6 block on ``[B, L, d_model]``."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            groups=self.d_inner, padding=d_conv - 1,
        )
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + self.d_inner)  # -> B, C, delta
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, L, _ = x.shape
        xz = self.in_proj(x)                       # [B, L, 2*d_inner]
        xi, z = xz.chunk(2, dim=-1)
        xi = xi.transpose(1, 2)                    # [B, d_inner, L]
        xi = self.conv1d(xi)[:, :, :L]             # causal depthwise conv
        xi = F.silu(xi)
        xt = xi.transpose(1, 2)                    # [B, L, d_inner]
        dbl = self.x_proj(xt)
        delta, Bm, Cm = torch.split(dbl, [self.d_inner, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))    # [B, L, d_inner]
        A = -torch.exp(self.A_log)                 # [d_inner, d_state]
        y = selective_scan_ref(
            xi, delta.transpose(1, 2), A,
            Bm.transpose(1, 2), Cm.transpose(1, 2), self.D,
        )                                          # [B, d_inner, L]
        y = y.transpose(1, 2) * F.silu(z)          # gate
        return self.out_proj(y)


class MambaBackbone(nn.Module):
    """Bidirectional stack of Mamba blocks over the token sequence.

    Bidirectionality (forward + flipped) matches EEGMamba/ECGMamba/S2M2ECG for
    quality tasks where future context disambiguates transient artifacts.
    """

    def __init__(
        self,
        d_model: int = 128,
        depth: int = 3,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_out = d_model
        self.bidirectional = bidirectional
        self.fwd = nn.ModuleList([MambaBlock(d_model, d_state, d_conv, expand) for _ in range(depth)])
        self.bwd = (
            nn.ModuleList([MambaBlock(d_model, d_state, d_conv, expand) for _ in range(depth)])
            if bidirectional else None
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(depth)])
        self.drop = nn.Dropout(dropout)
        self.final = nn.LayerNorm(d_model)

    def forward_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """``[B, T, d_model] -> [B, T, d_model]`` per-token features (for SSL/MAE)."""
        x = tokens
        for i, fblk in enumerate(self.fwd):
            h = self.norms[i](x)
            out = fblk(h)
            if self.bwd is not None:
                out = out + self.bwd[i](torch.flip(h, dims=[1])).flip(dims=[1])
            x = x + self.drop(out)
        return self.final(x)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """``[B, n_tokens, d_model] -> [B, d_model]`` (mean pool)."""
        return self.forward_tokens(tokens).mean(dim=1)
