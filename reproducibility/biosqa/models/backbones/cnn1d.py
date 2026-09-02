"""1D-CNN backbone — the exportable baseline and CPU fallback (Plan 1 §7.2).

Operates on the patch-token sequence ``[B, n_tokens, d_model]`` (d_model as
channels, token axis as length) via a residual dilated-conv stack + global
average pooling. Small, fast, and traces to ONNX with a static graph — the
default deployment fallback if a transformer/Mamba can't meet the latency budget.
"""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["CNN1DBackbone"]


class _ResBlock(nn.Module):
    def __init__(self, ch: int, k: int = 3, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        pad = (k - 1) // 2 * dilation
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, k, padding=pad, dilation=dilation),
            nn.BatchNorm1d(ch),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(ch, ch, k, padding=pad, dilation=dilation),
            nn.BatchNorm1d(ch),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class CNN1DBackbone(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        n_blocks: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_out = d_model
        # dilations grow to widen the receptive field over the token sequence
        self.blocks = nn.ModuleList(
            [_ResBlock(d_model, kernel_size, dilation=2**i, dropout=dropout) for i in range(n_blocks)]
        )
        self.norm = nn.BatchNorm1d(d_model)

    def forward_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """``[B, T, d_model] -> [B, T, d_model]`` per-token features (for SSL/MAE)."""
        x = tokens.transpose(1, 2)  # [B, d_model, n_tokens]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x.transpose(1, 2)     # [B, T, d_model]

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """``[B, n_tokens, d_model] -> [B, d_model]`` (global avg pool)."""
        return self.forward_tokens(tokens).mean(dim=1)
