"""PatchTST-style transformer backbone — the default EXPORTABLE encoder (Plan 1 §7.2).

Channel-independent patch tokens (produced by :class:`biosqa.models.adapters.PatchAdapter`)
are encoded by a standard pre-norm transformer encoder and mean-pooled. Uses
``nn.MultiheadAttention`` explicitly (rather than a fused SDPA path) so the graph
traces cleanly to ONNX opset 17 and quantizes with ORT dynamic quantization
(ORT's own guidance for transformer backbones, Plan 1 §11.1).
"""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["PatchTSTBackbone"]


class _EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(a)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class PatchTSTBackbone(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        depth: int = 3,
        n_heads: int = 8,
        d_ff: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_out = d_model
        d_ff = d_ff or 4 * d_model
        self.layers = nn.ModuleList(
            [_EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """``[B, T, d_model] -> [B, T, d_model]`` encoded token sequence (for SSL/MAE)."""
        x = tokens
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """``[B, n_tokens, d_model] -> [B, d_model]`` (mean pool over tokens)."""
        return self.forward_tokens(tokens).mean(dim=1)
