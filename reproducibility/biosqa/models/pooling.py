"""Token pooling for the window-level head (Architecture Exp 2/1).

Replaces the fixed ``tokens.mean(1)`` with a learned **gated-attention MIL pool**
(Ilse, Tomczak & Welling, ICML 2018): the window logit becomes a quality-weighted
sum over tokens, so locally-confined artifacts (motion bursts, electrode pop) get
the weight they deserve instead of being averaged away. One tiny Linear stack +
softmax over the token axis — microseconds on CPU and fully ONNX-exportable.
The attention weights double as a free unsupervised dense-quality proxy.
"""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["MeanPool", "AttnPool1d", "build_pool"]


class MeanPool(nn.Module):
    def forward(self, feats: torch.Tensor, return_weights: bool = False):
        pooled = feats.mean(dim=1)
        if return_weights:
            T = feats.shape[1]
            w = torch.full(feats.shape[:2], 1.0 / T, device=feats.device)
            return pooled, w
        return pooled


class AttnPool1d(nn.Module):
    """Gated-attention MIL pool over the token axis. ``[B, T, D] -> [B, D]``."""

    def __init__(self, d_model: int, hidden: int | None = None, gated: bool = True):
        super().__init__()
        h = hidden or max(16, d_model // 2)
        self.V = nn.Linear(d_model, h)
        self.U = nn.Linear(d_model, h) if gated else None
        self.w = nn.Linear(h, 1)
        self.gated = gated

    def forward(self, feats: torch.Tensor, return_weights: bool = False):
        a = torch.tanh(self.V(feats))
        if self.gated:
            a = a * torch.sigmoid(self.U(feats))
        scores = self.w(a).squeeze(-1)              # [B, T]
        attn = torch.softmax(scores, dim=1)         # [B, T]
        pooled = (attn.unsqueeze(-1) * feats).sum(dim=1)
        return (pooled, attn) if return_weights else pooled


def build_pool(name: str, d_model: int) -> nn.Module:
    name = (name or "mean").lower()
    if name == "mean":
        return MeanPool()
    if name in ("attn", "attention", "mil"):
        return AttnPool1d(d_model, gated=True)
    raise ValueError(f"unknown pool '{name}'")
