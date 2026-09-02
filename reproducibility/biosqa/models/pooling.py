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

__all__ = ["MeanPool", "AttnPool1d", "AggMatchedPool", "build_pool"]


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


class AggMatchedPool(nn.Module):
    """Pool the token axis with the SAME rule that generated the window LABEL (Plan 10 B1).

    ``store.AGG_BY_KEY`` declares how a window grade is formed from its parts:
    ``worst`` (the minimum over constituent samples) for every ECG/PPG/EEG cohort,
    ``fraction`` for EDABE, ``burden`` for eda_artifact. The model has always mean-
    pooled regardless, which is a known misspecification -- and Plan 09 C2 showed the
    rule is not cosmetic: moving eda_artifact from ``worst`` to ``burden`` changed the
    corpus enough to collapse the manuscript's relabeling headline.

    This makes the pooling operator the differentiable image of that rule:

      worst              soft-min, ``w = softmax(-s/T)`` -- mass on the worst token,
                         which is what a minimum over parts means
      burden / fraction  ``w`` proportional to per-token badness -- extent-weighted
      mean               uniform (identical to :class:`MeanPool`)

    ``s`` is a learned per-token quality scalar (higher = better); no dense labels are
    required. The temperature is learned, and mean pooling is **approached as T grows**
    (measured: max|pooled - mean| = 7e-4 at T=403, the clamp caps T at 1e3). So if
    matching the rule does not help, the model can fall back to within numerical noise
    of the baseline -- which makes the B1 gate conservative rather than flattering, and
    means a null cannot be blamed on the baseline being unreachable.

    Distinct from :class:`AttnPool1d`, which *learns* an arbitrary attention shape.
    Here the shape is fixed by the declared label-generating process and only its
    sharpness is learned.
    """

    def __init__(self, d_model: int, rule: str = "worst", temperature: float = 1.0):
        super().__init__()
        self.rule = (rule or "worst").lower()
        self.q = nn.Linear(d_model, 1)
        self.log_t = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(temperature)))))

    def forward(self, feats: torch.Tensor, return_weights: bool = False):
        s = self.q(feats).squeeze(-1)                      # [B, T], higher = better
        t = self.log_t.exp().clamp(min=1e-3, max=1e3)
        if self.rule == "worst":
            w = torch.softmax(-s / t, dim=1)
        elif self.rule in ("burden", "fraction"):
            b = torch.sigmoid(-s / t)
            w = b / b.sum(dim=1, keepdim=True).clamp(min=1e-6)
        else:
            w = torch.full_like(s, 1.0 / s.shape[1])
        pooled = (w.unsqueeze(-1) * feats).sum(dim=1)
        return (pooled, w) if return_weights else pooled


def build_pool(name: str, d_model: int) -> nn.Module:
    name = (name or "mean").lower()
    if name == "mean":
        return MeanPool()
    if name in ("attn", "attention", "mil"):
        return AttnPool1d(d_model, gated=True)
    # Plan 10 B1: aggregation-matched pooling, named for the rule it mirrors.
    if name in ("worst", "softmin", "agg_worst"):
        return AggMatchedPool(d_model, rule="worst")
    if name in ("burden", "agg_burden"):
        return AggMatchedPool(d_model, rule="burden")
    if name in ("fraction", "agg_fraction"):
        return AggMatchedPool(d_model, rule="fraction")
    raise ValueError(f"unknown pool '{name}'")
