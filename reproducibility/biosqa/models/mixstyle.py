"""MixStyle feature-statistic perturbation for cross-cohort DG (research slate
quick-win #2; Zhou et al., "Domain Generalization with MixStyle", ICLR 2021).

Cross-cohort shift in biosignals is dominated by amplitude/gain/baseline "style".
MixStyle mixes the per-channel first/second moments (mean, std over the token/time
axis) between two samples in the batch, synthesising unseen cohort styles while
preserving the shape/content the quality label depends on. It is a **train-time
regulariser only**: at ``eval()`` it is the identity, so the exported ONNX graph
is byte-identical to the plain trunk (no MixStyle op is ever traced).

Note the interaction flagged by the SOTA sweep: the trunk already applies per-
window instance-norm, which strips first-order input style, so MixStyle's marginal
headroom is smaller than on vision benchmarks — it must be judged on LODO folds,
not assumed. Insert immediately after the patch tokenizer (channel = d_model,
axis = token sequence).
"""
from __future__ import annotations

import random

import torch
import torch.nn as nn

__all__ = ["MixStyle"]


class MixStyle(nn.Module):
    """Mix per-channel (mean, std) statistics over the token axis between batch pairs.

    Parameters
    ----------
    p : probability of applying MixStyle to a given batch (else identity).
    alpha : Beta(alpha, alpha) shape for the mixing coefficient lambda.
    eps : numerical floor on the std.
    mix : ``"random"`` shuffles the batch (domain-agnostic, the ICLR-2021 default).
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6, mix: str = "random"):
        super().__init__()
        self.p = float(p)
        self.eps = float(eps)
        self.mix = mix
        self._beta = torch.distributions.Beta(alpha, alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D] token sequence. Identity at eval / with prob (1-p) / tiny batch
        # -> the exported (eval-mode) graph never contains a MixStyle op.
        if not self.training or x.size(0) < 2 or random.random() > self.p:
            return x
        mu = x.mean(dim=1, keepdim=True)                                   # [B, 1, D]
        sig = (x.var(dim=1, keepdim=True, unbiased=False) + self.eps).sqrt()
        x_norm = (x - mu) / sig
        lam = self._beta.sample((x.size(0), 1, 1)).to(x.device, x.dtype)   # [B, 1, 1]
        perm = torch.randperm(x.size(0), device=x.device)
        mu_mix = lam * mu + (1.0 - lam) * mu[perm]
        sig_mix = lam * sig + (1.0 - lam) * sig[perm]
        return x_norm * sig_mix + mu_mix
