"""Per-modality quality heads (Plan 1 §7.2).

Deliberately small (2-layer MLP) so most model capacity lives in the shared
backbone — this is what makes the shared-vs-specialist comparison (C1) fair.

Supports the project's THREE classification levels off one shared trunk:
* ``q_out``       — ordinal Q0..Q3 grade (the primary head).
* ``bin_out``     — binary usable/unusable (OK/BAD), a directly-calibrated
                    2-way head (Plan 2 surfaces the OK/BAD badge).
* ``glitch_out``  — artifact-TYPE head (Plan 1 §12.1, LIGO/Gravity-Spy glitch
                    taxonomy multi-task), MULTI-LABEL (independent sigmoids;
                    artifacts co-occur), which regularizes the Q task and gives
                    the app its explainability. ``n_glitch`` == len(ARTIFACT_TYPES).

The 2-tuple ``forward`` is preserved for the frozen single-task harness
(:mod:`biosqa.train.loop`); multi-task callers use :meth:`forward_all`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["QualityHead"]


class QualityHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_classes: int = 4,
        hidden: int | None = None,
        dropout: float = 0.1,
        n_glitch: int = 0,
        n_binary: int = 0,
        ordinal_grade: bool = False,
    ):
        super().__init__()
        hidden = hidden or d_model
        self.n_classes = n_classes
        self.ordinal_grade = ordinal_grade
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if ordinal_grade:
            # ORDERED-LOGIT (proportional-odds) grade head: a scalar latent quality
            # axis + K-1 learnable ordered cutpoints (arch-search 2026-07-05: beats
            # nominal softmax cross-cohort, no in-dist regression on ECG/EEG). The
            # head EMITS LOG-PROBABILITIES as q_logits so downstream softmax recovers
            # the ordinal class probabilities and argmax gives the grade -> the ONNX
            # output contract, temperature scaling, and app decode are all UNCHANGED,
            # and CE/SORD losses compose (log_softmax is idempotent on log-probs).
            self.q_out = nn.Linear(hidden, 1)
            self.cut_base = nn.Parameter(torch.tensor(-1.0))
            self.cut_gap = nn.Parameter(torch.zeros(n_classes - 2))  # softplus -> positive gaps
        else:
            self.q_out = nn.Linear(hidden, n_classes)
        self.glitch_out = nn.Linear(hidden, n_glitch) if n_glitch > 0 else None
        self.bin_out = nn.Linear(hidden, n_binary) if n_binary > 0 else None

    def _q(self, h: torch.Tensor) -> torch.Tensor:
        """Grade output: 4-way logits (nominal) or 4-way LOG-PROBS (ordinal head).
        Broadcasts over any leading dims (window-level or per-token dense use)."""
        if not self.ordinal_grade:
            return self.q_out(h)
        z = self.q_out(h).squeeze(-1)                              # scalar latent [...]
        sp = F.softplus(self.cut_gap)                             # [K-2] positive gaps
        theta = self.cut_base + torch.cat(
            [torch.zeros(1, device=sp.device, dtype=sp.dtype), torch.cumsum(sp, 0)])  # [K-1] ordered
        s = torch.sigmoid(z.unsqueeze(-1) - theta)               # [..., K-1] = P(y>k)
        p = torch.cat([1 - s[..., :1], s[..., :-1] - s[..., 1:], s[..., -1:]], dim=-1)  # [..., K]
        return torch.log(p.clamp(min=1e-6))                      # log-probs (softmax recovers p)

    def forward(self, pooled: torch.Tensor):
        """``[B, d_model] -> (q_logits [B, n_classes], glitch_logits | None)``.

        Backward-compatible 2-tuple used by the single-task training loop.
        """
        h = self.mlp(pooled)
        glitch_logits = self.glitch_out(h) if self.glitch_out is not None else None
        return self._q(h), glitch_logits

    def forward_all(self, pooled: torch.Tensor) -> dict:
        """All heads at once: ``{"q", "binary", "type"}`` (value None if the head
        is absent). Broadcasts over any leading dims (per-token dense use too)."""
        h = self.mlp(pooled)
        return {
            "q": self._q(h),
            "binary": self.bin_out(h) if self.bin_out is not None else None,
            "type": self.glitch_out(h) if self.glitch_out is not None else None,
        }
