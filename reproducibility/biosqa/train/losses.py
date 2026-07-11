"""Losses for the class-imbalanced, ordinal SQA problem (Plan 1 §9.1).

* cross-entropy (balanced case),
* class-weighted CE and **focal loss** (γ≈2) for the usual Q3≫Q0 imbalance,
* **CORN** ordinal loss — since Q0..Q3 is ordinal, misranking should cost by
  distance (Q3->Q0 worse than Q3->Q2). CORN (Shi, Cao & Raschka 2021) trains
  K-1 conditional binary classifiers; here we apply it on the same head logits
  by reinterpreting them as cumulative rank scores.
* **SORD** soft ordinal targets (Diaz & Marathe, CVPR 2019) — bleed target mass
  into ordinally-adjacent grades so a Q3->Q2 error costs less than Q3->Q0 and
  cross-cohort annotation-boundary disagreements are cheap. Target-side only:
  the exported softmax head is byte-identical to plain CE.
* **KappaCE** — CE plus a differentiable quadratic-weighted-kappa surrogate
  (de la Torre et al., Pattern Recognition Letters 2018): train the exact metric
  the frozen harness reports. Head/graph unchanged.
* **AsymmetricLoss** (Ben-Baruch et al., ICCV 2021) + effective-number class
  balancing (Cui et al., CVPR 2019) for the rare-positive multi-label artifact
  TYPE head (macro-F1 << micro-F1 is the rare-class-collapse signature).

Ordinal-family design note: SORD and KappaCE keep the 4-way softmax head, so the
decode stays argmax and the ONNX graph is unchanged. CORN reinterprets the first
K-1 logits as conditional rank scores, so it needs a *cumulative* decode
(:func:`corn_decode`) — softmax-argmax would silently sabotage it.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "FocalLoss", "CornLoss", "SORDLoss", "KappaCELoss", "AsymmetricLoss",
    "effective_number_weights", "corn_decode", "build_loss",
]


class FocalLoss(nn.Module):
    """Multiclass focal loss. ``alpha`` optionally per-class-weights."""

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha if alpha is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=-1)
        p = logp.exp()
        logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        p_t = p.gather(1, target.unsqueeze(1)).squeeze(1)
        loss = -((1 - p_t) ** self.gamma) * logp_t
        if self.alpha is not None:
            loss = loss * self.alpha.gather(0, target)
        return loss.mean()


class CornLoss(nn.Module):
    """CORN ordinal loss over ``n_classes`` ordered levels.

    Uses the first ``n_classes-1`` logits as conditional rank scores
    P(y > k | y >= k). Rank-consistent by construction and distance-aware.
    Requires the head to output at least ``n_classes-1`` logits (our
    n_classes-wide head satisfies this; the last logit is unused here).
    """

    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        k = self.n_classes - 1
        scores = logits[:, :k]                              # [B, K-1]
        losses = []
        for j in range(k):
            mask = target >= j                              # condition y >= j
            if mask.sum() == 0:
                continue
            s = scores[mask, j]
            t = (target[mask] > j).float()                  # label y > j
            losses.append(F.binary_cross_entropy_with_logits(s, t, reduction="mean"))
        if not losses:
            return logits.sum() * 0.0
        return torch.stack(losses).mean()


class SORDLoss(nn.Module):
    """Soft ORDinal targets (Diaz & Marathe, CVPR 2019).

    For true grade ``y`` the target distribution is ``t_k ∝ exp(-(y-k)^2 / tau)``
    (a soft, distance-decaying one-hot), trained with soft cross-entropy against
    the ordinary 4-way softmax head. Lower ``tau`` -> sharper (approaches one-hot
    CE as tau->0); higher ``tau`` -> more mass bled into neighbours. Optional
    ``class_weights`` reweights each sample by its true class (imbalance parity
    with class-weighted CE). Export-neutral: the head/decode are unchanged.
    """

    def __init__(self, n_classes: int, tau: float = 1.0, class_weights: torch.Tensor | None = None):
        super().__init__()
        self.tau = float(tau)
        self.register_buffer("levels", torch.arange(n_classes, dtype=torch.float32))
        self.register_buffer("alpha", class_weights if class_weights is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        d2 = (target.float().unsqueeze(1) - self.levels.unsqueeze(0)) ** 2   # [B, C]
        soft = torch.softmax(-d2 / self.tau, dim=1)                          # [B, C]
        logp = F.log_softmax(logits, dim=1)
        loss = -(soft * logp).sum(1)                                         # [B]
        if self.alpha is not None:
            loss = loss * self.alpha.gather(0, target)
        return loss.mean()


class KappaCELoss(nn.Module):
    """CE + differentiable quadratic-weighted-kappa surrogate (de la Torre 2018).

    The kappa term is the soft ratio (observed weighted disagreement)/(expected),
    which equals ``1 - QWK`` up to the additive one; minimising it drives QWK->1.
    CE is kept as a stabiliser (kappa gradients are weak near chance), so this is
    effectively a permanent CE-warmup rather than an annealed schedule. Head and
    decode unchanged -> export-neutral.
    """

    def __init__(self, n_classes: int, class_weights: torch.Tensor | None = None,
                 kappa_weight: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.n = n_classes
        self.kw = float(kappa_weight)
        self.eps = eps
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        r = torch.arange(n_classes, dtype=torch.float32)
        W = (r.view(-1, 1) - r.view(1, -1)) ** 2 / float((n_classes - 1) ** 2)  # [C, C]
        self.register_buffer("W", W)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = torch.softmax(logits, dim=1)                    # [B, C]
        ce = self.ce(logits, target)
        onehot = F.one_hot(target, self.n).float()          # [B, C]
        num = (self.W[target] * p).sum()                    # observed weighted disagreement
        hist_true = onehot.sum(0)                           # [C]
        hist_pred = p.sum(0)                                # [C]
        den = (self.W * torch.outer(hist_true, hist_pred)).sum() / target.shape[0]
        kappa_term = num / (den + self.eps)                 # ~ 1 - QWK
        return ce + self.kw * kappa_term


class AsymmetricLoss(nn.Module):
    """Asymmetric multi-label loss (Ben-Baruch et al., ICCV 2021).

    Decouples positive/negative focusing (``gamma_neg > gamma_pos``) and hard-
    negative probability shifting (``clip``) so abundant easy negatives stop
    dominating the rare-positive artifact types. ``pos_weight`` (e.g. effective-
    number class weights) further up-weights rare positives. ``mask`` (per-sample
    label-present) supports partially-typed data. Sigmoid head unchanged.
    """

    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 1.0, clip: float = 0.05,
                 pos_weight: torch.Tensor | None = None, eps: float = 1e-8):
        super().__init__()
        self.gn, self.gp, self.clip, self.eps = gamma_neg, gamma_pos, clip, eps
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        xs_pos = torch.sigmoid(logits)                      # P(label=1)
        xs_neg = 1.0 - xs_pos                               # P(label=0) = true-class prob for negatives
        if self.clip and self.clip > 0:                     # probability shift: drop very-easy negatives
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)
        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg                            # log-likelihood (<=0)
        if self.gn > 0 or self.gp > 0:                      # asymmetric focal modulation
            pt = xs_pos * targets + xs_neg * (1 - targets)  # true-class prob per element
            gamma = self.gp * targets + self.gn * (1 - targets)
            loss = loss * torch.pow(1.0 - pt, gamma)        # down-weight EASY (high-pt) examples
        loss = -loss                                        # [B, K] NLL (>=0)
        if self.pos_weight is not None:
            loss = loss * (1 + (self.pos_weight - 1) * targets)  # up-weight rare-class positives
        if mask is not None:
            loss = loss * mask.unsqueeze(1).float()
            return loss.sum() / (mask.float().sum() * loss.shape[1] + self.eps)
        return loss.mean()


def effective_number_weights(pos_counts, beta: float = 0.999, normalize: bool = True):
    """Class-balanced 'effective number' weights (Cui et al., CVPR 2019).

    ``w_c = (1 - beta) / (1 - beta**n_c)`` — down-weights common classes far less
    aggressively than inverse-frequency, which over-corrects on very rare types.
    Returns a float32 tensor aligned with ``pos_counts``.
    """
    n = np.asarray(pos_counts, dtype=np.float64)
    eff = (1.0 - np.power(beta, np.clip(n, 1.0, None))) / (1.0 - beta)
    w = 1.0 / eff
    if normalize:
        w = w * (len(w) / w.sum())
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def corn_decode(logits: torch.Tensor, n_classes: int):
    """Rank-consistent decode for a CORN-trained head.

    The first ``n_classes-1`` logits are conditional scores ``P(y>j | y>=j)``.
    Returns ``(pred_rank, probs)`` where ``pred_rank`` counts thresholds with
    cumulative prob > 0.5 and ``probs[:, k] = P(y=k)`` from the cumulative
    products (for AUROC). Using softmax-argmax here instead would be wrong.
    """
    k = n_classes - 1
    cond = torch.sigmoid(logits[:, :k])                     # [B, K-1] P(y>j | y>=j)
    cum = torch.cumprod(cond, dim=1)                        # [B, K-1] P(y>j)
    pred = (cum > 0.5).sum(dim=1)                           # rank
    probs = torch.zeros(logits.shape[0], n_classes, device=logits.device, dtype=cum.dtype)
    probs[:, 0] = 1.0 - cum[:, 0]
    for j in range(1, k):
        probs[:, j] = cum[:, j - 1] - cum[:, j]
    probs[:, k] = cum[:, k - 1]
    return pred, probs.clamp(min=0.0)


def build_loss(name: str, n_classes: int = 4, class_weights: torch.Tensor | None = None,
               gamma: float = 2.0, tau: float = 1.0, kappa_weight: float = 0.5):
    name = (name or "ce").lower()
    if name in ("ce", "cross_entropy"):
        return nn.CrossEntropyLoss(weight=class_weights)
    if name in ("focal",):
        return FocalLoss(gamma=gamma, alpha=class_weights)
    if name in ("corn", "ordinal"):
        return CornLoss(n_classes)
    if name in ("sord",):
        return SORDLoss(n_classes, tau=tau, class_weights=class_weights)
    if name in ("qwk", "kappa", "kappace"):
        return KappaCELoss(n_classes, class_weights=class_weights, kappa_weight=kappa_weight)
    raise ValueError(f"unknown loss '{name}'")
