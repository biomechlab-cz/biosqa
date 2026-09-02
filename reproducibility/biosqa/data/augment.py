"""Physiological augmentations (Plan 1 §8.4).

For a *quality* task, augmentations must **preserve the quality label** — adding
strong noise/baseline-wander would turn a Q3 window into a Q1, corrupting the
target. So we use only conservative, quality-preserving transforms: mild
amplitude scaling, small time shifts, tiny jitter, and per-lead dropout (a
multi-lead robustness regularizer). Aggressive noise/warp transforms are
deliberately omitted here (they belong to SSL pretraining, not supervised SQA).
Applied to TRAINING data only.
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset

__all__ = ["RandomAugment", "AugmentedDataset"]


class RandomAugment:
    """Compose of label-preserving augmentations on a ``[C, L]`` tensor."""

    def __init__(
        self,
        p_scale: float = 0.5, scale_range: tuple[float, float] = (0.9, 1.1),
        p_shift: float = 0.5, max_shift_frac: float = 0.08,
        p_jitter: float = 0.3, jitter_sigma: float = 0.02,
        p_lead_dropout: float = 0.2, lead_dropout_frac: float = 0.15,
    ):
        self.p_scale, self.scale_range = p_scale, scale_range
        self.p_shift, self.max_shift_frac = p_shift, max_shift_frac
        self.p_jitter, self.jitter_sigma = p_jitter, jitter_sigma
        self.p_lead_dropout, self.lead_dropout_frac = p_lead_dropout, lead_dropout_frac

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        C, L = x.shape
        if torch.rand(1).item() < self.p_scale:
            lo, hi = self.scale_range
            x = x * (lo + (hi - lo) * torch.rand(1).item())
        if torch.rand(1).item() < self.p_shift:
            shift = int((torch.rand(1).item() * 2 - 1) * self.max_shift_frac * L)
            x = torch.roll(x, shifts=shift, dims=-1)
        if torch.rand(1).item() < self.p_jitter:
            x = x + torch.randn_like(x) * self.jitter_sigma
        if C > 1 and torch.rand(1).item() < self.p_lead_dropout:
            k = max(1, int(self.lead_dropout_frac * C))
            leads = torch.randperm(C)[:k]
            x = x.clone()
            x[leads] = 0.0
        return x


class AugmentedDataset(Dataset):
    """Wrap ``(X, y)`` arrays/tensors and apply ``augment`` on-the-fly (train only)."""

    def __init__(self, X, y, augment: RandomAugment | None = None):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.X[i]
        if self.augment is not None:
            x = self.augment(x)
        return x, self.y[i]
