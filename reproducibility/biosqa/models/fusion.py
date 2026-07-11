"""Dual-input SQA: deep 1D branch + hand-crafted SQI-feature branch (the CMPB
archetype, e.g. Liu et al. 2021 CMPB 208:106269 — hand-crafted stats + deep branch).

A deep branch (multi-lead PatchAdapter -> backbone -> pool) captures learned
morphology; a small MLP branch ingests per-lead SQI features (kurtosis, skewness,
band-SNR, flatline, saturation, HF-ratio); the two d_model vectors are fused and
classified. Stays 1D + exportable (two ONNX inputs: signal + sqi vector).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .adapters import PatchAdapter
from .heads import QualityHead
from .model import build_backbone
from .pooling import build_pool

__all__ = ["DualInputSQAModel", "DualBranchMultiHead", "DualBranchExport",
           "FeatureFusionMultiHead", "FeatureFusionExport"]


class DualBranchMultiHead(nn.Module):
    """Two-branch multi-head SQA model that ROUTES spectral features per head
    (arch-search 2026-07-05). A raw-signal branch and a precomputed-spectral-channel
    branch each produce a ``d_model`` embedding; the ordinal GRADE head reads the raw
    branch ONLY (spectral regresses in-distribution grade), while the USABLE and
    artifact-TYPE heads read the FUSED (raw+spectral) embedding (spectral lifts type
    macro-F1 and cross-cohort usable-AUROC). Exports as a two-input legacy-ONNX graph:
    ``(x_raw [B,1,L], x_spec [B,C,L]) -> q_logits, bin_logits, type_logits``; the app
    computes ``x_spec`` in numpy (spectral_band_channels) — no in-graph FFT.
    """

    def __init__(self, patch_len: int, stride: int, n_spec_ch: int, *, d_model: int = 128,
                 n_classes: int = 4, n_type: int = 10, backbone_cfg: dict | None = None,
                 max_tokens: int = 64, dropout: float = 0.1, ordinal_grade: bool = True,
                 spec_blocks: int = 2):
        super().__init__()
        bcfg = backbone_cfg or {"n_blocks": 4}
        self.raw_adapter = PatchAdapter(patch_len, stride, d_model, c_in=1, max_tokens=max_tokens, dropout=dropout)
        self.raw_backbone = build_backbone("cnn1d", d_model, **bcfg)
        self.spec_adapter = PatchAdapter(patch_len, stride, d_model, c_in=n_spec_ch, max_tokens=max_tokens, dropout=dropout)
        self.spec_backbone = build_backbone("cnn1d", d_model, n_blocks=spec_blocks)
        self.pool = build_pool("mean", d_model)
        # GRADE: raw-only ordinal head (reuse QualityHead's ordinal path, grade output only)
        self.grade_head = QualityHead(d_model, n_classes, dropout=dropout, ordinal_grade=ordinal_grade)
        # USABLE + TYPE: on the fused 2*d_model embedding
        self.usable_head = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 2))
        self.type_head = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, n_type))

    def _embed(self, x_raw, x_spec):
        rf = self.pool(self.raw_backbone.forward_tokens(self.raw_adapter(x_raw)))
        sf = self.pool(self.spec_backbone.forward_tokens(self.spec_adapter(x_spec)))
        return rf, torch.cat([rf, sf], dim=-1)

    def forward_multitask(self, x_raw, x_spec, modality=None) -> dict:
        rf, fused = self._embed(x_raw, x_spec)
        return {"q": self.grade_head._q(self.grade_head.mlp(rf)),   # raw-only grade
                "binary": self.usable_head(fused),
                "type": self.type_head(fused)}

    def forward(self, x_raw, x_spec):
        out = self.forward_multitask(x_raw, x_spec)
        return out["q"], out["binary"], out["type"]


class DualBranchExport(nn.Module):
    """Two-input deployment wrapper for :class:`DualBranchMultiHead`. Bakes per-window
    per-channel instance-norm on BOTH inputs and optional temperature scaling into the
    graph, emitting ``(q_logits, bin_logits, type_logits)`` as RAW logits (q is already
    log-probs from the ordinal head, so the app's softmax recovers grade probabilities).
    ONNX inputs: ``x_raw [B,1,L]`` and ``x_spec [B,C,L]`` (the app computes ``x_spec``
    with :func:`biosqa.data.signal_channels.spectral_band_channels` in numpy — no
    in-graph FFT). Dynamic batch on both inputs.
    """

    def __init__(self, model: DualBranchMultiHead, temperature: dict | None = None):
        super().__init__()
        self.model = model
        self.temperature = temperature or {}

    @staticmethod
    def _inorm(x):
        return (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)

    def forward(self, x_raw, x_spec):
        out = self.model.forward_multitask(self._inorm(x_raw), self._inorm(x_spec))
        vals = []
        for name, key in (("q", "q"), ("binary", "binary"), ("type", "type")):
            v = out[key]
            T = self.temperature.get(name)
            if T is not None and T != 1.0:
                v = v / T
            vals.append(v)
        return tuple(vals)


class FeatureFusionMultiHead(nn.Module):
    """Raw 1D trunk fused with a host-precomputed SQI-feature VECTOR -> ordinal GRADE +
    USABLE heads (PPG deployable, campaign 2026-07-06). Distinct from DualBranchMultiHead
    (which fuses spectral CHANNELS): here the 2nd input is a small SCALE-INVARIANT scalar
    vector (e.g. biosqa.data.sqa_features.ppg_sqi_vector: skew/kurt/cardiac-ratio/pulse-
    regularity/...), passed through an MLP and concatenated with the pooled trunk embedding.
    Both heads read the fused embedding; the grade head is the ordinal ordered-logit head
    (helps cross-cohort). Exports as a two-input legacy-ONNX graph
    ``(x_raw [B,1,L], x_feat [B,D]) -> q_logits, bin_logits``.
    """

    def __init__(self, patch_len: int, stride: int, n_feat: int, *, d_model: int = 128,
                 n_classes: int = 4, backbone_cfg: dict | None = None, max_tokens: int = 64,
                 dropout: float = 0.1, ordinal_grade: bool = True, n_type: int = 0):
        super().__init__()
        bcfg = backbone_cfg or {"n_blocks": 4}
        self.adapter = PatchAdapter(patch_len, stride, d_model, c_in=1, max_tokens=max_tokens, dropout=dropout)
        self.backbone = build_backbone("cnn1d", d_model, **bcfg)
        self.pool = build_pool("mean", d_model)
        self.f_mlp = nn.Sequential(nn.Linear(n_feat, d_model), nn.GELU(), nn.Dropout(dropout),
                                   nn.Linear(d_model, d_model), nn.GELU())
        self.grade_head = QualityHead(2 * d_model, n_classes, dropout=dropout, ordinal_grade=ordinal_grade)
        self.usable_head = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                                         nn.Linear(d_model, 2))
        # optional multilabel artifact-TYPE head (level-3); None -> 2-head model
        self.n_type = n_type
        self.type_head = (nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                                        nn.Linear(d_model, n_type)) if n_type else None)

    def _embed(self, x_raw, x_feat):
        rf = self.pool(self.backbone.forward_tokens(self.adapter(x_raw)))
        return torch.cat([rf, self.f_mlp(x_feat)], dim=-1)

    def forward_multitask(self, x_raw, x_feat, modality=None) -> dict:
        h = self._embed(x_raw, x_feat)
        out = {"q": self.grade_head._q(self.grade_head.mlp(h)), "binary": self.usable_head(h)}
        if self.type_head is not None:
            out["type"] = self.type_head(h)
        return out

    def forward(self, x_raw, x_feat):
        out = self.forward_multitask(x_raw, x_feat)
        if self.type_head is not None:
            return out["q"], out["binary"], out["type"]
        return out["q"], out["binary"]


class FeatureFusionExport(nn.Module):
    """Two-input deployment wrapper for :class:`FeatureFusionMultiHead`. Bakes per-window
    instance-norm on ``x_raw`` AND the training feature-standardization (mean/std) on
    ``x_feat`` into the graph, so the app feeds the RAW SQI vector it computed in numpy.
    Emits ``(q_logits, bin_logits)``; ``q`` is already log-probs from the ordinal head, so
    the app's softmax recovers grade probabilities. Optional per-head temperature scaling.
    """

    def __init__(self, model: FeatureFusionMultiHead, feat_mean, feat_std, temperature: dict | None = None):
        super().__init__()
        self.model = model
        self.register_buffer("feat_mean", torch.as_tensor(feat_mean, dtype=torch.float32))
        self.register_buffer("feat_std", torch.as_tensor(feat_std, dtype=torch.float32))
        self.temperature = temperature or {}

    @staticmethod
    def _inorm(x):
        return (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)

    def forward(self, x_raw, x_feat):
        xf = (x_feat - self.feat_mean) / (self.feat_std + 1e-6)
        out = self.model.forward_multitask(self._inorm(x_raw), xf)
        # emit (q, binary[, type]) — type head only when the model has one; the app
        # applies sigmoid to type_logits (multilabel), softmax to q/binary.
        names = ("q", "binary", "type") if "type" in out else ("q", "binary")
        vals = []
        for name in names:
            v = out[name]
            T = self.temperature.get(name)
            if T is not None and T != 1.0:
                v = v / T
            vals.append(v)
        return tuple(vals)


class DualInputSQAModel(nn.Module):
    def __init__(
        self,
        patch_len: int,
        stride: int,
        n_sqi: int,
        c_in: int = 12,
        d_model: int = 128,
        n_classes: int = 2,
        backbone: str = "cnn1d",
        backbone_cfg: dict | None = None,
        pool: str = "attn",
        dropout: float = 0.1,
        max_tokens: int = 64,
    ):
        super().__init__()
        self.adapter = PatchAdapter(patch_len, stride, d_model, c_in=c_in, max_tokens=max_tokens, dropout=dropout)
        self.backbone = build_backbone(backbone, d_model, **(backbone_cfg or {}))
        self.pool = build_pool(pool, d_model)
        self.sqi_mlp = nn.Sequential(
            nn.Linear(n_sqi, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, d_model),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, n_classes),
        )

    def forward(self, x: torch.Tensor, sqi: torch.Tensor) -> torch.Tensor:
        feats = self.pool(self.backbone.forward_tokens(self.adapter(x)))   # [B, d_model]
        s = self.sqi_mlp(sqi)                                              # [B, d_model]
        return self.classifier(torch.cat([feats, s], dim=-1))             # [B, n_classes]
