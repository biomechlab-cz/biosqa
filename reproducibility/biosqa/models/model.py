"""Assemble adapter + shared backbone + per-modality heads (Plan 1 §7.1).

``BioSQAModel`` is flexible enough to be BOTH sides of the C1 comparison:
* **shared**  — instantiate with several modalities: one backbone, per-modality
  adapters/heads/modality-embedding.
* **specialist** — instantiate with a single modality: same class, one adapter +
  its own backbone + one head. Four specialists = four single-modality models.

Modality routing happens in Python (select the adapter/head by name), so an
exported per-modality graph is a static adapter->backbone->head chain with the
modality embedding folded in as a constant — clean for ONNX (Plan 1 §14).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .adapters import PatchAdapter
from .backbones.cnn1d import CNN1DBackbone
from .backbones.patchtst import PatchTSTBackbone
from .heads import QualityHead
from .pooling import build_pool

__all__ = ["BioSQAModel", "SingleModalityExport", "MultiHeadExport", "build_backbone", "build_model"]


def _filter_kwargs(cls, cfg: dict) -> dict:
    """Keep only kwargs the backbone's __init__ accepts. Lets ``base.yaml`` carry
    a superset ``backbone_cfg`` (OmegaConf deep-merges it across backbones) without
    leaking e.g. transformer ``n_heads`` into the CNN backbone."""
    import inspect

    accepted = set(inspect.signature(cls.__init__).parameters) - {"self"}
    return {k: v for k, v in cfg.items() if k in accepted}


def build_backbone(name: str, d_model: int, **cfg) -> nn.Module:
    name = name.lower()
    if name in ("cnn1d", "cnn"):
        return CNN1DBackbone(d_model=d_model, **_filter_kwargs(CNN1DBackbone, cfg))
    if name in ("patchtst", "transformer"):
        return PatchTSTBackbone(d_model=d_model, **_filter_kwargs(PatchTSTBackbone, cfg))
    if name in ("mamba", "ssm"):
        from .backbones.mamba_backbone import MambaBackbone  # lazy: research arm

        return MambaBackbone(d_model=d_model, **_filter_kwargs(MambaBackbone, cfg))
    raise ValueError(f"unknown backbone '{name}'")


class BioSQAModel(nn.Module):
    """Shared-backbone-with-adapters model (or a single-modality specialist)."""

    def __init__(
        self,
        modalities: dict[str, dict],   # name -> {patch_len, stride, c_in, max_tokens}
        backbone: str = "patchtst",
        d_model: int = 128,
        n_classes: int = 4,
        n_glitch: int = 0,
        n_binary: int = 0,
        ordinal_grade: bool = False,
        head_hidden: int | None = None,
        dropout: float = 0.1,
        backbone_cfg: dict | None = None,
        use_experts: bool = False,
        expert_backbone: str = "cnn1d",
        expert_cfg: dict | None = None,
        pool: str = "mean",
        mixstyle: bool | dict = False,
    ):
        super().__init__()
        self.modality_names = list(modalities.keys())
        self.d_model = d_model
        self.n_classes = n_classes
        self.use_experts = use_experts

        self.adapters = nn.ModuleDict(
            {
                m: PatchAdapter(
                    patch_len=cfg["patch_len"],
                    stride=cfg["stride"],
                    d_model=d_model,
                    c_in=cfg.get("c_in", 1),
                    max_tokens=cfg.get("max_tokens", 128),
                    dropout=dropout,
                )
                for m, cfg in modalities.items()
            }
        )
        self.modality_embed = nn.Embedding(len(self.modality_names), d_model)
        nn.init.trunc_normal_(self.modality_embed.weight, std=0.02)
        self._mod_index = {m: i for i, m in enumerate(self.modality_names)}

        self.backbone = build_backbone(backbone, d_model, **(backbone_cfg or {}))
        self.pool = build_pool(pool, d_model)   # mean (default, == old behaviour) or gated-attn MIL
        # optional train-only MixStyle DG regulariser after the tokenizer (identity
        # at eval -> exported graph unchanged). Off by default = old behaviour.
        self.mixstyle = None
        if mixstyle:
            from .mixstyle import MixStyle

            self.mixstyle = MixStyle(**(mixstyle if isinstance(mixstyle, dict) else {}))
        self.heads = nn.ModuleDict(
            {
                m: QualityHead(d_model, n_classes, head_hidden, dropout, n_glitch, n_binary,
                               ordinal_grade=ordinal_grade)
                for m in self.modality_names
            }
        )

        # C1 fix: optional per-modality expert branch + learned gate. Lets the
        # distinct modality (EDA) keep dedicated capacity while the rest share.
        if use_experts:
            ecfg = expert_cfg or {"n_blocks": 1}
            self.experts = nn.ModuleDict(
                {m: build_backbone(expert_backbone, d_model, **ecfg) for m in self.modality_names}
            )
            # gate logit per modality; sigmoid(0)=0.5 init -> learns reliance on expert
            self.expert_gate = nn.ParameterDict(
                {m: nn.Parameter(torch.zeros(())) for m in self.modality_names}
            )

    def encode(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        tokens = self.adapters[modality](x)
        idx = torch.tensor(self._mod_index[modality], device=x.device)
        tokens = tokens + self.modality_embed(idx)  # broadcast [d_model] over tokens
        if self.mixstyle is not None:               # train-only; identity at eval
            tokens = self.mixstyle(tokens)
        shared = self.pool(self.backbone.forward_tokens(tokens))   # [B, d_model]
        if self.use_experts:
            g = torch.sigmoid(self.expert_gate[modality])
            expert = self.pool(self.experts[modality].forward_tokens(tokens))
            return (1.0 - g) * shared + g * expert
        return shared

    def gate_values(self) -> dict[str, float]:
        """Learned expert-reliance gate per modality (interpretable C1 result)."""
        if not self.use_experts:
            return {}
        return {m: float(torch.sigmoid(self.expert_gate[m]).item()) for m in self.modality_names}

    def forward(self, x: torch.Tensor, modality: str):
        pooled = self.encode(x, modality)
        return self.heads[modality](pooled)         # (q_logits, glitch_logits|None)

    def forward_multitask(self, x: torch.Tensor, modality: str) -> dict:
        """All three levels at once: ``{"q", "binary", "type"}`` (None where the
        head is absent). Used by the multi-task trainer and the export wrapper."""
        pooled = self.encode(x, modality)
        return self.heads[modality].forward_all(pooled)

    def forward_dense(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        """DENSE per-token quality (fine-grained SQA): ``[B, 1, L] -> [B, T, n_classes]``.
        Reuses backbone.forward_tokens + the (per-token-broadcasting) QualityHead, so
        each token gives a sub-window quality label (patch-resolution). No pooling."""
        tokens = self.adapters[modality](x)
        idx = torch.tensor(self._mod_index[modality], device=x.device)
        tokens = tokens + self.modality_embed(idx)
        feats = self.backbone.forward_tokens(tokens)    # [B, T, d_model]
        q_logits, _ = self.heads[modality](feats)       # QualityHead broadcasts -> [B, T, C]
        return q_logits


class SingleModalityExport(nn.Module):
    """Wrap a model + fixed modality into a single-input, single-output module
    for ONNX export (deployment contract: ``[B, 1, L] -> Q0..Q3 logits``)."""

    def __init__(self, model: BioSQAModel, modality: str, return_glitch: bool = False):
        super().__init__()
        self.model = model
        self.modality = modality
        self.return_glitch = return_glitch

    def forward(self, x: torch.Tensor):
        q, g = self.model(x, self.modality)
        if self.return_glitch and g is not None:
            return q, g
        return q


class MultiHeadExport(nn.Module):
    """Wrap a model + fixed modality into the MULTI-HEAD deployment graph:
    ``[B, 1, L] -> (q_logits[, bin_logits][, type_logits])`` as named ONNX
    outputs of RAW logits (the app applies softmax/sigmoid per head). Only the
    heads present on the model are emitted; ``output_order`` fixes their order.
    """

    def __init__(self, model: "BioSQAModel", modality: str,
                 output_order: tuple[str, ...] = ("q", "binary", "type"),
                 instance_norm: bool = True, temperature: dict | None = None):
        super().__init__()
        self.model = model
        self.modality = modality
        self.instance_norm = instance_norm     # per-window z-score baked into the graph
        self.temperature = temperature or {}   # head-name -> T (calibration baked in)
        present = model.heads[modality].forward_all(torch.zeros(1, model.d_model))
        self.active = tuple(h for h in output_order if present.get(h) is not None)

    def forward(self, x: torch.Tensor):
        if self.instance_norm:                  # normalize INSIDE the graph -> no deploy drift
            x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)
        out = self.model.forward_multitask(x, self.modality)
        vals = []
        for h in self.active:                   # temperature-scale logits so app softmax is calibrated
            v = out[h]
            T = self.temperature.get(h)
            if T is not None and T != 1.0:
                v = v / T
            vals.append(v)
        return tuple(vals) if len(vals) > 1 else vals[0]


def build_model(cfg) -> BioSQAModel:
    """Build a :class:`BioSQAModel` from an OmegaConf/dict config.

    Expects ``cfg.model`` (backbone/d_model/n_classes/...) and ``cfg.modalities``
    (name -> adapter geometry). If ``cfg.model.modalities`` lists a subset, only
    those are built (used to make single-modality specialists from one config).
    """
    from omegaconf import OmegaConf

    c = OmegaConf.to_container(cfg, resolve=True) if not isinstance(cfg, dict) else cfg
    mcfg = c["model"]
    all_mods = c["modalities"]
    subset = mcfg.get("modalities") or list(all_mods.keys())
    mods = {m: all_mods[m] for m in subset}
    return BioSQAModel(
        modalities=mods,
        backbone=mcfg.get("backbone", "patchtst"),
        d_model=mcfg.get("d_model", 128),
        n_classes=mcfg.get("n_classes", 4),
        n_glitch=mcfg.get("n_glitch", 0),
        n_binary=mcfg.get("n_binary", 0),
        ordinal_grade=mcfg.get("ordinal_grade", False),
        head_hidden=mcfg.get("head_hidden"),
        dropout=mcfg.get("dropout", 0.1),
        backbone_cfg=mcfg.get("backbone_cfg", {}),
        use_experts=mcfg.get("use_experts", False),
        expert_backbone=mcfg.get("expert_backbone", "cnn1d"),
        expert_cfg=mcfg.get("expert_cfg"),
        pool=mcfg.get("pool", "mean"),
        mixstyle=mcfg.get("mixstyle", False),
    )
