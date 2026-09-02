"""Multi-task trainer for the THREE classification levels off one shared trunk
(Plan 1 §7.2, §9): ordinal Q0..Q3 grade + binary usable/unusable + MULTI-LABEL
artifact-type. Kept SEPARATE from the frozen single-task :func:`biosqa.train.loop.fit`
so existing experiments are untouched.

Loss = ordinal(q) + ``bin_weight``·CE(binary) + ``type_weight``·maskedBCE(type).
The artifact-type term is masked per-sample: datasets with no native type labels
(``type_mask==0``) contribute to the quality/binary tasks but not the type task,
so heterogeneous supervision composes without polluting the type head.

Per-window instance z-score is applied here to match the exported graph
(:class:`biosqa.models.model.MultiHeadExport`, ``instance_norm=True``) — train
and deploy see identical normalization.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ..eval.metrics import evaluate
from .losses import build_loss
from .loop import _cosine_warmup

__all__ = ["fit_multitask", "predict_multitask", "make_multitask_dataset", "MultiTaskFitResult"]


def _inst_norm(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)


def make_multitask_dataset(
    X: np.ndarray, y_q: np.ndarray, y_type: np.ndarray | None, type_mask: np.ndarray | None,
    *, usable_min: int = 2, grade_mask: np.ndarray | None = None,
) -> TensorDataset:
    """Build a ``(x, y_q, y_bin, y_type, type_mask, grade_mask)`` dataset.

    ``y_bin`` is derived from the ordinal grade: usable (1) iff ``y_q >= usable_min``
    (Q2/Q3), else unusable (0). ``y_type`` is a ``[N, K]`` multi-hot float array
    (zeros where a sample has no type labels), ``type_mask`` a ``[N]`` bool.
    ``grade_mask`` [N] bool (default all-True) gates the GRADE + usable losses: set
    it False for synthetic type-augmentation windows whose proxy grade would pollute
    the real grade head (they still supervise the type head via ``type_mask``).
    """
    n = len(X)
    y_bin = (np.asarray(y_q) >= usable_min).astype(np.int64)
    if y_type is None:
        y_type = np.zeros((n, 1), dtype=np.float32)
        type_mask = np.zeros(n, dtype=bool)
    if type_mask is None:
        type_mask = np.ones(n, dtype=bool)
    if grade_mask is None:
        grade_mask = np.ones(n, dtype=bool)
    return TensorDataset(
        torch.from_numpy(X.astype(np.float32)),
        torch.from_numpy(np.asarray(y_q).astype(np.int64)),
        torch.from_numpy(y_bin),
        torch.from_numpy(np.asarray(y_type).astype(np.float32)),
        torch.from_numpy(np.asarray(type_mask).astype(bool)),
        torch.from_numpy(np.asarray(grade_mask).astype(bool)),
    )


@dataclass
class MultiTaskFitResult:
    best_metrics: dict
    best_epoch: int
    history: list[dict] = field(default_factory=list)
    best_state: dict | None = None


@torch.no_grad()
def predict_multitask(model, loader: DataLoader, modality: str, device: str, artifact_types=None,
                      q_loss: str = "ce", n_classes: int = 4) -> dict:
    """Return per-head predictions/metrics: q (frozen evaluate), binary (bal-acc/
    auroc), type (multi-label macro-F1 over masked samples). A CORN-trained Q head is rank-decoded
    (corn_decode), not softmax-argmax'd."""
    from sklearn.metrics import f1_score

    model.eval()
    q_ordinal = (q_loss or "ce").lower() in ("corn", "ordinal")
    yq, qpred, qprob = [], [], []
    ybin, binpred = [], []
    ytype_t, ytype_p, tmask_all = [], [], []
    for batch in loader:
        x, yq_b, ybin_b, ytype_b, tmask_b = batch[0], batch[1], batch[2], batch[3], batch[4]
        x = _inst_norm(x.to(device))
        out = model.forward_multitask(x, modality)
        if q_ordinal:
            from .losses import corn_decode
            pr, p = corn_decode(out["q"].float(), n_classes)
            qprob.append(p.cpu().numpy()); qpred.append(pr.cpu().numpy()); yq.append(yq_b.numpy())
        else:
            p = torch.softmax(out["q"].float(), -1).cpu().numpy()
            qprob.append(p); qpred.append(p.argmax(1)); yq.append(yq_b.numpy())
        if out["binary"] is not None:
            binpred.append(torch.softmax(out["binary"].float(), -1).argmax(-1).cpu().numpy())
            ybin.append(ybin_b.numpy())
        if out["type"] is not None:
            ytype_p.append(torch.sigmoid(out["type"].float()).cpu().numpy())
            ytype_t.append(ytype_b.numpy()); tmask_all.append(tmask_b.numpy())

    yq = np.concatenate(yq); qpred = np.concatenate(qpred); qprob = np.concatenate(qprob)
    labels = sorted(int(c) for c in np.unique(yq))
    m = {"q": evaluate(yq, qpred, qprob, labels=labels)}
    if ybin:
        ybin = np.concatenate(ybin); binpred = np.concatenate(binpred)
        m["binary"] = {
            "accuracy": float((ybin == binpred).mean()),
            "macro_f1": float(f1_score(ybin, binpred, average="macro", zero_division=0)),
        }
    if ytype_t:
        yt = np.concatenate(ytype_t); yp = np.concatenate(ytype_p); tm = np.concatenate(tmask_all)
        if tm.any():
            yt, yp = yt[tm], yp[tm]
            pred = (yp >= 0.5).astype(int)
            per = f1_score(yt.astype(int), pred, average=None, zero_division=0)
            m["type"] = {
                "macro_f1": float(f1_score(yt.astype(int), pred, average="macro", zero_division=0)),
                "micro_f1": float(f1_score(yt.astype(int), pred, average="micro", zero_division=0)),
                "n": int(tm.sum()),
            }
            if artifact_types is not None:
                m["type"]["per_class_f1"] = {t: float(v) for t, v in zip(artifact_types, per)}
    return m


def fit_multitask(
    model, train_loader: DataLoader, val_loader: DataLoader, *, modality: str,
    n_classes: int = 4, artifact_types=None, device: str = "cuda", epochs: int = 30,
    lr: float = 1e-3, weight_decay: float = 1e-2, warmup_frac: float = 0.05,
    q_loss: str = "ce", q_loss_tau: float = 2.0, class_weights: torch.Tensor | None = None,
    bin_weight: float = 0.5, type_weight: float = 0.5, type_pos_weight: torch.Tensor | None = None,
    type_crit=None,
    monitor: str = "macro_f1", patience: int = 8, grad_clip: float = 1.0, amp: bool = True, log_fn=None,
) -> MultiTaskFitResult:
    """``type_crit`` optionally overrides the default masked BCE on the artifact-
    TYPE head: a callable ``(logits[B,K], targets[B,K], mask[B]) -> scalar`` (e.g.
    :class:`biosqa.train.losses.AsymmetricLoss`). ``None`` keeps the pos-weighted
    BCE (unchanged behaviour)."""
    device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
    model.to(device)
    q_crit = build_loss(q_loss, n_classes, class_weights.to(device) if class_weights is not None else None,
                        tau=q_loss_tau)
    if hasattr(q_crit, "to"):
        q_crit = q_crit.to(device)          # move buffers (SORD levels / KappaCE W) to device
    if type_pos_weight is not None:
        type_pos_weight = type_pos_weight.to(device)
    if type_crit is not None and hasattr(type_crit, "to"):
        type_crit = type_crit.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * max(1, len(train_loader))
    sched = _cosine_warmup(opt, int(warmup_frac * total_steps), total_steps)
    use_amp = amp and device == "cuda"

    best = MultiTaskFitResult(best_metrics={}, best_epoch=-1)
    best_score, since = -math.inf, 0
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for x, yq_b, ybin_b, ytype_b, tmask_b, gmask_b in train_loader:
            x = _inst_norm(x.to(device))
            yq_b = yq_b.to(device).long(); ybin_b = ybin_b.to(device).long()
            ytype_b = ytype_b.to(device); tmask_b = tmask_b.to(device); gmask_b = gmask_b.to(device).bool()
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                out = model.forward_multitask(x, modality)
                if gmask_b.any():                                   # grade + usable on grade-masked samples only
                    loss = q_crit(out["q"][gmask_b], yq_b[gmask_b])
                    if out["binary"] is not None:
                        loss = loss + bin_weight * F.cross_entropy(out["binary"][gmask_b], ybin_b[gmask_b])
                else:
                    loss = out["q"].sum() * 0.0
                if out["type"] is not None and tmask_b.any():
                    if type_crit is not None:
                        tl = type_crit(out["type"], ytype_b, tmask_b)
                    else:
                        tl = F.binary_cross_entropy_with_logits(
                            out["type"][tmask_b], ytype_b[tmask_b],
                            pos_weight=type_pos_weight, reduction="mean")
                    loss = loss + type_weight * tl
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step(); sched.step()
            running += float(loss.detach())

        m = predict_multitask(model, val_loader, modality, device, artifact_types,
                              q_loss=q_loss, n_classes=n_classes)
        m["train_loss"] = running / max(1, len(train_loader)); m["epoch"] = epoch
        best.history.append({"epoch": epoch, "train_loss": m["train_loss"],
                             "q_macro_f1": m["q"].get("macro_f1"), "q_qwk": m["q"].get("cohen_kappa_quadratic"),
                             "bin_f1": m.get("binary", {}).get("macro_f1"),
                             "type_macro_f1": m.get("type", {}).get("macro_f1")})
        if log_fn:
            log_fn(epoch, m)
        score = m["q"].get(monitor, -math.inf)
        if score > best_score:
            best_score = score; best.best_metrics = m; best.best_epoch = epoch
            best.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            since = 0
        else:
            since += 1
            if since >= patience:
                break
    if best.best_state is not None:
        model.load_state_dict(best.best_state)
    return best
