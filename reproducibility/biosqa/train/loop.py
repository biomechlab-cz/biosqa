"""Single training entrypoint used by every model (Plan 1 §9.2).

AdamW + cosine LR with linear warmup, bf16 autocast on CUDA, early stopping on a
validation metric computed by the **frozen** evaluator (:mod:`biosqa.eval.metrics`).
``fit`` trains one modality's loaders (a specialist, or one modality-homogeneous
stream of a shared model); multi-modal interleaving is layered on top in Phase 2.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..eval.metrics import evaluate
from .losses import build_loss

__all__ = ["fit", "FitResult", "predict"]


@dataclass
class FitResult:
    best_metrics: dict
    best_epoch: int
    history: list[dict] = field(default_factory=list)
    best_state: dict | None = None


def _cosine_warmup(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def predict(model, loader: DataLoader, modality: str, device: str,
            loss: str = "ce", n_classes: int = 4):
    """Decode a head's outputs to (y_true, y_pred, y_prob). A CORN-trained head outputs rank
    thresholds P(y>j|y>=j), NOT class logits, so softmax-argmax would be wrong — use corn_decode
    for it and plain softmax otherwise."""
    model.eval()
    ordinal = (loss or "ce").lower() in ("corn", "ordinal")
    ys, preds, probs = [], [], []
    for batch in loader:
        x, y = batch[0].to(device), batch[1]
        logits, _ = model(x, modality)
        if ordinal:
            from .losses import corn_decode
            pred, p = corn_decode(logits.float(), n_classes)
            preds.append(pred.cpu().numpy())
            probs.append(p.cpu().numpy())
        else:
            p = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            preds.append(p.argmax(1))
            probs.append(p)
        ys.append(np.asarray(y))
    return (
        np.concatenate(ys),
        np.concatenate(preds),
        np.concatenate(probs),
    )


def fit(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    modality: str,
    n_classes: int = 4,
    labels: list[int] | None = None,
    device: str = "cuda",
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    warmup_frac: float = 0.05,
    loss: str = "ce",
    class_weights: torch.Tensor | None = None,
    monitor: str = "macro_f1",
    patience: int = 8,
    grad_clip: float = 1.0,
    amp: bool = True,
    log_fn=None,
) -> FitResult:
    device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
    model.to(device)
    labels = labels or list(range(n_classes))
    crit = build_loss(loss, n_classes, class_weights.to(device) if class_weights is not None else None)
    if hasattr(crit, "to"):
        crit = crit.to(device)   # SORD/KappaCE/QWK hold buffers (levels/W) that must live on `device`
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * max(1, len(train_loader))
    sched = _cosine_warmup(opt, int(warmup_frac * total_steps), total_steps)
    use_amp = amp and device == "cuda"

    best = FitResult(best_metrics={}, best_epoch=-1)
    best_score = -math.inf
    since_improved = 0

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for batch in train_loader:
            x, y = batch[0].to(device), batch[1].to(device).long()
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits, _ = model(x, modality)
                loss_val = crit(logits, y)
            loss_val.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            sched.step()
            running += float(loss_val.detach())

        y_true, y_pred, y_prob = predict(model, val_loader, modality, device,
                                         loss=loss, n_classes=n_classes)
        m = evaluate(y_true, y_pred, y_prob, labels=labels)
        m["train_loss"] = running / max(1, len(train_loader))
        m["epoch"] = epoch
        best.history.append({k: v for k, v in m.items() if not isinstance(v, (list, dict))})
        if log_fn:
            log_fn(epoch, m)

        score = m.get(monitor, -math.inf)
        if score > best_score:
            best_score = score
            best.best_metrics = m
            best.best_epoch = epoch
            best.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            since_improved = 0
        else:
            since_improved += 1
            if since_improved >= patience:
                break

    if best.best_state is not None:
        model.load_state_dict(best.best_state)
    return best
