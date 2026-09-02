"""Fine-grained overlapping-window SQA -> dense quality track -> intervals.

Slides a window-level model across a long signal with OVERLAP (stride < window),
averages per-window class probabilities onto every covered sample (overlap-add),
and RLE-encodes the resulting per-sample label track into quality intervals
``(t_start, t_end, q_level, confidence)`` — the Plan 2 §7.3 segmentation contract.
Also evaluates the dense track against per-sample ground truth (e.g. EDABE) with
the frozen metric harness.
"""
from __future__ import annotations

import numpy as np
import torch

from .metrics import evaluate

__all__ = ["quality_track", "rle_intervals", "dense_eval"]


@torch.no_grad()
def quality_track(
    model,
    signal: np.ndarray,          # [L] single channel at canonical fs
    modality: str,
    *,
    window_len: int,
    stride: int,
    device: str = "cpu",
    batch: int = 256,
    zscore_windows: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(prob_track [L, C], label_track [L])`` — per-sample class
    probabilities from overlap-averaging all windows covering each sample."""
    model.eval().to(device)
    L = len(signal)
    if L < window_len:
        signal = np.pad(signal, (0, window_len - L))
        L = window_len
    starts = list(range(0, L - window_len + 1, stride))
    if starts[-1] != L - window_len:
        starts.append(L - window_len)  # ensure the tail is covered

    # infer class count from a dry window
    x0 = torch.from_numpy(signal[None, None, :window_len].astype(np.float32)).to(device)
    C = model(x0, modality)[0].shape[-1]
    prob_sum = np.zeros((L, C), dtype=np.float64)
    count = np.zeros(L, dtype=np.float64)

    for i in range(0, len(starts), batch):
        chunk = starts[i:i + batch]
        wins = np.stack([signal[s:s + window_len] for s in chunk]).astype(np.float32)
        if zscore_windows:
            mu = wins.mean(1, keepdims=True); sd = wins.std(1, keepdims=True) + 1e-6
            wins = (wins - mu) / sd
        xb = torch.from_numpy(wins[:, None, :]).to(device)
        probs = torch.softmax(model(xb, modality)[0].float(), dim=-1).cpu().numpy()
        for s, p in zip(chunk, probs):
            prob_sum[s:s + window_len] += p          # broadcast window prob to its samples
            count[s:s + window_len] += 1.0
    count = np.clip(count, 1.0, None)
    prob_track = prob_sum / count[:, None]
    return prob_track.astype(np.float32), prob_track.argmax(1).astype(np.int64)


def rle_intervals(label_track: np.ndarray, fs: float, prob_track: np.ndarray | None = None,
                  min_dur_s: float = 0.0) -> list[dict]:
    """Run-length-encode a per-sample label track into quality intervals."""
    L = len(label_track)
    out, start = [], 0
    for i in range(1, L + 1):
        if i == L or label_track[i] != label_track[start]:
            t0, t1 = start / fs, i / fs
            if t1 - t0 >= min_dur_s:
                conf = float(prob_track[start:i, label_track[start]].mean()) if prob_track is not None else 1.0
                out.append({"t_start": t0, "t_end": t1, "q_level": int(label_track[start]), "confidence": conf})
            start = i
    return out


def dense_eval(label_track: np.ndarray, gt_per_sample: np.ndarray, labels: list[int],
               prob_track: np.ndarray | None = None) -> dict:
    """Per-sample dense SQA metrics vs ground truth (frozen harness)."""
    n = min(len(label_track), len(gt_per_sample))
    yprob = prob_track[:n] if prob_track is not None else None
    return evaluate(gt_per_sample[:n], label_track[:n], yprob, labels=labels)
