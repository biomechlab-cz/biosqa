"""Out-of-core (streaming) inference + block-wise plot cache — so multi-day / big multichannel
recordings never hold the whole signal in RAM.

Sliding-window inference is exactly chunkable: read the analysis channel in blocks, resample each
to model rate with an overlap-save margin (kills block-edge artifacts), and carry the unconsumed
tail between blocks so windows straddling a boundary are still produced. The per-window results are
identical to processing the whole signal, but memory is bounded to one block. The plot cache is
built the same way — block-wise strided decimation, no full-length read.

Scope (honesty): the streaming path produces the primary quality segmentation (+ artifact tags).
The recoverability second pass and the false-clean integrity guard run on normal-size records only
(they need a filtered/prefiltered view of the whole signal); a streamed record shows a notice.
"""
from __future__ import annotations

import numpy as np

from biosqa.inference.preprocess import normalize_window
from biosqa.inference.segmenter import threshold_artifact_labels
from biosqa.io.loaders import read_window

#: analysis-channel sample count above which the streaming path kicks in (~9 h @ 250 Hz,
#: ~2 h @ 1 kHz). Below this the app keeps the simpler full-in-memory path unchanged.
LARGE_RECORD_SAMPLES = 8_000_000


def _resample(sig, fs_in: float, fs_out: float):
    from biosqa.workers.qt_threads import resample_signal  # local import avoids a circular import
    return resample_signal(sig, fs_in, fs_out)


def estimate_analysis_samples(handle, channel: str) -> int:
    """Native-rate sample count of the analysis channel (the streaming-vs-in-memory decision)."""
    try:
        return int(handle.n_samples[channel])
    except Exception:  # noqa: BLE001
        return 0


def stream_infer(handle, channel: str, runner, *, overlap: float = 0.0,
                 block_sec: float = 300.0):
    """Chunked sliding-window inference that never materializes the whole signal.

    Returns ``(tiers[list[str]], confidences[np.ndarray], artifacts_per_window[list[list[str]]|None],
    uncertainty[np.ndarray], stride_sec, window_sec, n_windows)`` — the exact inputs
    :func:`segmenter.run_length_encode` needs (the in-memory path carries per-window uncertainty, so the
    streamed path must too, else streamed intervals silently export uncertainty=0).
    """
    card = runner.card
    if card is None:
        raise RuntimeError("OnnxRunner.load() must be called before stream_infer()")
    fs_in = float(handle.fs_hz[channel])
    fs_out = float(card.fs_hz)
    L = int(card.l_m)
    ov = min(max(float(overlap), 0.0), 0.9)
    stride = max(1, int(round(L * (1.0 - ov))))
    n_in = int(handle.n_samples[channel])
    codes = [g.split("_")[0] for g in card.primary_head.class_order]
    art = card.artifact_head

    # one raw block ~= block_sec, but never smaller than a couple of windows' worth
    block_in = max(int(block_sec * fs_in), int(np.ceil(2 * L * fs_in / fs_out)) + 1)
    margin_in = max(1, int(round(0.5 * fs_in)))     # overlap-save margin (trimmed after resampling)

    tiers: list[str] = []
    confs: list[float] = []
    uncs: list[float] = []
    arts: list[list[str]] | None = [] if art is not None else None
    carry = np.empty(0, dtype=np.float32)

    read0 = 0
    while read0 < n_in:
        end = min(n_in, read0 + block_in)
        a = max(0, read0 - margin_in)
        b = min(n_in, end + margin_in)
        raw = np.asarray(read_window(handle, [channel], a, b), dtype=np.float32).reshape(-1)
        res = _resample(raw, fs_in, fs_out)
        # trim the resampled overlap-save margins → model-rate samples for exactly [read0, end)
        lt = int(round((read0 - a) * fs_out / fs_in))
        rt = int(round((b - end) * fs_out / fs_in))
        res = res[lt:len(res) - rt] if rt > 0 else res[lt:]
        read0 = end

        stream = np.concatenate([carry, res]) if carry.size else res
        nwin = (stream.shape[0] - L) // stride + 1 if stream.shape[0] >= L else 0
        if nwin <= 0:
            carry = stream
            continue
        wins = np.stack([stream[i * stride:i * stride + L] for i in range(nwin)])
        pred = runner.predict_windows_multihead(
            np.stack([normalize_window(w, card.normalization) for w in wins]))
        q = pred.primary
        tiers.extend(codes[i] for i in q.argmax(axis=1))
        confs.extend(q.max(axis=1).astype(float).tolist())
        # per-window normalized softmax entropy (predictive uncertainty) — nearly free here; matches the
        # in-memory InferenceTask so streamed intervals don't export a uniform uncertainty of 0.
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.clip(np.asarray(q, dtype=np.float64), 1e-12, 1.0)
            ent = -(p * np.log(p)).sum(axis=1) / np.log(p.shape[1])
        uncs.extend(np.nan_to_num(ent, nan=0.0).astype(float).tolist())
        if arts is not None:
            tp = pred.get(art.name)
            if tp is not None and len(tp):
                arts.extend(threshold_artifact_labels(tp, art.class_order, art.threshold))
            else:
                arts.extend([[] for _ in range(nwin)])
        carry = stream[nwin * stride:]

    stride_sec = (L * (1.0 - ov)) / fs_out
    window_sec = L / fs_out
    return (tiers, np.asarray(confs, dtype=np.float64), arts,
            np.asarray(uncs, dtype=np.float64), stride_sec, window_sec, len(tiers))


def build_plot_cache_blockwise(handle, channel: str, fs: float, cap_points: int = 400_000,
                               block_samples: int = 4_000_000):
    """Strided plot cache built block-by-block (no full-length read). Produces exactly the points a
    full ``raw[::stride]`` would, so the trace looks identical, but memory/read stays bounded.
    Returns ``(full_t, full_y, lo, hi)``."""
    n = int(handle.n_samples[channel])
    fs = float(fs) or 1.0
    if n <= 0:
        return np.zeros(0, np.float64), np.zeros(0, np.float64), -1.0, 1.0
    stride = max(1, int(np.ceil(n / cap_points)))
    ts: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    gmin, gmax = np.inf, -np.inf
    read0 = 0
    while read0 < n:
        end = min(n, read0 + block_samples)
        raw = np.asarray(read_window(handle, [channel], read0, end)).reshape(-1)
        if raw.size:
            gmin = min(gmin, float(raw.min()))
            gmax = max(gmax, float(raw.max()))
            first = (-read0) % stride                # first local index on the global stride grid
            idx = np.arange(first, raw.shape[0], stride)
            if idx.size:
                ts.append((read0 + idx).astype(np.float64) / fs)
                ys.append(raw[idx].astype(np.float64))
        read0 = end
    if not ts:
        return np.zeros(0, np.float64), np.zeros(0, np.float64), -1.0, 1.0
    full_t = np.ascontiguousarray(np.concatenate(ts))
    full_y = np.ascontiguousarray(np.concatenate(ys))
    if gmin > gmax:
        gmin, gmax = -1.0, 1.0
    return full_t, full_y, float(gmin), float(gmax if gmax > gmin else gmin + 1.0)
