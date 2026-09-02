"""Out-of-core (streaming) inference + block-wise plot cache — so multi-day / big multichannel
recordings never hold the whole signal in RAM.

Sliding-window inference is exactly chunkable: read the analysis channel in blocks, resample each
to model rate with an overlap-save margin (kills block-edge artifacts), and carry the unconsumed
tail between blocks so windows straddling a boundary are still produced. The per-window results are
identical to processing the whole signal, but memory is bounded to one block. The plot cache is
built the same way — block-wise strided decimation, no full-length read.

"Exactly" is load-bearing and it is not free: the blocks are cut on the RESAMPLER'S POLYPHASE GRID
(multiples of the ``resample_poly`` denominator), because a block that starts off that grid is filtered
on a different phase than the whole-signal resample would use there, and its output length rounds to
+-1 sample — errors that accumulate block after block. See ``stream_infer``. Without it a streamed
record whose native rate is not the model rate was scoring a drifted, phase-shifted signal.

Scope (honesty): the streaming path produces the primary quality segmentation (+ artifact tags) and,
whenever the memory budget below allows, the SAME boundary refinement the in-memory path applies —
a record must not get coarser segment boundaries merely because its file was big enough to take this
path. The recoverability second pass and the false-clean integrity guard run on normal-size records
only (they need a filtered/prefiltered view of the whole signal); a streamed record shows a notice
naming exactly which of the three did NOT run.
"""
from __future__ import annotations

import numpy as np

from biosqa.inference.postprocess import (
    calibrate_grade_probs,
    confidences_from,
    normalized_entropy,
)
from biosqa.inference.preprocess import normalize_window
from biosqa.inference.segmenter import threshold_artifact_labels
from biosqa.io.loaders import read_window
from biosqa.io.pyramid import minmax_envelope_indices, samples_per_bucket

#: analysis-channel sample count above which the streaming path kicks in (~9 h @ 250 Hz,
#: ~2 h @ 1 kHz). Below this the app keeps the simpler full-in-memory path unchanged.
LARGE_RECORD_SAMPLES = 8_000_000

#: Model-rate samples the streaming pass is willing to RETAIN so that BOUNDARY REFINEMENT can run
#: (``stream_infer(collect_signal=True)``).
#:
#: Refinement is not chunkable the way inference is: :func:`inference.refine.fine_badness` scores every
#: fine bin against WHOLE-SIGNAL robust statistics (the median, the 99th percentile of |x - median| and
#: max|x|), so it needs the entire scored signal at once — and it promotes it to float64 and holds a
#: couple of transient copies, i.e. ~32 B per model-rate sample at peak (measured: 24 M samples -> 0.8 s).
#: 24 M samples is therefore ~96 MB retained (float32) and ~0.75 GB at peak inside ``fine_badness``,
#: and covers a 26 h ECG @ 250 Hz, a 26 h EEG @ 256 Hz, a 4-day PPG @ 64 Hz or a month of EDA @ 8 Hz —
#: comfortably past ``LARGE_RECORD_SAMPLES``, so there is a WIDE band of streamed records that still get
#: refined boundaries (the discontinuity the streaming threshold used to introduce is gone).
#:
#: Beyond this the module keeps its memory promise and refinement is SKIPPED — and the caller must say
#: so out loud (``workers.qt_threads.StreamInferenceTask`` names it in its notice). Silently handing a
#: user coarser boundaries for a bigger file is exactly what this constant exists to prevent.
REFINE_MAX_MODEL_SAMPLES = 24_000_000


def _resample(sig, fs_in: float, fs_out: float):
    from biosqa.workers.qt_threads import resample_signal  # local import avoids a circular import
    return resample_signal(sig, fs_in, fs_out)


def _ratio(fs_in: float, fs_out: float):
    from biosqa.workers.qt_threads import resample_ratio   # local import avoids a circular import
    return resample_ratio(fs_in, fs_out)


def estimate_analysis_samples(handle, channel: str) -> int:
    """Native-rate sample count of the analysis channel (the streaming-vs-in-memory decision)."""
    try:
        return int(handle.n_samples[channel])
    except Exception:  # noqa: BLE001
        return 0


def stream_infer(handle, channel: str, runner, *, overlap: float = 0.0,
                 block_sec: float = 300.0, cancel=None, collect_signal: bool = False,
                 max_collect_samples: "int | None" = None):
    """Chunked sliding-window inference that never materializes the whole signal.

    Returns ``(tiers[list[str]], confidences[np.ndarray], artifacts_per_window[list[list[str]]|None],
    uncertainty[np.ndarray], grade_probs[n_windows, n_classes], starts_sec[np.ndarray], stride_sec,
    window_sec, n_windows, signal[np.ndarray|None])`` — the first nine are the exact inputs
    :func:`segmenter.run_length_encode` needs.

    ``signal`` is the model-rate signal the windows were actually cut from, returned ONLY when
    ``collect_signal`` is set AND it fits ``max_collect_samples`` (see
    :data:`REFINE_MAX_MODEL_SAMPLES`) — it is what :func:`inference.refine.refine_intervals` needs to
    run on this path, so a streamed record gets the same boundaries as the in-memory one. ``None``
    means refinement CANNOT run for this record and the caller must say so rather than quietly ship
    window-resolution boundaries. It is never a partial signal: a cancelled pass returns ``None``.

    ``starts_sec`` is the TRUE start time of each window, because the grid is not uniform: like
    :func:`preprocess.make_windows`, a record that is not a whole number of windows gets a final
    END-ANCHORED window (start ``n - L_m``) so its tail is graded rather than dropped. Bounding the
    intervals on ``i * stride_sec`` instead would attribute that window's grade to a span running past
    the end of the recording.

    Every per-window number comes from :mod:`inference.postprocess`, the same helper the in-memory
    :class:`workers.qt_threads.InferenceTask` calls: the raw softmax is sanitized (non-finite ->
    uniform) BEFORE confidence/entropy are read off it — the grade temperature is baked into the ONNX
    graph, so the softmax is already calibrated and is never re-scaled host-side — and the
    calibrated distribution is returned so a streamed record gets the same APS prediction sets. A
    record must not export different numbers just because it was big enough to take this path.

    ``cancel`` is an optional cooperative token (``threading.Event``) checked once per block; when it
    is set the loop stops and the windows produced SO FAR are returned — a PARTIAL result the caller
    must discard (it re-checks the token), never show as a complete segmentation. The end-anchored tail
    window is only ever appended to a COMPLETE pass (a partial one has no tail to anchor to).
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

    # Blocks are cut on the POLYPHASE GRID: `resample_poly`'s filter phase at a block start depends on
    # that start index MOD `down`, so a block starting off the grid is resampled on a DIFFERENT phase
    # than the whole-signal resample uses at the same time -- and its output length rounds to +-1 sample.
    # Both errors accumulate block after block, so a streamed record whose native rate differs from the
    # model rate drifted away from the in-memory one: measured on a 137.5 s 500 Hz ECG at block_sec=17,
    # the streamed model-rate signal came out 3 samples LONG with a 0.11 max error on a 1.3 p2p signal --
    # a different signal, hence different grades and a visibly different segmentation. With `read0`,
    # `end` and `margin_in` all multiples of `down`, the model-rate index of every block start is exactly
    # `read0 * up // down` and the trims below are exact integers, so the streamed signal is the
    # whole-signal resample. (up == down == 1 when the rates match: this is then a no-op.)
    up, down = _ratio(fs_in, fs_out)

    def _grid(x: int) -> int:                       # round UP to a whole number of `down` input samples
        return int(np.ceil(max(1, x) / down)) * down

    # one raw block ~= block_sec, but never smaller than a couple of windows' worth
    block_in = _grid(max(int(block_sec * fs_in), int(np.ceil(2 * L * fs_in / fs_out)) + 1))
    # Overlap-save margin, trimmed after resampling. It has to swallow the polyphase filter's TRANSIENT,
    # so it is sized off the filter, not off a wall-clock guess: `resample_poly` builds a half-length of
    # 10*max(up, down) taps at the UPSAMPLED rate, i.e. ceil(10*max(up,down)/up) INPUT samples; take 2x.
    # A flat "0.5 s" is ample at 250-1000 Hz but NOT at a low native rate -- EDA at 32 Hz gave a 16-sample
    # margin against a 40-sample transient, and the block edges leaked into the scored signal.
    transient_in = int(np.ceil(10.0 * max(up, down) / up))
    margin_in = _grid(max(int(round(0.5 * fs_in)), 2 * transient_in))

    tiers: list[str] = []
    confs: list[float] = []
    uncs: list[float] = []
    gprobs: list[np.ndarray] = []          # calibrated per-window grade distributions (n x 4 float64)
    arts: list[list[str]] | None = [] if art is not None else None
    starts: list[int] = []                 # model-rate first-sample index of each window (GLOBAL)
    carry = np.empty(0, dtype=np.float32)

    def _score(wins: np.ndarray) -> None:
        """Grade a stack of windows and append every per-window number, in window order."""
        pred = runner.predict_windows_multihead(
            np.stack([normalize_window(w, card.normalization) for w in wins]))
        # Sanitize ONCE (the graph already applied the temperature), then read every user-facing
        # number off that ONE calibrated distribution (identical to the in-memory path — see postprocess).
        q, non_finite = calibrate_grade_probs(pred.primary, card)
        tiers.extend(codes[i] for i in q.argmax(axis=1))
        confs.extend(confidences_from(q, non_finite).tolist())
        uncs.extend(normalized_entropy(q).tolist())
        gprobs.append(q)
        if arts is not None:
            tp = pred.get(art.name)
            if tp is not None and len(tp):
                arts.extend(threshold_artifact_labels(tp, art.class_order, art.threshold))
            else:
                arts.extend([[] for _ in range(len(wins))])

    # Keep the model-rate blocks for BOUNDARY REFINEMENT when the caller asked and the record fits the
    # budget. The estimate is checked up front (so an over-budget record never starts accumulating) and
    # again per block (so a mis-estimate still cannot blow the bound) -- dropping `kept` to None is the
    # single signal to the caller that refinement is off for this record.
    # (the budget is read HERE, not bound as a default argument, so it stays one tunable knob)
    budget = int(REFINE_MAX_MODEL_SAMPLES if max_collect_samples is None else max_collect_samples)
    kept: list[np.ndarray] | None = None
    if collect_signal:
        est_model = int(round(n_in * fs_out / fs_in)) if fs_in > 0 else 0
        if est_model <= budget:
            kept = []

    read0 = 0
    stream_off = 0                         # model-rate GLOBAL index of the unconsumed carry
    n_model = 0                            # model-rate samples produced so far (the record's length)
    hist = np.empty(0, dtype=np.float32)   # the last L model-rate samples ever seen (for the tail window)
    cancelled = False
    while read0 < n_in:
        if cancel is not None and cancel.is_set():
            cancelled = True
            break                          # superseded run: stop reading blocks (partial, see docstring)
        end = min(n_in, read0 + block_in)          # `read0` is on the grid; so is `end` unless it is n_in
        a = max(0, read0 - margin_in)              # ...and so are both margins
        b = min(n_in, end + margin_in)
        raw = np.asarray(read_window(handle, [channel], a, b), dtype=np.float32).reshape(-1)
        res = _resample(raw, fs_in, fs_out)
        # Trim the resampled overlap-save margins -> the model-rate samples for exactly [read0, end).
        # Both trims are EXACT integers because the cuts are on the polyphase grid (see above); the last
        # block keeps everything to the end of the record (`end == n_in` need not be on the grid).
        lt = (read0 - a) * up // down
        res = res[lt:] if end >= n_in else res[lt:lt + (end - read0) * up // down]
        read0 = end
        n_model += int(res.shape[0])
        hist = (np.concatenate([hist, res])[-L:] if hist.size else res[-L:])
        if kept is not None:
            if n_model > budget:
                kept = None                # over budget after all -> refinement off, memory bound held
            else:                          # copy: `res` is a view of the (margin-padded) block, and
                                           # holding it alive would pin the margins too
                kept.append(np.ascontiguousarray(res, dtype=np.float32))

        stream = np.concatenate([carry, res]) if carry.size else res
        nwin = (stream.shape[0] - L) // stride + 1 if stream.shape[0] >= L else 0
        if nwin <= 0:
            carry = stream
            continue
        _score(np.stack([stream[i * stride:i * stride + L] for i in range(nwin)]))
        starts.extend(stream_off + i * stride for i in range(nwin))
        carry = stream[nwin * stride:]
        stream_off += nwin * stride

    # END-ANCHORED tail window: the uniform grid stops at the last window that FITS, so up to one
    # stride of signal (60 s for EDA at overlap 0) would go UNGRADED at the end of the record — and an
    # ungraded tail reads to a user as "no problems found there". `hist` holds the record's last L
    # model-rate samples, which IS that window, so it is scored exactly as make_windows would in
    # memory. Skipped on a cancelled (partial) pass — there is no end to anchor to yet.
    if not cancelled and starts and n_model >= L and starts[-1] + L < n_model:
        _score(np.asarray(hist[-L:], dtype=np.float32).reshape(1, L))
        starts.append(n_model - L)

    stride_sec = (L * (1.0 - ov)) / fs_out
    window_sec = L / fs_out
    n_classes = len(codes)
    grade_probs = (np.concatenate(gprobs, axis=0) if gprobs
                   else np.zeros((0, n_classes), dtype=np.float64))
    starts_sec = np.asarray(starts, dtype=np.float64) / fs_out
    # The retained signal IS the one the windows were cut from (the blocks are exactly the trimmed,
    # margin-free model-rate stream), so refinement scores the same samples the model graded. A
    # cancelled pass read only part of the record -- never hand that out as "the signal".
    signal = (np.concatenate(kept) if kept else np.zeros(0, dtype=np.float32)) \
        if (kept is not None and not cancelled) else None
    return (tiers, np.asarray(confs, dtype=np.float64), arts,
            np.asarray(uncs, dtype=np.float64), grade_probs, starts_sec,
            stride_sec, window_sec, len(tiers), signal)


def build_plot_cache_blockwise(handle, channel: str, fs: float, cap_points: int = 400_000,
                               block_samples: int = 4_000_000):
    """MIN/MAX-envelope plot cache built block-by-block (no full-length read), with memory/read bounded.

    Block ends are snapped down to bucket boundaries, so the bucket GRID is global: every bucket lies
    inside one block and each carries the same two extrema (min and max, in time order) the
    whole-signal :func:`workers.qt_threads.build_plot_cache` would emit — no extremum is ever dropped
    or displaced. The ONE difference, and it is not a dropped sample: when the final block holds a
    single sample, :func:`io.pyramid.minmax_envelope_indices` short-circuits and emits that index ONCE,
    where the whole-signal build emits it TWICE (its min and its max coincide). Same values, same
    trace, one duplicate point fewer. Returns ``(full_t, full_y, lo, hi)``."""
    n = int(handle.n_samples[channel])
    fs = float(fs) or 1.0
    if n <= 0:
        return np.zeros(0, np.float64), np.zeros(0, np.float64), -1.0, 1.0
    spb = samples_per_bucket(n, int(cap_points))
    ts: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    gmin, gmax = np.inf, -np.inf
    read0 = 0
    while read0 < n:
        end = min(n, read0 + block_samples)
        if end < n and spb > 1:
            # Snap the block END down to a bucket boundary (blocks start at 0 and each is then a whole
            # number of buckets) so the bucket grid is GLOBAL: no bucket straddles a block, and the
            # cache is identical to the whole-signal build.
            end = min(n, read0 + max(spb, ((end - read0) // spb) * spb))
        raw = np.asarray(read_window(handle, [channel], read0, end)).reshape(-1)
        if raw.size:
            gmin = min(gmin, float(raw.min()))
            gmax = max(gmax, float(raw.max()))
            idx = minmax_envelope_indices(raw, spb)
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
