"""The ``(up, down)`` polyphase ratio every resample in the app is cut from.

``resample_ratio`` used ``Fraction(fs_out / fs_in).limit_denominator(1000)``, and a FIXED 3-digit
denominator runs out of resolution as the ratio shrinks -- well inside the rates this app accepts
(``io.loaders`` treats anything up to 20 kHz as a plausible acquisition). Against the shipped 8 Hz
EDA model, reachable through the documented force-modality feature, it failed in two ways:

* **16001 Hz and up -> ``(0, 1)``** (20/22.05/24/30/32/44.1/48/96 kHz all). ``resample_poly(sig, 0,
  1)`` raises ``ValueError``, which the bare ``except Exception`` in :func:`resample_signal` swallowed
  into un-antialiased linear interpolation: decimating 20000 -> 8 Hz a 0.05 Hz tone contaminated by a
  3001 Hz one folds the interferer onto 1 Hz, *inside* the model's band -- std 0.9658 against a true
  0.7071 (+36.6%). Streaming is worse still: ``inference.streaming`` sizes its overlap-save margin as
  ``10 * max(up, down) / up`` and trims each block by ``(read0 - a) * up // down``, so ``up = 0``
  either divides by zero or hands every non-final block a zero-length slice.
* **16000 Hz -> ``(1, 1000)``**, which every resampler accepts and which is wrong by a FACTOR OF TWO:
  a 300 s record resampled to 4800 samples, i.e. 16 Hz fed to an 8 Hz model. 10000 -> 8 Hz was 25%
  high, 11025 -> 8 Hz 37.8%, 96000 -> 64 Hz 50%. A silent 2x rate error is the worse of the two, so
  the fix triggers on ACCURACY, not merely on ``up == 0``.

These tests pin all of it: the ratios that were already accurate are byte-for-byte unchanged, the
ones that were not are now exact, and a >16 kHz record streams to the same windows the in-memory path
scores.
"""
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from biosqa.workers.qt_threads import resample_ratio, resample_signal

MODELS = Path(__file__).resolve().parent.parent / "models"

#: Rate pairs the app already resolved accurately, with the ratio they resolved to. The escalation
#: must NOT touch these -- a changed ``down`` silently re-cuts the streaming block grid
#: (``streaming.stream_infer`` snaps every block to a multiple of ``down``).
_UNCHANGED = {
    (250.0, 250.0): (1, 1), (500.0, 250.0): (1, 2), (1000.0, 250.0): (1, 4),
    (360.0, 250.0): (25, 36), (2048.0, 250.0): (99, 811), (16001.0, 250.0): (1, 64),
    (128.0, 256.0): (2, 1), (512.0, 256.0): (1, 2), (44100.0, 256.0): (4, 689),
    (32.0, 8.0): (1, 4), (4.0, 8.0): (2, 1), (125.0, 8.0): (8, 125), (1000.0, 8.0): (1, 125),
    (700.0, 64.0): (16, 175), (64.0, 64.0): (1, 1), (11025.0, 64.0): (4, 689),
}

#: Every pair whose ratio this fix CHANGES, and the exact value it changes to. Each old value was
#: either ``(0, 1)`` (no resampler accepts it) or wrong by 0.01%-100%; each new one is exact.
_CORRECTED = {
    (10000.0, 8.0): (1, 1250),      # was (1, 1000) -- 25% high
    (11025.0, 8.0): (8, 11025),     # was (1, 1000) -- 37.8% high
    (16000.0, 8.0): (1, 2000),      # was (1, 1000) -- 2x: 16 Hz into an 8 Hz model
    (16001.0, 8.0): (8, 16001),     # was (0, 1)
    (20000.0, 8.0): (1, 2500),      # was (0, 1)
    (22050.0, 8.0): (4, 11025),     # was (0, 1)
    (24000.0, 8.0): (1, 3000),      # was (0, 1)
    (30000.0, 8.0): (1, 3750),      # was (0, 1)
    (32000.0, 8.0): (1, 4000),      # was (0, 1)
    (44100.0, 8.0): (2, 11025),     # was (0, 1)
    (48000.0, 8.0): (1, 6000),      # was (0, 1)
    (96000.0, 8.0): (1, 12000),     # was (0, 1)
    (30000.0, 64.0): (4, 1875),     # was (1, 469)  -- 5.3e-4 high
    (96000.0, 64.0): (1, 1500),     # was (1, 1000) -- 50% high
    (30000.0, 256.0): (16, 1875),   # was (5, 586)  -- 1.1e-4 high
}


def test_resample_ratio_is_unchanged_for_every_pair_that_already_resolved_accurately():
    for (fs_in, fs_out), expected in _UNCHANGED.items():
        assert resample_ratio(fs_in, fs_out) == expected, f"{fs_in} -> {fs_out}"


def test_resample_ratio_corrects_the_pairs_a_1000_denominator_could_not_express():
    for (fs_in, fs_out), expected in _CORRECTED.items():
        up, down = resample_ratio(fs_in, fs_out)
        assert (up, down) == expected, f"{fs_in} -> {fs_out}"
        assert up / down == pytest.approx(fs_out / fs_in, rel=1e-12)   # exact, not merely non-zero


@pytest.mark.parametrize("fs_in", [16000.0, 16001.0, 20000.0, 22050.0, 24000.0,
                                   30000.0, 32000.0, 44100.0, 48000.0, 96000.0])
def test_resample_ratio_never_returns_a_zero_numerator(fs_in):
    """``up >= 1`` is the hard part of the contract: ``resample_poly`` rejects 0 and ``stream_infer``
    divides by it. Everything from 16001 Hz up returned ``(0, 1)``."""
    up, down = resample_ratio(fs_in, 8.0)
    assert up >= 1 and down >= 1
    assert up / down == pytest.approx(8.0 / fs_in, rel=1e-9)


def test_resample_ratio_across_the_16khz_boundary():
    """The exact step where the old cap gave out. One hertz of native rate separated a 2x-wrong ratio
    from an unusable one, with nothing correct on either side."""
    assert resample_ratio(16000.0, 8.0) == (1, 2000)     # was (1, 1000): 16 Hz, not 8
    assert resample_ratio(16001.0, 8.0)[0] >= 1          # was (0, 1)
    assert resample_ratio(20000.0, 8.0)[0] >= 1          # was (0, 1)


def test_resample_ratio_raises_rather_than_faking_a_degenerate_ratio():
    with pytest.raises(ValueError, match="degenerate resample ratio"):
        resample_ratio(1e9, 8.0)


def test_resample_signal_at_16khz_lands_on_the_model_rate():
    """The 2x bug in the one unit that matters: output LENGTH. 300 s at the model's 8 Hz is 2400
    samples; ``(1, 1000)`` produced 4800, so every window covered half the time it claimed."""
    fs_in, secs = 16000.0, 300
    x = np.sin(2 * np.pi * 0.05 * np.arange(int(fs_in * secs)) / fs_in).astype(np.float32)
    assert resample_signal(x, fs_in, 8.0).size == int(secs * 8.0)


def test_resample_signal_above_16khz_is_antialiased_not_linearly_interpolated():
    """The ``up == 0`` consequence in memory, on the numbers: 20000 -> 8 Hz on a 0.05 Hz tone plus a
    3001 Hz one, which folds onto 1 Hz -- squarely inside what the EDA model reads. Anti-aliased,
    max|error| against the pure tone is 5.9e-5; through the linear fallback the interferer survives at
    full amplitude (std 0.9658 against the true 0.7071)."""
    fs_in, fs_out, secs = 20000.0, 8.0, 120
    t = np.arange(int(fs_in * secs)) / fs_in
    x = (np.sin(2 * np.pi * 0.05 * t) + np.sin(2 * np.pi * 3001.0 * t)).astype(np.float32)

    out = resample_signal(x, fs_in, fs_out)
    assert out.size == int(secs * fs_out)
    truth = np.sin(2 * np.pi * 0.05 * np.arange(out.size) / fs_out)
    assert float(np.abs(out[16:-16] - truth[16:-16]).max()) < 1e-3
    assert float(out.std()) == pytest.approx(0.7071, abs=5e-3)


def test_resample_signal_logs_the_linear_fallback(caplog):
    """The fallback is legitimate for a ratio no polyphase filter can express -- staying SILENT about
    it is what let the ``(0, 1)`` bug run for every recording above 16 kHz. It must name the rates,
    the ratio it tried and the reason."""
    x = np.sin(2 * np.pi * 0.05 * np.arange(4096) / 1e9).astype(np.float32)
    with caplog.at_level("WARNING", logger="biosqa.workers.qt_threads"):
        out = resample_signal(x, 1e9, 8.0)
    assert out.size >= 2
    msgs = [r.getMessage() for r in caplog.records]
    assert any("un-antialiased linear interpolation" in m and "degenerate resample ratio" in m
               for m in msgs), msgs


def _eda_runner():
    pytest.importorskip("onnxruntime")
    if not (MODELS / "eda.onnx").exists():
        pytest.skip("eda.onnx model not present")
    from biosqa.inference.onnx_runner import OnnxRunner

    r = OnnxRunner("eda", MODELS)
    r.load()
    return r


def test_stream_infer_over_a_20khz_record_matches_the_in_memory_path(tmp_path):
    """The streaming consequence, end to end: a 20 kHz recording forced onto the 8 Hz EDA model.
    With ``(up, down) == (0, 1)`` ``stream_infer`` never reaches a window at all -- it divides by
    ``up`` sizing the overlap-save margin, and its per-block trim ``(read0 - a) * up // down`` is a
    zero-length slice, so a record that survived that would be graded on its last block alone."""
    wfdb = pytest.importorskip("wfdb")
    from biosqa.inference.streaming import stream_infer
    from biosqa.io.loaders import open_recording, read_window

    r = _eda_runner()
    fs_in, secs = 20000.0, 150
    t = np.arange(int(fs_in * secs)) / fs_in
    rng = np.random.default_rng(0)
    sig = 5.0 + 2.0 * np.sin(2 * np.pi * 0.05 * t)                     # tonic EDA-ish drift
    sig[int(fs_in * 60):int(fs_in * 90)] += 3.0 * rng.standard_normal(int(fs_in * 30))
    wfdb.wrsamp("hi", fs=fs_in, units=["uS"], sig_name=["EDA"],
                p_signal=sig.reshape(-1, 1).astype(np.float64), write_dir=str(tmp_path))
    h = open_recording(str(tmp_path / "hi.hea"))

    codes = [g.split("_")[0] for g in r.card.primary_head.class_order]
    raw = np.asarray(read_window(h, ["EDA"], 0, h.n_samples["EDA"]), dtype=np.float32).reshape(-1)
    mem = resample_signal(raw, fs_in, 8.0)
    q_mem = r.run_sliding_window_multihead(mem).primary
    mem_tiers = [codes[i] for i in q_mem.argmax(axis=1)]

    # block_sec=40 on a 150 s record -> 4 streamed blocks, so the polyphase grid is exercised too
    tiers, _confs, _arts, _uncs, gprobs, starts, _ss, _ws, nwin, _sig = stream_infer(
        h, "EDA", r, overlap=0.0, block_sec=40.0)

    assert nwin == len(mem_tiers) > 0        # the count the bug collapsed to (at best) one block's
    assert tiers == mem_tiers
    assert np.allclose(gprobs, q_mem, atol=1e-9)
    assert np.allclose(starts, r.window_starts_sec(mem))
