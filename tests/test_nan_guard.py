"""InferenceTask fail-safes.

(1) Non-finite input must never produce a NaN confidence in the exported intervals: a NaN/inf window
passes through card-constant normalization to a NaN softmax, so the inference task must fail-safe it to a
low-confidence Q0 (worst tier) rather than RLE-encoding and exporting a NaN.
(2) A CANCELLED (superseded) task must stop at the next phase boundary and emit NOTHING AT ALL — an
emitted ``failed`` would clobber the status text of the run that replaced it."""
import math
import os
import threading
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("onnxruntime")
from PySide6.QtWidgets import QApplication  # noqa: E402

from biosqa.inference.onnx_runner import OnnxRunner  # noqa: E402
from biosqa.workers.qt_threads import InferenceTask  # noqa: E402
from biosqa.workers.signals import InferenceWorkerSignals  # noqa: E402

_app = QApplication.instance() or QApplication([])
MODELS = Path(__file__).resolve().parents[1] / "models"


def test_non_finite_window_fails_safe_not_nan():
    r = OnnxRunner("ecg", str(MODELS))
    r.load()
    fs, lm = r.card.fs_hz, r.card.l_m
    # 4 windows of ECG; blow the 3rd one up with NaN/inf so its softmax goes non-finite
    sig = 0.02 * np.random.default_rng(0).standard_normal(lm * 4)
    for i in range(int(0.4 * fs), sig.size - 3, int(fs * 60 / 72)):
        sig[i:i + 3] += 1.2
    sig[lm * 2:lm * 3:2] = np.nan
    sig[lm * 2 + 1:lm * 3:2] = np.inf

    got = {}
    carrier = InferenceWorkerSignals()
    carrier.intervalsReady.connect(lambda mod, ivs: got.setdefault("ivs", ivs))
    InferenceTask(r, sig.astype(np.float32), window_stride_sec=lm / fs, window_length_sec=lm / fs,
                  signals=carrier, guard_enabled=False, recovery_enabled=False, refine_enabled=False).run()

    ivs = got.get("ivs")
    assert ivs, "inference emitted no intervals"
    for iv in ivs:                                     # nothing exported may be NaN/inf
        assert math.isfinite(iv.confidence), f"NaN/inf confidence exported: {iv}"
        assert math.isfinite(getattr(iv, "uncertainty", 0.0))
    # the corrupted span (samples [2*lm, 3*lm]) must read as worst-tier, zero-confidence
    bad = [iv for iv in ivs if iv.start_sec < (3 * lm / fs) and iv.end_sec > (2 * lm / fs)]
    assert any(iv.tier == "Q0" and iv.confidence == 0.0 for iv in bad), [(iv.tier, iv.confidence) for iv in bad]


def _quiet_ecg(rng_seed: int = 1):
    r = OnnxRunner("ecg", str(MODELS))
    r.load()
    lm = r.card.l_m
    sig = 0.02 * np.random.default_rng(rng_seed).standard_normal(lm * 4)
    return r, sig.astype(np.float32)


def _emissions(carrier: InferenceWorkerSignals) -> list:
    """Record EVERY signal an InferenceTask can emit."""
    seen: list = []
    carrier.intervalsReady.connect(lambda *a: seen.append("intervals"))
    carrier.guardReady.connect(lambda *a: seen.append("guard"))
    carrier.dataQualityReady.connect(lambda *a: seen.append("dataQuality"))
    carrier.failed.connect(lambda *a: seen.append("failed"))
    return seen


def test_cancelled_inference_task_emits_nothing():
    r, sig = _quiet_ecg()
    fs, lm = r.card.fs_hz, r.card.l_m
    cancel = threading.Event()
    cancel.set()                                   # superseded before the pool even got to it
    carrier = InferenceWorkerSignals()
    seen = _emissions(carrier)
    InferenceTask(r, sig, window_stride_sec=lm / fs, window_length_sec=lm / fs,
                  signals=carrier, cancel=cancel).run()
    assert seen == []                              # not even `failed` — it would clobber the live run


def test_cancel_after_the_primary_pass_skips_the_second_onnx_pass():
    """The recoverability pass is a SECOND full ONNX pass — the biggest chunk of a superseded run's
    wasted CPU. Cancelling mid-flight must stop before it, and still emit nothing."""
    r, sig = _quiet_ecg(2)
    fs, lm = r.card.fs_hz, r.card.l_m
    cancel = threading.Event()
    passes = []
    real = r.run_sliding_window_multihead

    def counting(signal, overlap=0.0):
        passes.append(overlap)
        cancel.set()               # a newer run supersedes us the instant the primary pass lands
        return real(signal, overlap=overlap)

    r.run_sliding_window_multihead = counting      # instance-level shadow
    carrier = InferenceWorkerSignals()
    seen = _emissions(carrier)
    InferenceTask(r, sig, window_stride_sec=lm / fs, window_length_sec=lm / fs,
                  signals=carrier, cancel=cancel).run()
    assert passes == [0.0]                         # the recoverability pass never ran
    assert seen == []
