"""The WINDOWING CONTRACT, end to end: model windows -> per-window grades -> run-length intervals ->
boundary refinement.

Every test here drives the PRODUCTION path (``workers.qt_threads.InferenceTask``): the real
``preprocess.make_windows`` / ``window_starts`` grid (including the END-ANCHORED tail window), the real
``segmenter.run_length_encode``, the real ``refine.refine_intervals``, joined by the real worker wiring.
Only the ONNX forward is stubbed -- by a grader that behaves exactly like a receptive-field-limited
model (a window that CONTAINS a burst sample is poor, whole), which is the smear this chain exists to
bound. The window geometry is the SHIPPED geometry, read off ``models/<modality>.model_card.json``.

Intervals are never hand-built here. A refinement test fed hand-built OVERLAPPING intervals is testing
an input the pipeline cannot produce (``run_length_encode`` output is strictly non-overlapping), and it
passes whether or not refinement does anything at all.

THE ORACLE IS EXHAUSTIVE, NOT SAMPLED (see :func:`_elementary_probes`). The oracle this replaces stepped
1.0 s from ``interval.start_sec + 0.5`` -- i.e. it probed exactly the BIN CENTRES, the one grid on which
refinement's centre-sampling bug was self-consistent. It reproduced the code's own error and agreed with
it, and so stayed green while the shipped 50%-overlap default displayed *excellent* over half-bins the
model had only ever graded *poor*. A test whose sampling grid is aligned to the code's internal grid is
not a test.
"""
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication  # noqa: E402

from biosqa.inference.onnx_runner import MultiHeadPrediction, OnnxRunner  # noqa: E402
from biosqa.inference.refine import _RANK, fine_badness, refine_intervals  # noqa: E402
from biosqa.inference.segmenter import QualityInterval as QI  # noqa: E402
from biosqa.inference.segmenter import window_intervals  # noqa: E402
from biosqa.model.model_card import Head, ModelCard, Normalization  # noqa: E402
from biosqa.workers.qt_threads import InferenceTask  # noqa: E402
from biosqa.workers.signals import InferenceWorkerSignals  # noqa: E402

_app = QApplication.instance() or QApplication([])

FS = 250.0
L_M = 2500                    # the shipped ECG receptive field: 10 s @ 250 Hz
WIN_SEC = L_M / FS            # 10.0
GOOD_CONF, POOR_CONF = 0.91, 0.42
ORDER = ("Q0", "Q1", "Q2", "Q3")

#: the SHIPPED per-modality geometry, ``(L_m samples, fs Hz)`` -- from models/<modality>.model_card.json.
#: EEG's 5 s window is why the app's DEFAULT 50% overlap strides 2.5 s: every odd window start then
#: lands exactly on a 1 s bin's CENTRE, which is what made the centre-sampling bug systematic there.
GEOMETRY = {
    "ecg": (2500, 250.0),     # 10 s
    "eeg": (1280, 256.0),     #  5 s
    "ppg": (640, 64.0),       # 10 s
    "eda": (480, 8.0),        # 60 s
}

#: every overlap the app actually offers (biosqa/ui/SettingsOverlay.qml).
OVERLAPS = [0.0, 0.25, 0.5, 0.75, 0.9]

#: ``(record_sec, burst, poor_tier)`` per modality. The SECOND record of each pair is deliberately NOT a
#: whole number of windows, so ``make_windows`` END-ANCHORS the tail window and its start carries an
#: arbitrary fractional part -- one of the three ways a point-sampled bin verdict went wrong.
RECORDS = {
    "ecg": [(60.0, (14.0, 15.0), "Q0"), (61.2, (18.0, 19.0), "Q1")],
    "eeg": [(60.0, (4.5, 5.5), "Q1"), (33.7, (12.0, 12.6), "Q0")],
    "ppg": [(60.0, (14.0, 15.0), "Q0"), (61.2, (18.0, 19.0), "Q1")],
    "eda": [(300.0, (90.0, 95.0), "Q1"), (317.5, (150.0, 155.0), "Q0")],
}

_GRADE = Head(name="grade", output_name="grade", kind="ordinal", activation="softmax",
              class_order=ORDER)


def _card(modality: str = "ecg") -> ModelCard:
    l_m, fs = GEOMETRY[modality]
    return ModelCard(
        modality=modality, l_m=l_m, fs_hz=fs, class_order=ORDER,
        normalization=Normalization(method="none"), training_data_hash="sha256:test",
        model_version="test", source_path=Path(f"{modality}.model_card.json"), heads=(_GRADE,),
    )


CARD = _card("ecg")


class _WindowGrader(OnnxRunner):
    """A runner whose ONLY stub is the ONNX FORWARD.

    Windowing (``make_windows`` / ``window_starts`` -- so the end-anchored tail window is the real
    one), the card normalization, ``run_length_encode``, ``refine_intervals`` and the ``InferenceTask``
    wiring that joins them are all production code, run exactly as the app runs them.

    The grade is the one a fixed-receptive-field model gives: ``poor_tier`` (conf 0.42) iff the window
    CONTAINS a burst sample, else Q3 (conf 0.91). The burst poisons the WHOLE window -- that is the
    smear.
    """

    def __init__(self, modality: str = "ecg", poor_tier: str = "Q0") -> None:
        super().__init__(modality, models_dir=".")
        self.card = _card(modality)
        self.poor_tier = poor_tier

    def predict_windows_multihead(self, windows) -> MultiHeadPrediction:
        w = np.asarray(windows, dtype=np.float64).reshape(len(windows), -1)
        poor = w.max(axis=1) > 1.0                       # burst amplitude 3.0 vs 0.1 baseline
        probs = np.tile(np.array([0.02, 0.02, 0.05, GOOD_CONF], dtype=np.float32), (len(w), 1))
        row = np.full(4, (1.0 - POOR_CONF) / 3.0, dtype=np.float32)
        row[ORDER.index(self.poor_tier)] = POOR_CONF     # argmax poor_tier at conf 0.42
        probs[poor] = row
        return MultiHeadPrediction(per_head={"grade": probs}, primary_name="grade")


def _burst_signal(secs=60.0, burst=(14.0, 15.0), modality="ecg"):
    fs = GEOMETRY[modality][1]
    n = int(round(secs * fs))
    t = np.arange(n) / fs
    x = 0.1 * np.sin(2 * np.pi * 1.1 * t)                                  # calm quasi-periodic base
    m = (t >= burst[0]) & (t < burst[1])
    x[m] += 3.0 * np.abs(np.random.default_rng(0).standard_normal(int(m.sum())))   # big-amplitude burst
    return x.astype(np.float32)


def _pipeline(sig, overlap: float, refine: bool = True, modality: str = "ecg", poor_tier: str = "Q0"):
    """Run the REAL inference worker over ``sig`` and return ``(intervals, model_windows)``.

    ``intervals`` are what the app displays/exports. ``model_windows`` are the model's raw per-window
    statements (overlapping at ``overlap > 0``), read back from the same runner through the same
    production windowing -- the oracle every honesty assertion below is checked against.
    """
    runner = _WindowGrader(modality, poor_tier)
    win_sec = runner.card.l_m / float(runner.card.fs_hz)
    got: dict = {}
    carrier = InferenceWorkerSignals()
    carrier.intervalsReady.connect(lambda mod, ivs: got.setdefault("ivs", list(ivs)))
    carrier.failed.connect(lambda mod, err: got.setdefault("err", err))
    InferenceTask(runner, sig, window_stride_sec=win_sec * (1.0 - overlap), window_length_sec=win_sec,
                  signals=carrier, overlap=overlap, guard_enabled=False, recovery_enabled=False,
                  refine_enabled=refine).run()
    assert "err" not in got, got["err"]
    assert got.get("ivs"), "inference emitted no intervals"

    starts = runner.window_starts_sec(sig, overlap=overlap)
    probs = runner.run_sliding_window_multihead(sig, overlap=overlap).primary
    tiers = [ORDER[i] for i in probs.argmax(axis=1)]
    model_windows = window_intervals(tiers, probs.max(axis=1), starts, win_sec)
    return got["ivs"], model_windows


def _ceiling(model_windows, t: float):
    """Best grade the MODEL gave any WINDOW covering ``t`` -- refinement may never display better."""
    ranks = [_RANK[w.tier] for w in model_windows if w.start_sec <= t < w.end_sec]
    return max(ranks) if ranks else None


def _elementary_probes(model_windows, ivs) -> list[float]:
    """Every instant that can possibly matter -- EXHAUSTIVELY, not on a sampling grid.

    Both quantities the invariant compares are piecewise-constant in ``t``: the DISPLAYED tier changes
    only at an interval edge, and the SET of model windows covering ``t`` changes only at a window edge.
    Cutting the record at every such edge therefore yields elementary spans on which both are constant,
    so ONE probe inside each span decides that entire span. Checking these probes proves the invariant
    for EVERY real instant -- there is no step size left for a violation to hide between.

    The 0.05 s sweep appended after them is pure redundancy (bins are 1 s, or 5 s for EDA, so it is
    strictly sub-bin): it would catch the violation even if the edge set above were ever computed wrong.
    It deliberately does NOT start at ``+0.5`` and does not step 1.0 s, which is precisely how the old
    oracle contrived to land only on bin CENTRES -- the one grid the bug was invisible on.
    """
    edges = sorted({iv.start_sec for iv in ivs} | {iv.end_sec for iv in ivs}
                   | {w.start_sec for w in model_windows} | {w.end_sec for w in model_windows})
    probes = [0.5 * (a + b) for a, b in zip(edges, edges[1:]) if b - a > 1e-9]
    end = max(iv.end_sec for iv in ivs)
    probes += np.arange(0.01, end, 0.05).tolist()
    return probes


def _assert_honest(model_windows, ivs, confs=(GOOD_CONF, POOR_CONF)):
    """The contract: never better than the model, never over signal the model never scored, never a
    confidence the model did not produce, every promotion auditable."""
    for t in _elementary_probes(model_windows, ivs):
        out = next((iv for iv in ivs if iv.start_sec <= t < iv.end_sec), None)
        if out is None:
            continue                       # nothing is displayed there, so nothing is claimed
        ceil_ = _ceiling(model_windows, t)
        assert ceil_ is not None, f"{out.tier} at t={t}s covers signal no window graded"
        assert _RANK[out.tier] <= ceil_, (
            f"{out.tier} at t={t}s is BETTER than anything the model said there "
            f"(best covering window = rank {ceil_})")
    for out in ivs:
        assert any(np.isclose(out.confidence, c) for c in confs), (
            f"confidence {out.confidence} is not a confidence the model produced")
        if out.model_tier:                                       # relaxed off the conservative grade
            assert _RANK[out.model_tier] < _RANK[out.tier], "model_tier must record a WORSE grade"


def _promoted(ivs, raw):
    """Times where refinement DISPLAYS BETTER than the RLE grade. Every one must be auditable, i.e.
    carry ``model_tier`` -- otherwise the export cannot show a reviewer what the model really said."""
    hits = []
    for t in _elementary_probes([], ivs):
        out = next((iv for iv in ivs if iv.start_sec <= t < iv.end_sec), None)
        rle = next((iv for iv in raw if iv.start_sec <= t < iv.end_sec), None)
        if out is not None and rle is not None and _RANK[out.tier] > _RANK[rle.tier]:
            hits.append((t, out))
    return hits


def _covers_contiguously(ivs, end_sec: float) -> bool:
    ivs = sorted(ivs, key=lambda iv: iv.start_sec)
    if not np.isclose(ivs[0].start_sec, 0.0):
        return False
    for a, b in zip(ivs, ivs[1:]):
        if not np.isclose(a.end_sec, b.start_sec):
            return False
    return bool(np.isclose(ivs[-1].end_sec, end_sec))


# --- N3: a bin is a SPAN. The verdict painted over it must be sampled over it, not at its centre ---

def test_refine_never_promotes_a_bin_half_the_model_only_ever_graded_poor_eeg_shipped_default():
    """THE REGRESSION. EEG at the app's SHIPPED DEFAULT 50% overlap: the 5 s window strides 2.5 s, so
    every odd window start lands EXACTLY on a 1 s bin's centre.

        MODEL WINDOWS:  [0,5] Q1   [2.5,7.5] Q1   [5,10] Q1   [7.5,12.5] Q3

    Bin [7, 8) is covered end-to-end by ONE window only, [5,10], which the model graded Q1. The Q3
    window [7.5, 12.5] starts at 7.5 -- it covers the bin's RIGHT half and says nothing whatever about
    its LEFT half. Sampling the bin at its centre (7.5) sees that Q3 window and paints Q3 across the
    WHOLE bin, so [7.0, 7.5) is displayed EXCELLENT although both windows the model ever ran over that
    time said POOR. In an SQA tool that is the worst possible direction of error.

    Before the span fix this emitted `[0, 7) Q1` then `[7, 9) Q3`; t=7.1 and t=7.4 were shown Q3 with a
    ceiling of Q1. The boundary must sit at 8.0 -- the first bin a Q3 window covers ENTIRELY.
    """
    sig = _burst_signal(secs=60.0, burst=(4.5, 5.5), modality="eeg")
    raw, mw = _pipeline(sig, overlap=0.5, refine=False, modality="eeg", poor_tier="Q1")
    ref, _ = _pipeline(sig, overlap=0.5, refine=True, modality="eeg", poor_tier="Q1")

    # the premise, pinned to the REAL windowing rather than to hand-typed numbers
    assert [(w.start_sec, w.end_sec, w.tier) for w in mw[:4]] == [
        (0.0, 5.0, "Q1"), (2.5, 7.5, "Q1"), (5.0, 10.0, "Q1"), (7.5, 12.5, "Q3")]
    assert ref != raw, "refinement is a NO-OP here: the fix must not be a retreat to doing nothing"

    def shown(t):
        return next(iv.tier for iv in ref if iv.start_sec <= t < iv.end_sec)

    for t in (7.05, 7.10, 7.25, 7.40, 7.49):        # the left half of bin [7, 8)
        assert _ceiling(mw, t) == _RANK["Q1"]       # ... the model only ever said Q1 there
        assert shown(t) == "Q1", f"t={t}s displayed {shown(t)} -- the model NEVER said better than Q1"
    assert shown(8.5) == "Q3"                       # bin [8, 9) IS covered whole by [7.5, 12.5] Q3
    poor_end = max(iv.end_sec for iv in ref if iv.tier in ("Q0", "Q1"))
    assert poor_end == pytest.approx(8.0), "the poor run must end at the first WHOLY-covered good bin"
    _assert_honest(mw, ref)


@pytest.mark.parametrize("modality", sorted(GEOMETRY))
@pytest.mark.parametrize("overlap", OVERLAPS)
def test_refined_output_is_never_better_than_the_model_over_every_shipped_configuration(modality, overlap):
    """The honesty invariant over the REAL pipeline, brute-forced across every modality the app ships
    and every overlap it offers (SettingsOverlay.qml: 0 / 25 / 50 / 75 / 90%), on both a whole-number-of-
    windows record AND one that is not (so the END-ANCHORED tail window, whose start has an arbitrary
    fractional part, is exercised at every overlap).

    For every elementary span of the record the displayed tier is never better than the BEST grade the
    model gave a WINDOW covering that span; no interval carries a confidence the model did not produce
    (the old code stamped a hardcoded 0.6 on every promoted window); and every displayed grade that is
    BETTER than the RLE grade is auditable through ``model_tier``.
    """
    for secs, burst, poor_tier in RECORDS[modality]:
        sig = _burst_signal(secs=secs, burst=burst, modality=modality)
        raw, mw = _pipeline(sig, overlap, refine=False, modality=modality, poor_tier=poor_tier)
        ref, _ = _pipeline(sig, overlap, refine=True, modality=modality, poor_tier=poor_tier)

        _assert_honest(mw, ref)
        assert not any(np.isclose(iv.confidence, 0.6) for iv in ref)   # the old fabricated constant
        assert any(iv.tier in ("Q0", "Q1") for iv in ref), "the artifact must never be refined away"
        for t, out in _promoted(ref, raw):
            assert out.model_tier, (
                f"{modality} ov={overlap}: t={t}s is displayed {out.tier} over the RLE grade with "
                f"model_tier EMPTY -- an unauditable promotion")


@pytest.mark.parametrize("modality", sorted(GEOMETRY))
def test_refine_still_fires_at_every_offered_overlap_and_is_a_no_op_only_at_zero(modality):
    """Guards the fix against the lazy 'fix': making refinement do nothing would satisfy the honesty
    invariant trivially and re-introduce the 10 s smear this module exists to bound. Refinement must
    still tighten a poor run at EVERY overlap the app offers.

    At overlap 0 it must be a no-op -- but ONLY because RECORDS[m][0] is deliberately an EXACT WHOLE
    NUMBER OF WINDOWS. Do not restate that as "overlap 0 is always a no-op": make_windows END-ANCHORS
    the tail window, so a record that is not an exact multiple has two genuinely overlapping windows at
    the end even at stride == L_m, and refinement legitimately promotes there. That case is pinned by
    test_at_overlap_zero_the_end_anchored_tail_window_can_still_promote_honestly below."""
    secs, burst, poor_tier = RECORDS[modality][0]
    l_m, fs = GEOMETRY[modality]
    assert (secs * fs) % l_m == 0, "this test's no-op branch relies on an exact-multiple record"
    sig = _burst_signal(secs=secs, burst=burst, modality=modality)
    for overlap in OVERLAPS:
        raw, _mw = _pipeline(sig, overlap, refine=False, modality=modality, poor_tier=poor_tier)
        ref, _ = _pipeline(sig, overlap, refine=True, modality=modality, poor_tier=poor_tier)
        raw_span = max(i.end_sec for i in raw if i.tier in ("Q0", "Q1")) - \
            min(i.start_sec for i in raw if i.tier in ("Q0", "Q1"))
        ref_span = max(i.end_sec for i in ref if i.tier in ("Q0", "Q1")) - \
            min(i.start_sec for i in ref if i.tier in ("Q0", "Q1"))
        if overlap == 0.0:
            assert ref == raw, (
                "overlap 0 on an exact-multiple record: no two windows overlap, so nothing is "
                "relaxable and refinement must not touch the output")
            assert not any(iv.model_tier for iv in ref)
        else:
            assert ref != raw, f"{modality} ov={overlap}: refinement is a NO-OP"
            assert ref_span < raw_span, (
                f"{modality} ov={overlap}: poor span {ref_span}s not tightened from {raw_span}s")


@pytest.mark.parametrize("modality", sorted(GEOMETRY))
def test_at_overlap_zero_the_end_anchored_tail_window_can_still_promote_honestly(modality):
    """The invariant "overlap 0 => refinement is a no-op" is FALSE, and asserting it would hand the next
    person to touch tail-anchoring a false green.

    make_windows END-ANCHORS the final window (start = n - L_m) so the trailing partial window is graded
    instead of silently dropped. Whenever the record is not a whole number of windows, that makes the
    LAST TWO windows genuinely overlap even at stride == L_m. A burst placed in the non-overlapping part
    of the second-to-last window therefore leaves the tail window clean, the two disagree, and refinement
    promotes the overlap zone -- which is HONEST, because a real model window really did grade it Q3.

    Pins that this happens, and that it stays honest when it does."""
    l_m, fs = GEOMETRY[modality]
    win = l_m / fs
    secs = win * 5.5                                  # deliberately NOT a whole number of windows
    # burst early in window 4 ([4*win, 5*win)), which the end-anchored tail window does NOT cover
    burst = (4.0 * win + 0.05 * win, 4.0 * win + 0.15 * win)
    sig = _burst_signal(secs=secs, burst=burst, modality=modality)

    raw, model_windows = _pipeline(sig, overlap=0.0, refine=False, modality=modality, poor_tier="Q0")
    ref, _ = _pipeline(sig, overlap=0.0, refine=True, modality=modality, poor_tier="Q0")

    starts = [w.start_sec for w in model_windows]
    assert starts[-1] != starts[-2] + win, "premise: the tail window must be end-anchored, not on-grid"
    tail_zone = (starts[-1], starts[-2] + win)        # the ONLY place two windows overlap at stride == L
    assert tail_zone[0] < tail_zone[1], "premise: the last two windows must genuinely overlap"

    assert ref != raw, (
        f"{modality}: the end-anchored tail window overlaps its predecessor, so refinement CAN promote "
        f"at overlap 0 -- if this is a no-op the tail window is not being fed to refine")

    # every promotion must lie inside that tail zone, and must be auditable
    for iv in ref:
        if iv.model_tier:
            assert iv.start_sec >= tail_zone[0] - 1e-9 and iv.end_sec <= tail_zone[1] + 1e-9, (
                f"{modality}: promotion at [{iv.start_sec}, {iv.end_sec}] lies OUTSIDE the tail-overlap "
                f"zone {tail_zone} -- at overlap 0 there is no other overlapping window to license it")

    _assert_honest(model_windows, ref)


# --- N2: the end-anchored tail window must not be timed on the uniform grid --------------------

@pytest.mark.parametrize("modality", ["ecg", "eeg"])
@pytest.mark.parametrize("overlap", OVERLAPS)
@pytest.mark.parametrize("refine", [False, True])
def test_tail_window_grade_is_not_attributed_past_the_end_of_the_record(modality, overlap, refine):
    """A 61.2 s ECG record is 6.12 windows: `make_windows` end-anchors the last window at [51.2, 61.2] so
    the tail is graded, which puts it OFF the `i * stride` grid. Timing it on the grid instead pushes
    the last interval past the end of a recording that stops at 61.2 s, which poisons the segment table,
    the durations, the bands drawn over the waveform and every exported start/end."""
    burst = {"ecg": (58.0, 59.0), "eeg": (58.0, 58.6)}[modality]
    sig = _burst_signal(secs=61.2, burst=burst, modality=modality)   # the burst sits IN the tail window
    record_sec = len(sig) / GEOMETRY[modality][1]                    # 61.2 s is not exact at every fs
    ivs, model_windows = _pipeline(sig, overlap=overlap, refine=refine, modality=modality)

    assert max(iv.end_sec for iv in ivs) <= record_sec + 1e-6, (
        f"an interval ends at {max(iv.end_sec for iv in ivs)}s, past the {record_sec}s record")
    assert _covers_contiguously(ivs, record_sec)                 # ... and the tail IS graded, not dropped
    assert sum(iv.duration_sec for iv in ivs) == pytest.approx(record_sec)
    tail = [iv for iv in ivs if iv.start_sec <= 60.0 < iv.end_sec]
    assert tail and tail[0].tier == "Q0", "the burst in the tail window must be flagged"
    _assert_honest(model_windows, ivs)


def test_the_tail_window_is_what_makes_the_grid_non_uniform():
    """Pins the premise of the test above to the REAL windowing (not to hand-typed numbers): the last
    start is end-anchored, i.e. not on the `i * stride` grid."""
    sig = _burst_signal(secs=61.2)
    starts = _WindowGrader().window_starts_sec(sig, overlap=0.0)
    assert starts.tolist() == [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 51.2]   # 51.2, not 60.0
    assert starts[-1] + WIN_SEC == pytest.approx(61.2)                    # the window ends AT the record


# --- N1: refinement must actually refine (it reads the model's WINDOWS, not the RLE output) ----

def test_refine_tightens_a_poor_run_around_a_short_burst_at_the_shipped_overlap():
    """The shipped 50% overlap. The model grades windows [5,15] and [10,20] poor for a 1 s burst, RLE
    displays Q0 over [7.5, 17.5] -- a 1 s artifact shown as a 10 s poor segment. Refinement must shrink
    that toward the burst, relaxing a clean flank bin ONLY to a grade the model itself gave a WINDOW
    covering that bin.

    This is the regression that a no-op refine hides: with refinement reading the (non-overlapping) RLE
    intervals, every bin has exactly one covering interval, the ceiling equals what is already shown,
    nothing is relaxable, and the 10 s smear is displayed unchanged.
    """
    sig = _burst_signal(secs=60.0, burst=(14.0, 15.0))
    raw, model_windows = _pipeline(sig, overlap=0.5, refine=False)
    ref, _ = _pipeline(sig, overlap=0.5, refine=True)

    raw_poor = [iv for iv in raw if iv.tier in ("Q0", "Q1")]
    assert [(iv.start_sec, iv.end_sec) for iv in raw_poor] == [(7.5, 17.5)]     # the 10 s RLE smear
    assert ref != raw, "refinement is a NO-OP: the poor run was not tightened at all"

    poor = [iv for iv in ref if iv.tier in ("Q0", "Q1")]
    assert poor, "the artifact must remain flagged"
    span = max(p.end_sec for p in poor) - min(p.start_sec for p in poor)
    assert span < 10.0 and span <= 6.5, f"poor span {span}s: not tightened toward the 1 s burst"
    assert min(p.start_sec for p in poor) <= 14.0 and max(p.end_sec for p in poor) >= 15.0  # covers it

    # the reclaimed flank carries the GOOD grade the model itself gave window [0, 10] -- and ITS
    # confidence, with the conservative grade preserved for the audit trail
    flank = [iv for iv in ref if iv.tier == "Q3" and iv.model_tier]
    assert flank, "a relaxed flank must record the model's conservative grade in model_tier"
    assert all(f.model_tier == "Q0" and np.isclose(f.confidence, GOOD_CONF) for f in flank)
    _assert_honest(model_windows, ref)


def test_refine_is_a_no_op_at_overlap_zero_on_an_exact_multiple_record():
    """Where no two windows overlap, the model made exactly ONE statement about each time, so there is
    nothing to refine and nothing may be promoted -- the honest answer, not a bug. The clean flank of a
    poor window must NOT be handed to the neighbouring good window: the model never graded that flank.

    60.0 s ECG is exactly 6 x 10 s windows, so the tail window is NOT end-anchored and this really is the
    no-overlap case. It is NOT the general "overlap 0" case -- see
    test_at_overlap_zero_the_end_anchored_tail_window_can_still_promote_honestly."""
    sig = _burst_signal(secs=60.0, burst=(14.0, 15.0))
    raw, model_windows = _pipeline(sig, overlap=0.0, refine=False)
    ref, _ = _pipeline(sig, overlap=0.0, refine=True)

    assert ref == raw                                            # untouched: nothing was relaxable
    assert not any(iv.model_tier for iv in ref)                  # ... so nothing was promoted
    poor = [(iv.start_sec, iv.end_sec) for iv in ref if iv.tier in ("Q0", "Q1")]
    assert poor == [(10.0, 20.0)]                                # the model's true resolution: 1 window
    _assert_honest(model_windows, ref)


@pytest.mark.parametrize("burst", [(6.0, 7.0), (11.0, 12.0), (14.0, 15.0), (18.0, 19.0), (49.0, 50.0)])
@pytest.mark.parametrize("overlap", OVERLAPS)
def test_refined_output_is_never_better_than_the_model_and_never_fabricates_a_number(burst, overlap):
    """The honesty invariant, over the real pipeline, across where the burst falls: for every emitted
    interval and every elementary span it covers, the displayed tier is never better than the BEST grade
    the model gave a WINDOW covering that span; no interval carries a confidence the model did not
    produce (the old code stamped a hardcoded 0.6 on every promoted window); every promotion is
    auditable through ``model_tier``.

    ``overlap=0.25`` with ``burst=(18, 19)`` is the case that stayed GREEN under the old bin-centre
    oracle: stride 7.5 s puts window starts at 7.5 / 22.5 / 37.5, so the tail of the poor run lands on a
    bin centre. Re-sampled sub-bin, the SAME fixture showed 9 violations around t=22.05..22.15 (displayed
    Q3, ceiling Q0).
    """
    sig = _burst_signal(secs=60.0, burst=burst)
    raw, model_windows = _pipeline(sig, overlap=overlap, refine=False)
    ref, _ = _pipeline(sig, overlap=overlap, refine=True)
    _assert_honest(model_windows, ref)
    assert not any(np.isclose(iv.confidence, 0.6) for iv in ref)      # the old fabricated constant
    assert any(iv.tier in ("Q0", "Q1") for iv in ref), "the artifact must never be refined away"
    for _t, out in _promoted(ref, raw):
        assert out.model_tier, "a promotion above the RLE grade must be auditable via model_tier"


def test_refine_keeps_a_single_poor_run_single():
    """Refinement erodes a poor run inward from its two edges. It must never relax a bin in the MIDDLE
    of a poor run (which would split it and claim clean signal between two artifacts)."""
    sig = _burst_signal(secs=60.0, burst=(11.0, 12.0))            # burst at the start of the poor run
    ref, model_windows = _pipeline(sig, overlap=0.5, refine=True)
    assert len([iv for iv in ref if iv.tier in ("Q0", "Q1")]) == 1
    _assert_honest(model_windows, ref)


def test_refine_does_not_fabricate_a_segment_over_signal_no_window_covers():
    """A time no window covers has no model grade at all -- it must be emitted as nothing, not as a
    default 'Q2' at a made-up confidence. (An interval list with a hole is not something RLE emits, so
    this one is a direct unit test of that guard -- it asserts an ABSENCE, which no promotion could
    fake.)"""
    sig = _burst_signal(secs=60.0, burst=(14.0, 15.0))
    ivs = [QI(0, 10, "Q3", GOOD_CONF, ()), QI(10, 20, "Q0", POOR_CONF, ("motion",)),
           QI(30, 60, "Q3", GOOD_CONF, ())]                       # [20, 30) covered by nothing
    mw = [QI(0, 10, "Q3", GOOD_CONF, ()), QI(5, 15, "Q0", POOR_CONF, ("motion",)),
          QI(10, 20, "Q0", POOR_CONF, ("motion",)), QI(15, 25, "Q3", GOOD_CONF, ())]
    ref = refine_intervals(ivs, sig, FS, "ecg", model_windows=mw)
    assert not any(iv.start_sec < 30 and iv.end_sec > 20 for iv in ref)      # the gap stays empty


def test_refine_leaves_clean_and_all_good_untouched():
    sig = (0.1 * np.sin(2 * np.pi * 1.1 * np.arange(int(60 * FS)) / FS)).astype(np.float32)
    ivs, model_windows = _pipeline(sig, overlap=0.5, refine=True)
    assert [iv.tier for iv in ivs] == ["Q3"]                      # no poor run -> nothing to refine
    assert ivs[0].end_sec == pytest.approx(60.0)
    _assert_honest(model_windows, ivs)


def test_refine_keeps_poor_run_with_no_localizable_core():
    """A poor segment whose signal has NO amplitude burst (diffuse in-band corruption) is trusted: with
    no core to erode toward, no boundary may move."""
    calm = (0.1 * np.sin(2 * np.pi * 1.1 * np.arange(int(30 * FS)) / FS)).astype(np.float32)
    ivs = [QI(0, 7.5, "Q3", GOOD_CONF, ()), QI(7.5, 17.5, "Q1", 0.5, ()), QI(17.5, 30, "Q3", GOOD_CONF, ())]
    mw = [QI(i * 5.0, i * 5.0 + 10.0, "Q1" if i in (1, 2) else "Q3",
             0.5 if i in (1, 2) else GOOD_CONF, ()) for i in range(5)]
    assert refine_intervals(ivs, calm, FS, "ecg", model_windows=mw) == ivs


def test_fine_badness_isolates_the_burst():
    bad, bs = fine_badness(_burst_signal(secs=60.0, burst=(49.0, 50.0)), FS, 1.0)
    assert bs == 250
    flagged = set(np.where(bad)[0].tolist())
    assert 49 in flagged                     # the burst second
    assert flagged <= {48, 49, 50}           # and essentially nothing else
