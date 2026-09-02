"""QThreadPool/QRunnable workers: decimation, chunk loading, inference, transcode (Plan 2 §9).

Each ``QRunnable.run()`` below does pure numpy/library work and emits its
result through the matching ``workers.signals`` carrier -- never touches
``QQuickItem``/scene-graph APIs directly (that discipline is enforced by
convention + code review per Plan 2 §14, since Python/Qt has no compiler-
level way to forbid it).
"""

from __future__ import annotations

import logging
import re

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Signal

from biosqa.inference.data_quality import record_quality
from biosqa.inference.llm_audit import audit_segment
from biosqa.inference.onnx_runner import OnnxRunner
from biosqa.inference.postprocess import (
    calibrate_grade_probs,
    calibrate_prediction,
    confidences_from,
    normalized_entropy,
)
from biosqa.inference.segmenter import (
    run_length_encode,
    threshold_artifact_labels,
    window_intervals,
)
from biosqa.io.loaders import read_window
from biosqa.io.pyramid import minmax_envelope_indices, samples_per_bucket
from biosqa.workers.signals import (
    AuditWorkerSignals,
    SaliencyWorkerSignals,
    InferenceWorkerSignals,
    LoadResampleWorkerSignals,
    StreamWorkerSignals,
    TranscodeWorkerSignals,
)

_log = logging.getLogger(__name__)

#: cap on the in-memory full-resolution plot cache (matches SignalViewController.loadTrace).
_PLOT_CACHE_CAP = 400000

#: the ONLY ``RuntimeError`` :func:`_emit` swallows: the carrier's C++ side is gone. PySide raises
#: ``'Signal source has been deleted'`` from ``emit()``; ``'Internal C++ object (X) already deleted.'``
#: is the sibling wording for a carrier collected a moment earlier. Everything else propagates.
_CARRIER_GONE = re.compile(r"(has been|already) deleted", re.IGNORECASE)

#: how far :func:`resample_ratio`'s cheap 1000-denominator approximation may miss the true rate ratio
#: before it escalates. 1e-4 is 0.36 s of drift per hour, far below any segment boundary the app draws,
#: so every ratio that was already accurate keeps its exact historical ``(up, down)`` -- and the pairs
#: that miss by 25%-100% (see :func:`resample_ratio`) are re-derived instead of shipped.
_RATIO_REL_TOL = 1e-4


def _emit(signal, *args) -> None:
    """Emit through a carrier that may already be gone.

    At shutdown Python can collect a carrier while its worker is still running, and then BOTH the
    result emit and the ``failed.emit`` in the handler below it raise ``RuntimeError: Signal source
    has been deleted`` — the second one from inside ``except``, so it escapes the Python override of
    ``QRunnable::run()``. A dead carrier means nobody is listening; that is not an error worth
    propagating out of a pool thread.

    ONLY that case is swallowed. Cross-thread emits are queued today, so a slot cannot raise inside
    ``emit()`` — but the moment one of these tasks is run inline on the GUI thread (a test double, a
    future synchronous path) the connection is DIRECT and a genuine ``RuntimeError`` from the
    receiving slot would come through here. Silently discarding that from a pool thread, with no log,
    is how a real bug becomes invisible: anything that is not the teardown case is re-raised."""
    try:
        signal.emit(*args)
    except RuntimeError as exc:   # carrier destroyed under the worker (teardown) — nothing to notify
        if not _CARRIER_GONE.search(str(exc)):
            raise


def resample_ratio(fs_in: float, fs_out: float) -> tuple[int, int]:
    """The ``(up, down)`` polyphase ratio :func:`resample_signal` resamples with, ``(1, 1)`` when it
    short-circuits (rates equal / degenerate).

    Exposed because a CHUNKED resample has to know it: ``resample_poly``'s polyphase phase at a block
    start depends on that block's start index MOD ``down``, so a streaming reader must cut its blocks on
    multiples of ``down`` or each block lands on a different phase of the filter than the whole-signal
    resample would (see :func:`inference.streaming.stream_infer`).

    ``limit_denominator`` is what keeps the pair small enough for a cheap polyphase filter, and a FIXED
    cap of 1000 is what used to break it: a 3-digit denominator's resolution runs out as the ratio
    shrinks, and it ran out well inside the rates this app accepts (``io.loaders`` calls anything up to
    20 kHz a plausible acquisition). Measured against the shipped 8 Hz EDA model, reachable today
    through force-modality on any high-rate recording:

    * 16001 Hz and up -- 20/22.05/24/30/32/44.1/48/96 kHz -- all gave ``(0, 1)``. ``up == 0`` is not a
      degraded resample but two wrong answers: ``resample_poly(sig, 0, 1)`` raises ``ValueError`` and
      the caller's fallback silently linearly interpolated with no antialiasing (20000 -> 8 Hz on a
      0.05 Hz tone plus a 3001 Hz one: std 0.9658 vs a true 0.7071, and the interferer folds to 1 Hz,
      inside the model's band), while streaming does not survive it at all -- ``stream_infer`` sizes
      its overlap-save margin as ``10 * max(up, down) / up`` and died on ``ZeroDivisionError`` at
      ``inference/streaming.py:146`` before reading a single block, i.e. the record got no analysis.
    * 16000 Hz gave ``(1, 1000)`` -- accepted by every resampler and wrong by a FACTOR OF TWO: a 300 s
      record came out 4800 samples long, i.e. 16 Hz fed to an 8 Hz model. Same for 10000 -> 8 Hz (25%
      high), 11025 -> 8 Hz (37.8%) and 96000 -> 64 Hz (50%). A silent 2x rate error is the worse of the
      two failure modes, so the escalation triggers on ACCURACY, not merely on ``up == 0``.

    Escalation therefore fires when the 1000-cap ratio is off by more than :data:`_RATIO_REL_TOL`,
    which no pair that resolved accurately before can hit; a genuinely inexpressible ratio raises
    instead of returning one no resampler accepts. Cost is unchanged -- polyphase work scales with the
    RATIO, not the denominator (44100 -> 256 Hz over a 300 s record: 0.16 s at ``(4, 689)``, 0.18 s at
    the exact ``(64, 11025)``)."""
    if fs_in <= 0 or abs(fs_in - fs_out) < 1e-3:
        return 1, 1
    from fractions import Fraction

    exact = Fraction(float(fs_out) / float(fs_in))
    frac = exact.limit_denominator(1000)
    if frac.numerator < 1 or abs(float(frac) - float(exact)) > _RATIO_REL_TOL * abs(float(exact)):
        # 100_000 spans every pair inside the app's own plausible band (0.5-20000 Hz, io.loaders)
        # against all four shipped model rates, exactly: 48 kHz -> 8 Hz is 1/6000, 44.1 kHz -> 256 Hz
        # is 64/11025.
        frac = exact.limit_denominator(100_000)
    if frac.numerator < 1:
        raise ValueError(f"degenerate resample ratio {fs_in} -> {fs_out} Hz")
    return frac.numerator, frac.denominator


def resample_signal(sig, fs_in: float, fs_out: float):
    """Resample a 1-D signal ``fs_in -> fs_out`` (anti-aliased where possible). Off-thread helper
    shared by :class:`LoadResampleTask` and the Coordinator so the full-signal resample never runs
    on the GUI thread."""
    sig = np.asarray(sig, dtype=np.float32)
    if fs_in <= 0 or abs(fs_in - fs_out) < 1e-3 or sig.size < 2:
        return sig
    up = down = None                    # named out here so the fallback can report what it tried
    try:
        from scipy.signal import resample_poly

        up, down = resample_ratio(fs_in, fs_out)
        return resample_poly(sig, up, down).astype(np.float32)
    except (ImportError, ValueError, MemoryError) as exc:
        # NARROW and LOUD. A bare ``except Exception`` here turned :func:`resample_ratio` returning
        # ``(0, 1)`` for every >16 kHz recording into a silent un-antialiased linear resample
        # (20000 -> 8 Hz: std 0.9658 against a true 0.7071, max|error| 1.01 on a 0.71-std signal)
        # instead of a reported defect.
        # ``ImportError`` is DEAD in a correctly installed app -- ``scipy>=1.11,<2`` is a hard
        # dependency in pyproject.toml, not an mne extra -- but it is the case this fallback was
        # written for, so it stays. ``ValueError`` is now only the genuinely degenerate ratio
        # :func:`resample_ratio` refuses to fake; linear is the best that can still be offered, and
        # the log is what makes it visible.
        _log.warning("resample_poly %g -> %g Hz (up=%s down=%s) failed; falling back to "
                     "un-antialiased linear interpolation: %s", fs_in, fs_out, up, down, exc)
        n_out = max(2, int(round(sig.size * fs_out / fs_in)))
        return np.interp(
            np.linspace(0.0, 1.0, n_out), np.linspace(0.0, 1.0, sig.size), sig
        ).astype(np.float32)


def build_plot_cache(raw, fs: float, cap: int = _PLOT_CACHE_CAP):
    """Primary-channel plot cache ``(full_t, full_y, lo, hi)`` — the work SignalViewController.loadTrace
    used to do inline on the GUI thread.

    Over the cap the cache is a MIN/MAX BUCKET ENVELOPE (:func:`io.pyramid.minmax_envelope_indices`),
    not ``raw[::stride]``: same point budget, but every extremum survives. No display path ever
    re-reads raw at a higher resolution on zoom, so a spike a stride skipped was gone for good."""
    raw = np.asarray(raw, dtype=np.float64)
    fs = float(fs) or 1.0
    t = np.arange(raw.shape[0], dtype=np.float64) / fs
    spb = samples_per_bucket(raw.shape[0], int(cap))
    if spb > 1:
        idx = minmax_envelope_indices(raw, spb)
        full_t = np.ascontiguousarray(t[idx])
        full_y = np.ascontiguousarray(raw[idx])
    else:
        full_t = np.ascontiguousarray(t)
        full_y = np.ascontiguousarray(raw)
    if raw.size:
        lo, hi = float(raw.min()), float(raw.max())
    else:
        lo, hi = -1.0, 1.0
    return full_t, full_y, lo, (hi if hi > lo else lo + 1.0)


class LoadResampleTask(QRunnable):
    """Off-thread OPEN work (Plan 2 §9): full read of the inference channel + resample to the
    model rate, plus the primary-channel plot cache — everything that used to freeze the GUI thread
    in ``Coordinator.on_recording_opened``. Emits one ``ready`` payload the Coordinator routes back
    to the (already-bound) viewmodels on the GUI thread; the event loop stays responsive throughout.

    ``cancel`` is the same cooperative token :class:`InferenceTask` takes: a superseded load (the user
    opened another recording) used to read and resample the WHOLE channel anyway and then emit — so the
    one path whose result could outlive its own carrier was also the one that always ran to completion.
    ``gen`` is the Coordinator's recording generation, carried IN THE PAYLOAD so the staleness check
    never depends on this carrier still being alive when the queued signal is delivered.
    """

    def __init__(self, handle, infer_ch: str, plot_ch: str, plot_channels: list,
                 fs_out: float, modality: str, signals: LoadResampleWorkerSignals,
                 cancel=None, gen: int | None = None):
        super().__init__()
        self.handle = handle
        self.infer_ch = infer_ch
        self.plot_ch = plot_ch
        self.plot_channels = list(plot_channels)
        self.fs_out = float(fs_out)
        self.modality = modality
        self.signals = signals
        self.cancel = cancel
        self.gen = gen

    def cancelled(self) -> bool:
        return self.cancel is not None and self.cancel.is_set()

    def run(self) -> None:
        try:
            h = self.handle
            fs_in = float(h.fs_hz[self.infer_ch])
            n = int(h.n_samples[self.infer_ch])
            raw_infer = np.asarray(read_window(h, [self.infer_ch], 0, n), dtype=np.float32).reshape(-1)
            if self.cancelled():
                return                                 # superseded: don't resample what nobody wants
            sig = resample_signal(raw_infer, fs_in, self.fs_out)   # inference signal @ model rate

            plot_fs = float(h.fs_hz[self.plot_ch])
            n_plot = int(h.n_samples[self.plot_ch])
            if self.plot_ch == self.infer_ch:
                raw_plot = raw_infer                                # avoid a second disk read
            else:
                if self.cancelled():
                    return
                raw_plot = np.asarray(read_window(h, [self.plot_ch], 0, n_plot)).reshape(-1)
            full_t, full_y, trace_lo, trace_hi = build_plot_cache(raw_plot, plot_fs)
            if self.cancelled():
                return                                 # a cancelled load is silent, like a cancelled run

            _emit(self.signals.ready, {
                "modality": self.modality, "handle": h, "sig": sig,
                "fs_in": fs_in, "fs_out": self.fs_out,
                "plot_channels": self.plot_channels, "plot_fs": plot_fs,
                "full_t": full_t, "full_y": full_y,
                "trace_lo": trace_lo, "trace_hi": trace_hi,
                "n_samples_primary": n_plot, "_rgen": self.gen,
            })
        except Exception as exc:  # noqa: BLE001 - surface to the UI, don't crash the pool
            if self.cancelled():
                return
            _emit(self.signals.failed, self.modality, str(exc))


class InferenceTask(QRunnable):
    """Runs sliding-window ONNX inference + RLE segmentation for one modality (Plan 2 §7.2/§7.3).

    ``cancel`` is a cooperative cancellation token (a ``threading.Event``): the Coordinator sets the
    PREVIOUS task's token when a new run supersedes it. A cancelled task returns at its next PHASE
    boundary and emits NOTHING AT ALL — not even ``failed``, which would clobber the live run's status.
    """

    def __init__(
        self,
        runner: OnnxRunner,
        signal,
        window_stride_sec: float,
        window_length_sec: float,
        signals: InferenceWorkerSignals,
        overlap: float = 0.0,
        guard_enabled: bool = True,
        bsqi_threshold: float = 0.72,
        recovery_enabled: bool = True,
        refine_enabled: bool = True,
        cancel=None,
    ):
        super().__init__()
        self.runner = runner
        self.signal = signal
        self.window_stride_sec = window_stride_sec
        self.window_length_sec = window_length_sec
        self.signals = signals
        self.overlap = float(overlap)
        self.guard_enabled = bool(guard_enabled)
        self.bsqi_threshold = float(bsqi_threshold)
        self.recovery_enabled = bool(recovery_enabled)
        self.refine_enabled = bool(refine_enabled)
        self.cancel = cancel

    def cancelled(self) -> bool:
        """True once the token is set — checked at every phase boundary (the phases are seconds-scale
        on a long record, and ``_recoverability`` is a whole SECOND ONNX pass)."""
        return self.cancel is not None and self.cancel.is_set()

    def run(self) -> None:
        try:
            if self.runner.card is None:
                raise RuntimeError("OnnxRunner.load() was not called before InferenceTask.run()")
            card = self.runner.card
            # One multi-head pass: the ordinal grade drives the tier track; the
            # optional multilabel artifact head decorates each window with its
            # glitch-type tags (empty for legacy single-head models).
            pred = self.runner.run_sliding_window_multihead(self.signal, overlap=self.overlap)
            if self.cancelled():
                return
            grade_order = card.primary_head.class_order
            worst_tier = grade_order[0].split("_")[0] if grade_order else "Q0"
            # Sanitation (non-finite -> uniform) + grade temperature scaling, from the SHARED
            # postprocess helper the streaming path also calls — the two paths must not derive
            # different confidences from the same signal.
            q_probs, non_finite = calibrate_grade_probs(pred.primary, card)
            # Emit the SHORT tier code ("Q0".."Q3") the filters / legend / band delegate expect —
            # class_order carries full labels ("Q0_unacceptable"), so split on "_" (a no-op if the card
            # already uses short codes).
            tiers = [grade_order[i].split("_")[0] for i in q_probs.argmax(axis=1)] if len(q_probs) else []
            confidences = confidences_from(q_probs, non_finite)

            # False-clean GUARD + record DATA-QUALITY (both best-effort; a failure here
            # must never break the primary segmentation). The integrity override re-flags
            # confidently-clean windows that the filter-robust bSQI marks corrupt on
            # pre-filtered input -> force them to the worst tier so the track shows them.
            # The guard reads P(unusable) off the usable head, so it gets the CALIBRATED prediction.
            if self.cancelled():
                return
            guard = (self._guard_report(tiers, worst_tier, calibrate_prediction(pred, card))
                     if self.guard_enabled else None)
            if self.cancelled():
                return                      # _data_quality_report EMITS — a cancelled run must not
            self._data_quality_report()

            artifacts_per_window = None
            art = card.artifact_head
            if art is not None:
                type_probs = pred.get(art.name)
                if type_probs is not None and len(type_probs):
                    artifacts_per_window = threshold_artifact_labels(
                        type_probs, art.class_order, art.threshold
                    )

            # Advisory recoverability: a SECOND pass on a filtered copy → which poor windows a
            # standard filter would lift to usable (never re-grades; the raw tier stays authoritative).
            # The single biggest chunk of a superseded run's wasted CPU, hence a token check first.
            if self.cancelled():
                return
            rec_pw, rtier_pw = (
                self._recoverability(tiers, grade_order)
                if self.recovery_enabled and len(tiers) else (None, None)
            )
            # Predictive uncertainty (research3 Rec.6): normalized entropy of the softmax the model
            # already returned — FREE (the workflow experiment found 8x TTA adds ~nothing over this).
            uncertainty_pw = normalized_entropy(q_probs)
            # Task-relative rate-usability: poor-morphology windows whose HR is still recoverable
            # (distinct from recoverability — catches wander/powerline the filtered pass misses).
            if self.cancelled():
                return
            rate_pw, hr_pw = self._rate_usability(tiers)
            # Conformal APS prediction set (research3 UQ): decode a coverage-guaranteed set per segment from
            # the already-calibrated grade distribution (q_probs was temperature-scaled above). Only when the
            # card ships a conformal threshold (else empty — e.g. modalities whose deployed model predates
            # the conformal calibration).
            conf_thr = getattr(card, "conformal_threshold", None)
            grade_probs_cal = q_probs if (conf_thr is not None and len(q_probs)) else None

            # The TRUE start time of every window the model scored. The grid is not uniform (the final
            # window is end-anchored so the tail is graded), so segmentation must bound its intervals on
            # these -- assuming `i * stride` shifts the tail window's grade past the end of the record.
            starts_sec = self._window_starts_sec(len(tiers))
            per_window = dict(
                artifacts_per_window=artifacts_per_window,
                recoverable_per_window=rec_pw, recovered_tier_per_window=rtier_pw,
                uncertainty_per_window=uncertainty_pw,
                rate_usable_per_window=rate_pw, hr_bpm_per_window=hr_pw,
                grade_probs_per_window=grade_probs_cal,
                class_order=[g.split("_")[0] for g in grade_order],
                conformal_threshold=conf_thr,
            )
            intervals = run_length_encode(
                tiers, confidences, self.window_stride_sec, self.window_length_sec,
                window_starts_sec=starts_sec, **per_window,
            )
            # Boundary refinement: localize a poor segment to its actual artefact using a fine
            # per-bin badness score (the model's coarse window smears a short burst; overlap can't
            # fix that — it makes it wider). Advisory; never breaks the primary segmentation.
            # It needs the model's PER-WINDOW grades, not the RLE output: RLE has already resolved each
            # multiply-covered time to ONE displayed grade, and the ceiling on an honest relaxation is
            # the BEST grade the model gave any window covering that time (see inference.refine).
            if self.cancelled():
                return
            if self.refine_enabled and starts_sec is not None:
                try:
                    from biosqa.inference.refine import refine_intervals
                    model_windows = window_intervals(
                        tiers, confidences, starts_sec, self.window_length_sec, **per_window,
                    )
                    intervals = refine_intervals(intervals, self.signal, float(card.fs_hz),
                                                 self.runner.modality, model_windows=model_windows)
                except Exception:  # noqa: BLE001
                    pass
            if self.cancelled():
                return
            _emit(self.signals.intervalsReady, self.runner.modality, intervals)
            if guard is not None:
                _emit(self.signals.guardReady, self.runner.modality, guard)
        except Exception as exc:  # noqa: BLE001
            if self.cancelled():
                return              # a cancelled run is silent even when it dies on the way out
            _emit(self.signals.failed, self.runner.modality, str(exc))

    def _window_starts_sec(self, n_windows: int):
        """The scored windows' real start times, or ``None`` when they cannot be established.

        ``None`` falls the segmenter back to the uniform ``i * stride`` grid — the pre-existing
        behaviour, correct whenever the record tiles evenly. A count mismatch means the starts do not
        describe THESE grades, and a wrong start time is a wrong segment time, so we decline to guess.
        """
        try:
            starts = self.runner.window_starts_sec(self.signal, overlap=self.overlap)
        except Exception:  # noqa: BLE001 - e.g. a runner stand-in without the card/method
            return None
        return starts if starts is not None and len(starts) == n_windows else None

    def _rate_usability(self, tiers: list):
        """Per-window rate-usability for ECG/PPG: for POOR windows only (cheap — the quality proxy is
        skipped on already-usable windows), is the heart/pulse RATE still reliably recoverable despite
        poor morphology? ECG gates on bSQI, PPG on pulse-band regularity (see task_usability). Returns
        ``(rate_usable[bool array], hr_bpm[float array])`` or ``(None, None)``."""
        modality = self.runner.modality
        if modality not in ("ecg", "ppg") or not len(tiers):
            return None, None
        try:
            from biosqa.inference.preprocess import make_windows
            from biosqa.inference.task_usability import rate_usability

            card = self.runner.card
            fs = float(card.fs_hz)
            windows = make_windows(self.signal, card, overlap=self.overlap)
            n = min(len(tiers), len(windows))
            rate = np.zeros(n, dtype=bool)
            hr = np.zeros(n, dtype=np.float64)
            for i in range(n):
                if tiers[i] not in ("Q0", "Q1"):     # only poor windows can gain a "rate-usable" tag
                    continue
                w = np.asarray(windows[i], dtype=np.float64)
                w = w[0] if w.ndim == 2 else w
                r = rate_usability(w, fs, modality)
                rate[i] = r["rate_usable"]
                hr[i] = r["hr_bpm"]
            return rate, hr
        except Exception:  # noqa: BLE001 - advisory only
            return None, None

    def _recoverability(self, tiers: list, grade_order):
        """Second inference pass on a filtered copy: which poor windows become usable after a
        standard per-modality filter? For ECG/PPG the call is corroborated by the filter-robust
        two-detector bSQI on the FILTERED window, so the model merely being *fooled* by filtering
        (the false-clean failure) is not mistaken for genuine recovery. Returns
        ``(recoverable[bool array], recovered_tier[list[str]])`` aligned to ``tiers``, or
        ``(None, None)`` on any failure — recovery is advisory and never breaks segmentation."""
        try:
            from biosqa.inference.preprocess import make_windows
            from biosqa.inference.recover import filter_for_modality, recoverable_windows

            card = self.runner.card
            fs = float(card.fs_hz)
            modality = self.runner.modality
            filtered = filter_for_modality(self.signal, fs, modality)
            gf = self.runner.run_sliding_window_multihead(filtered, overlap=self.overlap).primary
            if not len(gf):
                return None, None
            filtered_tiers = [grade_order[i].split("_")[0] for i in gf.argmax(axis=1)]
            m = min(len(tiers), len(filtered_tiers))
            rec, rtier = recoverable_windows(list(tiers[:m]), filtered_tiers[:m], grade_order)
            if modality == "ecg" and rec.any():      # bSQI corroboration is ECG-only (PPG = NN-only)
                from biosqa.inference.integrity import bsqi

                fwins = make_windows(filtered, card, overlap=self.overlap)
                for i in np.flatnonzero(rec):
                    if i < len(fwins):
                        w = np.asarray(fwins[i], dtype=np.float64)
                        w = w[0] if w.ndim == 2 else w
                        if bsqi(w, fs) < self.bsqi_threshold:  # detectors still disagree → deceptive
                            rec[i] = False
                            rtier[i] = ""
            if m < len(tiers):                                  # windowing mismatch → pad, don't crash
                rec = np.concatenate([rec, np.zeros(len(tiers) - m, dtype=bool)])
                rtier = list(rtier) + [""] * (len(tiers) - m)
            return rec, list(rtier)
        except Exception:  # noqa: BLE001 - advisory only
            return None, None

    def _guard_report(self, tiers: list, worst_tier: str, prediction=None):
        """Run the false-clean guard and apply its per-window integrity override to ``tiers``
        in place. Reuses the already-computed ``prediction`` (no second forward pass). Returns
        the guard report dict (or None on any failure)."""
        try:
            guard = self.runner.guard_record(self.signal, prediction, overlap=self.overlap,
                                              bsqi_corrupt=self.bsqi_threshold)
            mask = guard.get("override_mask")
            if mask is not None and len(mask) == len(tiers):
                for i, over in enumerate(mask):
                    if bool(over):
                        tiers[i] = worst_tier
            return {"prefiltered": bool(guard.get("prefiltered")),
                    "reasons": list(guard.get("reasons", [])),
                    "n_overridden": int(guard.get("n_overridden", 0)),
                    "score": float(guard.get("score", 0.0))}
        except Exception:  # noqa: BLE001 - guard is advisory; never break inference
            return None

    def _data_quality_report(self) -> None:
        """Emit the record-level data-quality report (completeness / validity / stability)."""
        try:
            from biosqa.inference.input_sanity import input_sanity

            card = self.runner.card
            dq = record_quality(self.signal, float(card.fs_hz))
            reg = input_sanity(self.signal, float(card.fs_hz))   # acquisition-regime / domain-shift report
            regime_flags = list(reg.flags)
            nov_frac, nov_top = self._novelty_fraction()         # feature-space novelty vs the training set
            if nov_frac > 0.08:                                  # >> the calibrated ~1% in-dist rate
                regime_flags.append(
                    f"{nov_frac:.0%} of windows have signal-quality features unlike the training set"
                    + (f" (mainly {nov_top})" if nov_top else "")
                    + " — possible new device/cohort; scores may not transfer.")
            _emit(self.signals.dataQualityReady, self.runner.modality, {
                "completeness": dq.completeness, "usable": dq.usable, "flags": list(dq.flags),
                "missing_frac": dq.missing_frac, "flatline_frac": dq.flatline_frac,
                "clipping_frac": dq.clipping_frac, "n_dropout_gaps": dq.n_dropout_gaps,
                "longest_gap_s": dq.longest_gap_s, "duration_s": dq.duration_s,
                "dsi": reg.dsi, "regime_flags": regime_flags,
                "f_edge_hz": reg.f_edge_hz, "band_ratio": reg.band_ratio,
                "novelty_frac": nov_frac, "novelty_top": nov_top,
            })
        except Exception:  # noqa: BLE001
            pass

    def _novelty_fraction(self):
        """Fraction of windows whose interpretable SQI vector is novel (Mahalanobis D² above the card's
        reference threshold) + the SQI that most often explains it. Robust cross-dataset signal: a new
        device/cohort lifts this far above the calibrated ~1% in-distribution rate. ``(0.0, "")`` when the
        card ships no novelty reference. Sub-sampled (≤200 windows) so it stays cheap on long records."""
        card = self.runner.card
        block = getattr(card, "novelty", None)
        if not block:
            return 0.0, ""
        try:
            from biosqa.inference.novelty import novelty_distance, sqi_feature_vector
            from biosqa.inference.preprocess import make_windows

            thr = float(block.get("d2_threshold", 1e18))
            fs = float(card.fs_hz)
            wins = make_windows(self.signal, card, overlap=self.overlap)
            if not len(wins):
                return 0.0, ""
            step = max(1, len(wins) // 200)
            novel_flags, tops = [], []
            for w in wins[::step]:
                ww = np.asarray(w, dtype=np.float64)
                ww = ww[0] if ww.ndim == 2 else ww
                feats, names = sqi_feature_vector(ww, fs, self.runner.modality)
                d2, top = novelty_distance(feats, block, names)     # names guard: skip if reordered
                novel_flags.append(d2 > thr)
                if d2 > thr and top:
                    tops.append(top)
            frac = float(np.mean(novel_flags)) if novel_flags else 0.0
            return frac, (max(set(tops), key=tops.count) if tops else "")
        except Exception:  # noqa: BLE001 - advisory only
            return 0.0, ""


class ChannelCacheTask(QRunnable):
    """Build one channel's bounded plot cache off-thread (block-wise, so a large channel neither
    blows memory nor freezes the GUI). Dispatched by SignalViewController when a non-primary lane is
    toggled on — the lane draws when ``ready`` fires."""

    def __init__(self, handle, channel: str, fs: float, signals: "ChannelCacheWorkerSignals"):
        super().__init__()
        self.handle = handle
        self.channel = channel
        self.fs = float(fs)
        self.signals = signals

    def run(self) -> None:
        try:
            from biosqa.inference.streaming import build_plot_cache_blockwise
            ft, fy, lo, hi = build_plot_cache_blockwise(self.handle, self.channel, self.fs)
            _emit(self.signals.ready, self.channel, ft, fy, float(lo), float(hi))
        except Exception as exc:  # noqa: BLE001
            _emit(self.signals.failed, self.channel, str(exc))


class StreamInferenceTask(QRunnable):
    """Out-of-core OPEN + inference for very long recordings (Plan 2 §6/§9): builds the plot cache
    block-by-block and runs sliding-window inference in a streaming pass — never holding the whole
    signal. Emits ``plotReady`` (bind the trace) then ``intervalsReady`` (the segmentation).

    BOUNDARY REFINEMENT runs here too, exactly as it does in :class:`InferenceTask` (same
    ``window_intervals`` -> ``refine_intervals`` call on the same per-window grades): a recording must
    not get coarser, window-resolution segment boundaries just because its file crossed
    ``streaming.LARGE_RECORD_SAMPLES``. It needs the whole scored signal (``fine_badness`` keys on
    whole-signal robust statistics), which ``stream_infer`` hands back when the record fits
    :data:`~biosqa.inference.streaming.REFINE_MAX_MODEL_SAMPLES`; past that it genuinely cannot run and
    the ``notice`` SAYS SO — as it does for the recoverability pass and the false-clean guard, which
    need a filtered/prefiltered whole-signal view and are always off in this mode.
    """

    def __init__(self, handle, infer_ch: str, plot_ch: str, plot_channels: list, modality: str,
                 runner, overlap: float, signals: StreamWorkerSignals, rebuild_plot: bool = True,
                 refine_enabled: bool = True, block_sec: float = 300.0, cancel=None,
                 gen: int | None = None):
        super().__init__()
        self.gen = gen                           # recording generation, carried in the plotReady payload
        self.handle = handle
        self.infer_ch = infer_ch
        self.plot_ch = plot_ch
        self.plot_channels = list(plot_channels)
        self.modality = modality
        self.runner = runner
        self.overlap = float(overlap)
        self.signals = signals
        self.rebuild_plot = bool(rebuild_plot)   # False on a settings re-run (keep the current view)
        self.refine_enabled = bool(refine_enabled)
        self.block_sec = float(block_sec)        # streamed read size; tests shrink it to force boundaries
        self.cancel = cancel                     # cooperative token, checked once per streamed block

    def cancelled(self) -> bool:
        return self.cancel is not None and self.cancel.is_set()

    def run(self) -> None:
        try:
            from biosqa.inference.streaming import build_plot_cache_blockwise, stream_infer

            h = self.handle
            if self.rebuild_plot:
                plot_fs = float(h.fs_hz[self.plot_ch])
                n_plot = int(h.n_samples[self.plot_ch])
                full_t, full_y, lo, hi = build_plot_cache_blockwise(h, self.plot_ch, plot_fs)
                if self.cancelled():
                    return
                _emit(self.signals.plotReady, {
                    "modality": self.modality, "handle": h, "sig": None, "streaming": True,
                    "_rgen": self.gen,
                    "plot_channels": self.plot_channels, "plot_fs": plot_fs,
                    "full_t": full_t, "full_y": full_y, "trace_lo": lo, "trace_hi": hi,
                    "n_samples_primary": n_plot,
                    "infer_ch": self.infer_ch, "fs_in": float(h.fs_hz[self.infer_ch]),
                    "fs_out": float(self.runner.card.fs_hz),
                })
            # `sig` is the model-rate signal the windows were cut from — retained ONLY when refinement
            # is on AND the record fits the streaming memory budget (else None, see the notice).
            (tiers, confs, arts, uncs, gprobs, starts_sec, stride_sec, window_sec, _n,
             sig) = stream_infer(h, self.infer_ch, self.runner, overlap=self.overlap,
                                 block_sec=self.block_sec, cancel=self.cancel,
                                 collect_signal=self.refine_enabled)
            if self.cancelled():
                return                     # stream_infer stopped mid-record: its windows are PARTIAL
            card = self.runner.card
            grade_order = card.primary_head.class_order
            # Same conformal decode as the in-memory path: the streamed grade probs are already
            # calibrated (stream_infer applies the card's grade temperature), which is what the
            # APS threshold was fit on.
            conf_thr = getattr(card, "conformal_threshold", None)
            # The streamed grid is non-uniform too (end-anchored tail window) — bound on the REAL start
            # times or the tail's grade lands past the end of the recording.
            starts = starts_sec if len(starts_sec) == len(tiers) else None
            per_window = dict(
                artifacts_per_window=arts,
                uncertainty_per_window=uncs,
                grade_probs_per_window=(gprobs if (conf_thr is not None and len(gprobs)) else None),
                class_order=[g.split("_")[0] for g in grade_order],
                conformal_threshold=conf_thr,
            )
            intervals = run_length_encode(tiers, confs, stride_sec, window_sec,
                                          window_starts_sec=starts, **per_window)
            # ``blocked`` is why refinement did NOT run despite being switched on — "" means it is in
            # force for this record. It drives the notice, so the app can never be silent about having
            # produced coarser boundaries than the setting promises.
            blocked, refined = "", False
            if self.refine_enabled:
                if sig is None:
                    blocked = "its analysis signal is too large to refine"
                elif starts is not None and len(tiers):
                    try:
                        from biosqa.inference.refine import refine_intervals
                        # The SAME two calls the in-memory path makes: refinement reads the model's raw
                        # PER-WINDOW grades (still overlapping), not the RLE output — the ceiling on an
                        # honest relaxation is the BEST grade any window covering that time was given.
                        model_windows = window_intervals(tiers, confs, starts, window_sec, **per_window)
                        intervals = refine_intervals(intervals, sig, float(card.fs_hz), self.modality,
                                                     model_windows=model_windows)
                        refined = True
                    except Exception:  # noqa: BLE001 - advisory; never break the primary segmentation
                        blocked = "boundary refinement could not be computed for it"
            if self.cancelled():
                return
            # The number of windows this pass actually scored, stamped on the carrier (the same idiom
            # the generation guards use) because the STREAMED count is not knowable at dispatch — the
            # record is read block by block. Without it the Coordinator divides by a dispatch-time 0
            # and reports latencyMs = 0.0 for exactly the recordings whose analysis takes longest.
            self.signals._n_windows = int(len(tiers))   # type: ignore[attr-defined]
            _emit(self.signals.intervalsReady, self.modality, intervals)
            _emit(self.signals.notice, self._notice(refined, blocked))
        except Exception as exc:  # noqa: BLE001
            if self.cancelled():
                return
            _emit(self.signals.failed, self.modality, str(exc))

    def _notice(self, refined: bool, blocked: str) -> str:
        """The streamed-mode notice — it must name EVERY analysis this record did not get, and why.

        The old text mentioned only recoverability + the guard while refinement was silently skipped as
        well. So the same signal came out with different segment boundaries either side of the streaming
        threshold, the "Refine boundaries" switch quietly did nothing, and the app's own explanation of
        the difference did not so much as mention it. Each of the three states below is asserted."""
        base = "Large recording — streaming analysis: "
        both_off = "the recoverability pass and the false-clean guard are disabled in this mode."
        if blocked:                     # refinement is ON but could not run — never leave this unsaid
            return base + ("boundary refinement, the recoverability pass and the false-clean guard are "
                           f"ALL disabled for this record ({blocked}) — 'Refine boundaries' has no "
                           "effect here and the segment boundaries stay at window resolution.")
        if refined:
            return base + "boundary refinement is applied; " + both_off
        return base + both_off          # refinement off in Settings, or nothing was graded to refine


class AuditTask(QRunnable):
    """On-demand LLM AUDIT of one selected window (off the decision path, Plan 1 §9).

    Runs :func:`biosqa.inference.llm_audit.audit_segment` on the pool so the (seconds-scale,
    local-ollama) call never blocks the GUI thread. Degrades gracefully: if ollama is
    unreachable ``audit_segment`` returns ``{"error": ...}`` rather than raising.
    """

    def __init__(self, runner: OnnxRunner, window, model_grade: dict, guard: dict | None,
                 signals: AuditWorkerSignals, model: str = "qwen3:32b", samples: int = 3,
                 host: str = "http://localhost:11434", timeout: float = 60.0, cancel=None):
        super().__init__()
        self.runner = runner
        self.window = window
        self.model_grade = model_grade
        self.guard = guard
        self.signals = signals
        self.model = model
        self.samples = samples
        self.host = host
        self.timeout = float(timeout)
        #: cooperative token; the audit pool is single-threaded, so a queued audit that is no longer
        #: wanted (app quitting / another recording opened) must not start its own ollama round trip.
        self.cancel = cancel

    def cancelled(self) -> bool:
        return self.cancel is not None and self.cancel.is_set()

    def run(self) -> None:
        if self.cancelled():
            return
        try:
            fs = float(self.runner.card.fs_hz)
            judgment = audit_segment(self.window, fs, self.runner.modality,
                                     model_grade=self.model_grade, guard=self.guard,
                                     model=self.model, samples=self.samples, host=self.host,
                                     timeout=self.timeout)
            if self.cancelled():
                return
            _emit(self.signals.auditReady, judgment)
        except Exception as exc:  # noqa: BLE001
            if self.cancelled():
                return
            _emit(self.signals.failed, str(exc))


class SaliencyTask(QRunnable):
    """On-demand EXPLAIN of the selected segment (XAI): both the spatial occlusion-SALIENCY heatmap (*where*
    in time — :mod:`biosqa.inference.saliency`) and the group-Shapley feature ATTRIBUTION (*which* quality
    property drives the grade — :mod:`biosqa.inference.feature_attribution`, fusion models only). Both are
    gradient-free perturbation methods needing only forward passes (~0.1-1 s total), run off the GUI thread.
    Downsamples the per-sample saliency to ``n_out`` points for QML; emits both in one payload."""

    def __init__(self, runner: OnnxRunner, window, signals: "SaliencyWorkerSignals",
                 target: str = "unusable", n_out: int = 256):
        super().__init__()
        self.runner = runner
        self.window = window
        self.signals = signals
        self.target = target
        self.n_out = int(n_out)

    def run(self) -> None:
        try:
            from biosqa.inference.saliency import signal_saliency
            sal = signal_saliency(self.window, self.runner, target=self.target)
            n = sal.size
            if n > self.n_out:                              # block-max downsample (keep peaks) for the UI
                idx = np.linspace(0, n, self.n_out + 1).astype(int)
                sal = np.array([sal[idx[i]:max(idx[i] + 1, idx[i + 1])].max() for i in range(self.n_out)])
            attribution = None
            try:                                            # feature attribution is advisory + fusion-only
                from biosqa.inference.feature_attribution import grade_group_attribution
                attribution = grade_group_attribution(self.window, self.runner)
            except Exception:  # noqa: BLE001
                attribution = None
            _emit(self.signals.saliencyReady,
                  {"map": [round(float(v), 4) for v in sal], "n": int(n), "attribution": attribution})
        except Exception:  # noqa: BLE001 - advisory only
            _emit(self.signals.saliencyReady, {"map": [], "n": 0, "attribution": None})


class SqiWorkerSignals(QObject):
    """Carrier for :class:`SqiTask` (lives here next to its task; ``workers.signals`` holds the older
    carriers). ``sqiReady`` payload: ``{rows, filtered, consensus, usability}``."""

    sqiReady = Signal(object)


class SqiTask(QRunnable):
    """The interpretable classical-SQI breakdown + task-usability verdicts for one selected window.

    This ran DIRECTLY on the GUI thread. It is not cheap: the bank is computed TWICE (raw, then a
    band-pass-filtered copy for the Raw/Filtered toggle) and its beat/peak-detection stages are
    seconds-scale on a long segment — a 40-minute clean ECG run-length-encodes to ONE Q3 segment, and
    selecting it froze the window for ~10-20 s, again on every re-selection and boundary nudge. Same
    off-thread shape as :class:`SaliencyTask`; the panel fills in when the result lands."""

    def __init__(self, window, fs: float, modality: str, signals: "SqiWorkerSignals"):
        super().__init__()
        self.window = window
        self.fs = float(fs)
        self.modality = modality
        self.signals = signals

    def run(self) -> None:
        try:
            from biosqa.inference.sqi_breakdown import sqi_breakdown, sqi_consensus
            rows = sqi_breakdown(self.window, self.fs, self.modality)
            # the SAME bank on a band-pass-filtered copy (Raw/Filtered toggle): shows what a standard
            # filter would/would not fix — e.g. baseline wander clears, in-band EMG does not.
            try:
                from biosqa.inference.recover import filter_for_modality
                filt_rows = sqi_breakdown(filter_for_modality(self.window, self.fs, self.modality),
                                          self.fs, self.modality)
            except Exception:  # noqa: BLE001 - filtered view is advisory
                filt_rows = []
            consensus = sqi_consensus(rows)      # consensus is on the RAW bank (what the model saw)
        except Exception:  # noqa: BLE001
            rows, filt_rows, consensus = [], [], 0.0
        try:  # per-modality "usable for what" verdicts (EEG per-band, EDA tonic/phasic)
            from biosqa.inference.task_usability import usability_verdicts
            usability = usability_verdicts(self.window, self.fs, self.modality)
        except Exception:  # noqa: BLE001
            usability = []
        _emit(self.signals.sqiReady, {"rows": rows, "filtered": filt_rows,
                                      "consensus": consensus, "usability": usability})


class TranscodeTask(QRunnable):
    """One-time foreign-format -> Zarr transcode + pyramid build (Plan 2 §6.1).

    TODO(Plan2 §6.1): implement the actual transcode body (open via
    ``io.loaders``, write via ``io.store.RecordingStore.create``, build
    levels via ``io.pyramid.build_minmax_pyramid``); currently a stub that
    reports immediate completion so the UI plumbing can be exercised end to
    end before the heavy lifting lands.
    """

    def __init__(self, recording_path: str, signals: TranscodeWorkerSignals):
        super().__init__()
        self.recording_path = recording_path
        self.signals = signals

    def run(self) -> None:
        raise NotImplementedError(
            "TranscodeTask.run: transcode-to-Zarr + pyramid build not yet implemented "
            "(TODO Plan2 §6.1, Phase 1)"
        )


__all__ = [
    "LoadResampleTask",
    "InferenceTask",
    "StreamInferenceTask",
    "ChannelCacheTask",
    "AuditTask",
    "SaliencyTask",
    "SqiTask",
    "SqiWorkerSignals",
    "TranscodeTask",
]
