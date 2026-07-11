"""Coordinator: the connective tissue wiring recording-open -> ONNX inference -> UI (Plan 2 §7/§9).

This is the layer the app audit found missing. On ``recordings.recordingOpened`` it:
  1. looks up the lazily-opened handle + detected modality,
  2. loads (and caches) the matching ``OnnxRunner`` + model card,
  3. populates the channel list,
  4. reads the preferred channel in full and resamples it to the model's canonical rate,
  5. dispatches an :class:`InferenceTask` on a shared ``QThreadPool``, and
  6. routes the resulting quality intervals + status back to the bound viewmodels on the GUI thread.

Worker signals use Qt's auto/queued connection, so every viewmodel slot runs safely on the GUI
thread even though inference runs on the pool.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QThreadPool, QTimer, Slot

from biosqa.inference.onnx_runner import OnnxRunner
from biosqa.io.loaders import RecordingHandle, read_window
from biosqa.viewmodels.channel_model import ChannelEntry
from biosqa.workers.qt_threads import (
    AuditTask,
    SaliencyTask,
    InferenceTask,
    LoadResampleTask,
    StreamInferenceTask,
    resample_signal,
)
from biosqa.workers.signals import (
    AuditWorkerSignals,
    SaliencyWorkerSignals,
    InferenceWorkerSignals,
    LoadResampleWorkerSignals,
    StreamWorkerSignals,
)

# channel-name tokens per modality (for choosing which channel to run inference on)
_PREF_TOKENS = {
    "ecg": ("ii", "mlii", "ecg", "ekg", "lead"),
    "ppg": ("pleth", "ppg", "bvp", "pulse"),
    "eda": ("eda", "gsr", "scl", "scr"),
    "eeg": ("fpz", "fp1", "fp2", "cz", "eeg", "oz", "pz"),
}

#: the full-signal read + resample now runs off-thread in LoadResampleTask; kept as an alias so any
#: existing reference (tests) to ``coordinator._resample`` keeps working.
_resample = resample_signal


def _preferred_channel(handle: RecordingHandle, modality: str) -> str:
    """Pick the channel to run inference on: first name matching the modality, else the first."""
    toks = _PREF_TOKENS.get(modality, ())
    for name in handle.channel_names:
        low = name.lower()
        if any(t in low for t in toks):
            return name
    return handle.channel_names[0]


class Coordinator(QObject):
    """Wires the open->inference->viewmodel pipeline. Constructed once in ``build_engine``."""

    def __init__(self, *, models_dir, recordings, channels, segments, inference,
                 model_card, signal_view=None, selection=None, guard=None, settings=None,
                 parent=None) -> None:
        super().__init__(parent)
        self._models_dir = Path(models_dir)
        self._recordings = recordings
        self._channels = channels
        self._segments = segments
        self._inference = inference
        self._model_card = model_card
        self._signal_view = signal_view
        self._selection = selection
        self._guard = guard
        self._settings = settings
        self._pool = QThreadPool.globalInstance()
        # LLM audits get a DEDICATED single-thread pool: a slow/hung local ollama call (up to
        # samples×timeout) would otherwise occupy a shared-pool slot and starve inference/saliency. On its
        # own pool it can only delay other audits.
        self._audit_pool = QThreadPool()
        self._audit_pool.setMaxThreadCount(1)
        self._runners: dict[str, OnnxRunner] = {}
        self._carriers: list[InferenceWorkerSignals] = []  # keep signal carriers alive
        self._audit_carriers: list[AuditWorkerSignals] = []
        self._saliency_carriers: list[SaliencyWorkerSignals] = []
        self._selection_gen = 0   # bumped on each segment select; a saliency result from an old selection is dropped
        self._load_carriers: list[LoadResampleWorkerSignals] = []
        self._stream_carriers: list[StreamWorkerSignals] = []
        self._pending: dict[str, tuple[float, int]] = {}   # modality -> (t0, n_windows)
        self._current: tuple[str, "np.ndarray"] | None = None  # (modality, resampled signal) for audit
        #: for a STREAMED (large) recording there is no in-memory signal; audit re-reads the selected
        #: window from the handle. (modality, handle, infer_channel, fs_in, fs_out).
        self._current_stream: tuple | None = None
        self._guard_reports: dict[str, dict] = {}          # modality -> latest guard report
        # Generation guards: every task carrier is stamped with the generation it was dispatched
        # under; a result whose stamp is stale (a newer open / re-run superseded it) is DROPPED, so
        # opening another file or changing a setting mid-flight can't load old data into the new view.
        self._recording_gen = 0    # bumped on every open (guards plot binding + current-signal state)
        self._inference_gen = 0    # bumped on every inference dispatch (guards intervals/guard/notice)
        if selection is not None and hasattr(selection, "attach_segments"):
            selection.attach_segments(segments)
        # Bump the selection generation from the AUTHORITATIVE selection signal, so a saliency result from a
        # superseded segment is dropped even if the UI didn't happen to re-request SQIs. (Previously the
        # only bump was a side-effect of `_on_sqi_requested`, i.e. correctness depended on QML firing
        # requestSqi on every selection change — fragile.)
        if selection is not None and hasattr(selection, "selectedSegmentChanged"):
            selection.selectedSegmentChanged.connect(self._on_selection_changed)
        if guard is not None:
            guard.auditRequested.connect(self._on_audit_requested)
            guard.sqiRequested.connect(self._on_sqi_requested)
            guard.saliencyRequested.connect(self._on_saliency_requested)
        # Analysis-setting changes re-segment the OPEN recording, but DEBOUNCED: a bSQI-slider drag
        # (or a burst of toggles) emits many signals; without coalescing each dispatched a full
        # (double, under recovery) inference pass. The timer collapses a burst into one re-run.
        # Overlap affects both paths → full re-run/re-stream; guard/recovery/bSQI/refine are disabled
        # in streaming mode, so they only re-run normal records.
        self._rerun_timer = QTimer(self)
        self._rerun_timer.setSingleShot(True)
        self._rerun_timer.setInterval(200)
        self._rerun_timer.timeout.connect(self._do_scheduled_rerun)
        self._pending_overlap = False
        overlap_sig = getattr(settings, "windowOverlapChanged", None)
        if overlap_sig is not None:
            overlap_sig.connect(self._schedule_rerun)
        for sig_name in ("recoveryEnabledChanged", "refineBoundariesChanged",
                         "guardEnabledChanged", "bsqiThresholdChanged"):
            sig = getattr(settings, sig_name, None)
            if sig is not None:
                sig.connect(self._schedule_rerun_normal)
        recordings.recordingOpened.connect(self.on_recording_opened)

    @staticmethod
    def _prune(carriers: list, attr: str, current: int) -> list:
        """Drop carriers whose generation stamp is stale (an in-flight QRunnable keeps its own carrier
        alive until it finishes, so this only frees already-superseded ones)."""
        return [c for c in carriers if getattr(c, attr, current) == current]

    @Slot()
    def _schedule_rerun(self) -> None:
        self._pending_overlap = True
        self._rerun_timer.start()

    @Slot()
    def _schedule_rerun_normal(self) -> None:
        self._rerun_timer.start()

    def _do_scheduled_rerun(self) -> None:
        if self._pending_overlap:
            self._pending_overlap = False
            self._rerun_inference()          # overlap changed → full re-run/re-stream
        else:
            self._rerun_inference_normal()

    def _runner(self, modality: str) -> OnnxRunner:
        runner = self._runners.get(modality)
        if runner is None:
            runner = OnnxRunner(modality, self._models_dir)
            runner.load()
            self._runners[modality] = runner
        return runner

    def _stale(self, gen_attr: str, current: int) -> bool:
        """True if the emitting carrier's generation stamp (``gen_attr``) is not ``current`` — i.e. a
        newer open / re-run has superseded it, so this result must be ignored."""
        return getattr(self.sender(), gen_attr, current) != current

    @Slot(str)
    def on_recording_opened(self, path: str) -> None:
        info = self._recordings.handle_for(path)
        if info is None:
            return
        handle, modality = info
        if self._selection is not None and hasattr(self._selection, "clear"):
            self._selection.clear()
        if self._guard is not None:
            self._guard.reset()
        try:
            runner = self._runner(modality)
        except Exception as exc:  # noqa: BLE001
            self._inference.report(f"Model load failed ({modality}): {exc}", "", 0.0)
            return
        card = runner.card

        # model-card panel + channel list
        try:
            self._model_card.load(str(self._models_dir / f"{modality}.model_card.json"))
        except Exception:  # noqa: BLE001 - a missing card panel shouldn't block inference
            pass
        # Only the first (primary/plot) channel is visible by default, so the plot opens as a
        # single lane (unchanged look); toggling other channels' eye adds stacked lanes.
        self._channels.set_channels([
            ChannelEntry(name=n, modality=modality, unit=getattr(handle, "units", {}).get(n, ""),
                         visible=(i == 0))
            for i, n in enumerate(handle.channel_names)
        ])

        # Heavy work (full-channel read + resample + primary-channel plot cache) goes OFF-THREAD so
        # on_recording_opened returns to the event loop immediately (no freeze). Very long records
        # take the OUT-OF-CORE path (StreamInferenceTask): block-wise plot cache + streaming
        # inference, never holding the whole signal — so multi-day / big multichannel files don't OOM.
        from biosqa.inference.streaming import LARGE_RECORD_SAMPLES, estimate_analysis_samples

        infer_ch = _preferred_channel(handle, modality)
        plot_ch = handle.channel_names[0]
        self._current = None
        self._current_stream = None
        self._recording_gen += 1
        self._inference_gen += 1
        rgen, igen = self._recording_gen, self._inference_gen
        self._load_carriers = self._prune(self._load_carriers, "_rgen", rgen)
        self._stream_carriers = self._prune(self._stream_carriers, "_rgen", rgen)
        self._audit_carriers = self._prune(self._audit_carriers, "_rgen", rgen)
        self._saliency_carriers = self._prune(self._saliency_carriers, "_rgen", rgen)
        # Treat the threshold as a COMPUTE budget, not a raw sample count: recovery runs a second
        # full inference pass, so it ~doubles the effective work — fold that into the decision so an
        # "almost too big" record with recovery on also takes the streaming path.
        threshold = int(getattr(self._settings, "streamingThresholdSamples", LARGE_RECORD_SAMPLES)) \
            if self._settings else LARGE_RECORD_SAMPLES
        recovery_on = bool(getattr(self._settings, "recoveryEnabled", True)) if self._settings else True
        effective = estimate_analysis_samples(handle, infer_ch) * (2 if recovery_on else 1)
        if effective >= threshold:
            self._inference.report(f"Streaming {modality.upper()}…", card.model_version, 0.0)
            overlap = float(getattr(self._settings, "windowOverlap", 0.0)) if self._settings else 0.0
            runner = self._runners[modality]
            carrier = StreamWorkerSignals()
            carrier._rgen, carrier._igen = rgen, igen  # type: ignore[attr-defined]
            carrier.plotReady.connect(self._on_stream_plot)
            carrier.intervalsReady.connect(self._on_intervals)
            carrier.notice.connect(self._on_notice)
            carrier.failed.connect(self._on_failed)
            self._stream_carriers.append(carrier)
            self._pending[modality] = (time.perf_counter(), 0)
            self._pool.start(StreamInferenceTask(handle, infer_ch, plot_ch,
                                                 list(handle.channel_names), modality, runner,
                                                 overlap, carrier))
            return
        self._inference.report(f"Loading {modality.upper()}…", card.model_version, 0.0)
        carrier = LoadResampleWorkerSignals()
        carrier._rgen, carrier._igen = rgen, igen  # type: ignore[attr-defined]
        carrier.ready.connect(self._on_loaded)
        carrier.failed.connect(self._on_load_failed)
        self._load_carriers.append(carrier)
        self._pool.start(LoadResampleTask(handle, infer_ch, plot_ch, list(handle.channel_names),
                                          float(card.fs_hz), modality, carrier))

    @Slot(object)
    def _on_loaded(self, payload) -> None:
        """LoadResampleTask finished off-thread: bind the plot from the precomputed cache, keep the
        resampled signal for audit/re-run, then dispatch inference — all back on the GUI thread."""
        if self._stale("_rgen", self._recording_gen):
            return                                # a newer open superseded this load
        modality = payload["modality"]
        sig = payload["sig"]
        self._current = (modality, sig)          # kept for on-demand audit + live re-segment
        if self._signal_view is not None and hasattr(self._signal_view, "set_recording_cached"):
            try:
                self._signal_view.set_recording_cached(
                    payload["handle"], payload["plot_channels"], payload["plot_fs"],
                    payload["full_t"], payload["full_y"],
                    payload["trace_lo"], payload["trace_hi"], payload["n_samples_primary"])
            except Exception:  # noqa: BLE001 - a plot-bind failure shouldn't block inference
                pass
        self._start_inference(modality, sig)

    @Slot(object)
    def _on_stream_plot(self, payload) -> None:
        """Streamed (large) recording: bind the block-wise plot cache; keep the handle/channel so
        on-demand audit can re-read the selected window (there is no in-memory signal)."""
        if self._stale("_rgen", self._recording_gen):
            return
        modality = payload["modality"]
        self._current = None
        self._current_stream = (modality, payload["handle"], payload["infer_ch"],
                                float(payload["fs_in"]), float(payload["fs_out"]))
        if self._signal_view is not None and hasattr(self._signal_view, "set_recording_cached"):
            try:
                self._signal_view.set_recording_cached(
                    payload["handle"], payload["plot_channels"], payload["plot_fs"],
                    payload["full_t"], payload["full_y"],
                    payload["trace_lo"], payload["trace_hi"], payload["n_samples_primary"])
            except Exception:  # noqa: BLE001
                pass

    @Slot(str)
    def _on_notice(self, message: str) -> None:
        if self._stale("_igen", self._inference_gen):
            return
        if self._guard is not None and hasattr(self._guard, "reset"):
            self._guard.reset()   # streaming mode has no guard/recovery report to show
        self._inference.report(message, "", 0.0)

    def _start_inference(self, modality: str, sig) -> None:
        """Dispatch sliding-window inference over an already-loaded signal using the CURRENT settings.
        Called on load-complete and again on any analysis-setting change (live re-segment, no re-read)."""
        runner = self._runners.get(modality)
        if runner is None or runner.card is None or sig is None or getattr(sig, "size", 0) < 1:
            return
        card = runner.card
        window_len_sec = card.l_m / float(card.fs_hz)
        # user-configurable window overlap (0 / 0.25 / 0.5) -> stride; guard enable + bSQI threshold
        overlap = float(getattr(self._settings, "windowOverlap", 0.0)) if self._settings else 0.0
        overlap = min(max(overlap, 0.0), 0.9)
        stride_samples = max(1, int(round(card.l_m * (1.0 - overlap))))
        stride_sec = window_len_sec * (1.0 - overlap)
        guard_enabled = bool(getattr(self._settings, "guardEnabled", True)) if self._settings else True
        bsqi_threshold = float(getattr(self._settings, "bsqiThreshold", 0.72)) if self._settings else 0.72
        recovery_enabled = bool(getattr(self._settings, "recoveryEnabled", True)) if self._settings else True
        refine_enabled = bool(getattr(self._settings, "refineBoundaries", True)) if self._settings else True
        n_windows = max(0, (sig.size - card.l_m) // stride_samples + 1)
        self._inference_gen += 1
        self._carriers = self._prune(self._carriers, "_igen", self._inference_gen)
        carrier = InferenceWorkerSignals()
        carrier._igen = self._inference_gen  # type: ignore[attr-defined]
        carrier.intervalsReady.connect(self._on_intervals)
        carrier.guardReady.connect(self._on_guard)
        carrier.dataQualityReady.connect(self._on_data_quality)
        carrier.failed.connect(self._on_failed)
        self._carriers.append(carrier)
        self._pending[modality] = (time.perf_counter(), n_windows)
        self._inference.report(f"Running {modality.upper()}…", card.model_version, 0.0)
        self._pool.start(InferenceTask(runner, sig, stride_sec, window_len_sec, carrier,
                                       overlap=overlap, guard_enabled=guard_enabled,
                                       bsqi_threshold=bsqi_threshold,
                                       recovery_enabled=recovery_enabled,
                                       refine_enabled=refine_enabled))

    @Slot()
    def _rerun_inference_normal(self) -> None:
        """Re-segment a NORMAL (in-memory) recording after a guard/recovery/bSQI/refine change. A
        streamed record disables those features, so this is a no-op there (no wasted re-stream)."""
        if self._current is not None:
            modality, sig = self._current
            self._start_inference(modality, sig)

    @Slot()
    def _rerun_inference(self) -> None:
        """Re-segment after an OVERLAP change (affects both paths). Normal records reuse the in-memory
        signal; a streamed record re-streams (keeping the current view — no plot rebuild)."""
        if self._current is not None:
            modality, sig = self._current
            self._start_inference(modality, sig)
            return
        if self._current_stream is not None:
            modality, handle, infer_ch, _fi, _fo = self._current_stream
            runner = self._runners.get(modality)
            if runner is None:
                return
            overlap = float(getattr(self._settings, "windowOverlap", 0.0)) if self._settings else 0.0
            self._inference_gen += 1
            self._stream_carriers = self._prune(self._stream_carriers, "_igen", self._inference_gen)
            carrier = StreamWorkerSignals()
            carrier._rgen, carrier._igen = self._recording_gen, self._inference_gen  # type: ignore[attr-defined]
            carrier.intervalsReady.connect(self._on_intervals)
            carrier.notice.connect(self._on_notice)
            carrier.failed.connect(self._on_failed)
            self._stream_carriers.append(carrier)
            self._pending[modality] = (time.perf_counter(), 0)
            self._pool.start(StreamInferenceTask(handle, infer_ch, handle.channel_names[0],
                                                 list(handle.channel_names), modality, runner,
                                                 overlap, carrier, rebuild_plot=False))

    @Slot(str, str)
    def _on_load_failed(self, modality: str, message: str) -> None:
        if self._stale("_rgen", self._recording_gen):
            return
        self._inference.report(f"{modality.upper()} load failed: {message}", "", 0.0)

    @Slot(str, object)
    def _on_intervals(self, modality: str, intervals) -> None:
        if self._stale("_igen", self._inference_gen):
            return                                # a newer inference/open superseded this result
        self._segments.load_intervals(intervals)
        t0, n_windows = self._pending.get(modality, (None, 0))
        latency = 0.0
        if t0 is not None and n_windows > 0:
            latency = (time.perf_counter() - t0) * 1000.0 / n_windows
        runner = self._runners.get(modality)
        version = runner.card.model_version if runner and runner.card else ""
        n = len(list(intervals)) if intervals is not None else 0
        self._inference.report(f"{modality.upper()} · {n} segments", version, latency)

    @Slot(str, object)
    def _on_guard(self, modality: str, report) -> None:
        if self._stale("_igen", self._inference_gen):
            return
        self._guard_reports[modality] = report or {}
        if self._guard is not None:
            self._guard.setGuard(modality, report)

    @Slot(str, object)
    def _on_data_quality(self, modality: str, report) -> None:
        if self._stale("_igen", self._inference_gen):
            return
        if self._guard is not None:
            self._guard.setDataQuality(modality, report)

    @Slot(float, float, str, float)
    def _on_audit_requested(self, start_sec: float, end_sec: float, tier: str, confidence: float) -> None:
        """Slice the selected segment's window out of the current signal and run an off-path LLM audit."""
        if self._settings is not None and not self._settings.auditEnabled:
            if self._guard is not None:
                self._guard.setAuditResult({"error": "LLM audit disabled in settings"})
            return
        got = self._audit_window(start_sec, end_sec)
        if got is None:
            if self._guard is not None:
                self._guard.setAuditResult({"error": "no recording loaded"})
            return
        modality, runner, window = got
        grade = int(tier[1]) if len(tier) > 1 and tier[1].isdigit() else -1
        # `confidence` is the model's confidence in the PREDICTED grade (max softmax), NOT usability.
        # Sent verbatim as p_usable it lies to the auditor: a confidently-poor Q0 becomes 90% usable.
        # Derive a real usability estimate from the tier: usable = Q2/Q3, so P(usable) ≈ confidence when
        # the predicted grade is usable, else ≈ 1 − confidence.
        p_usable = float(confidence) if grade >= 2 else float(1.0 - confidence)
        model_grade = {"grade": grade, "p_usable": round(p_usable, 4), "grade_confidence": float(confidence)}
        guard = self._guard_reports.get(modality)
        carrier = AuditWorkerSignals()
        carrier._rgen = self._recording_gen  # type: ignore[attr-defined]  # drop if a new record opens
        carrier.auditReady.connect(self._on_audit_ready)
        carrier.failed.connect(self._on_audit_failed)
        self._audit_carriers.append(carrier)
        s = self._settings
        kw = ({"model": s.auditModel, "host": s.ollamaHost, "samples": s.auditSamples}
              if s is not None else {})
        self._audit_pool.start(AuditTask(runner, window, model_grade, guard, carrier, **kw))

    @Slot(float, float)
    def _on_saliency_requested(self, start_sec: float, end_sec: float) -> None:
        """Run occlusion SALIENCY (XAI 'what is the model looking at') for the selected window off the GUI
        thread and push the downsampled importance map to the guard."""
        if self._guard is None:
            return
        got = self._audit_window(start_sec, end_sec)
        if got is None:
            self._guard.setSaliency({"map": []})
            return
        _modality, runner, window = got
        carrier = SaliencyWorkerSignals()
        carrier._rgen = self._recording_gen  # type: ignore[attr-defined]  # drop if a new record opens
        carrier._sgen = self._selection_gen  # type: ignore[attr-defined]  # drop if the SELECTION changed
        carrier.saliencyReady.connect(self._on_saliency_ready)
        self._saliency_carriers.append(carrier)
        self._pool.start(SaliencyTask(runner, window, carrier))

    @Slot(object)
    def _on_saliency_ready(self, payload) -> None:
        sender = self.sender()
        self._saliency_carriers = [c for c in self._saliency_carriers if c is not sender]  # free the carrier
        # drop a late result whose recording OR segment selection has since changed — else it would paint
        # the old segment's heatmap over the newly-selected trace (misaligned attribution overlay).
        if self._stale("_rgen", self._recording_gen) or getattr(sender, "_sgen", -1) != self._selection_gen:
            return
        if self._guard is not None:
            self._guard.setSaliency(payload)

    def _on_selection_changed(self) -> None:
        """Authoritative selection change: bump the selection generation so any in-flight saliency for the
        previously-selected segment is dropped when it returns (see ``_on_saliency_ready``)."""
        self._selection_gen += 1

    @Slot(float, float)
    def _on_sqi_requested(self, start_sec: float, end_sec: float) -> None:
        """Compute the interpretable classical-SQI breakdown for the selected window (research3
        explainability) and push it to the guard. Cheap (pure numpy); off the decision path."""
        if self._guard is None:
            return
        got = self._audit_window(start_sec, end_sec)
        if got is None:
            self._guard.setSqiBreakdown([])
            self._guard.setUsability([])          # keep both panels in lock-step (don't leave stale bands)
            return
        modality, runner, window = got
        fs = float(runner.card.fs_hz)
        try:
            from biosqa.inference.sqi_breakdown import sqi_breakdown, sqi_consensus
            rows = sqi_breakdown(window, fs, modality)
            # the SAME bank on a band-pass-filtered copy (Raw/Filtered toggle): shows what a standard
            # filter would/would not fix — e.g. baseline wander clears, in-band EMG does not.
            filt_rows = []
            try:
                from biosqa.inference.recover import filter_for_modality
                filt_rows = sqi_breakdown(filter_for_modality(window, fs, modality), fs, modality)
            except Exception:  # noqa: BLE001 - filtered view is advisory
                filt_rows = []
            consensus = sqi_consensus(rows)          # consensus is on the RAW bank (what the model saw)
        except Exception:  # noqa: BLE001
            rows, filt_rows, consensus = [], [], 0.0
        self._guard.setSqiBreakdown(rows, filt_rows, consensus)
        # per-modality "usable for what" verdicts (EEG per-band, EDA tonic/phasic) for the same window
        try:
            from biosqa.inference.task_usability import usability_verdicts
            self._guard.setUsability(usability_verdicts(window, fs, modality))
        except Exception:  # noqa: BLE001
            self._guard.setUsability([])

    def _audit_window(self, start_sec: float, end_sec: float):
        """Return ``(modality, runner, window[np.float32])`` for the selected span, from the
        in-memory signal (normal record) or by re-reading it from the handle (streamed record)."""
        if self._current is not None:
            modality, sig = self._current
            runner = self._runners.get(modality)
            if runner is None or runner.card is None:
                return None
            fs = float(runner.card.fs_hz)
            lo = max(0, int(start_sec * fs))
            hi = min(sig.size, max(lo + 1, int(end_sec * fs)))
            return modality, runner, np.asarray(sig[lo:hi], dtype=np.float32)
        if self._current_stream is not None:
            modality, handle, infer_ch, fs_in, fs_out = self._current_stream
            runner = self._runners.get(modality)
            if runner is None or runner.card is None:
                return None
            n = int(handle.n_samples[infer_ch])
            s0 = max(0, int(start_sec * fs_in))
            s1 = min(n, max(s0 + 1, int(end_sec * fs_in)))
            try:
                raw = np.asarray(read_window(handle, [infer_ch], s0, s1)).reshape(-1)
            except Exception:  # noqa: BLE001
                return None
            return modality, runner, np.asarray(resample_signal(raw, fs_in, fs_out), dtype=np.float32)
        return None

    @Slot(object)
    def _on_audit_ready(self, judgment) -> None:
        sender = self.sender()
        self._audit_carriers = [c for c in self._audit_carriers if c is not sender]  # free the finished carrier
        if self._stale("_rgen", self._recording_gen):
            return                                # audit from a superseded recording — drop it
        if self._guard is not None:
            self._guard.setAuditResult(judgment)

    @Slot(str)
    def _on_audit_failed(self, message: str) -> None:
        sender = self.sender()
        self._audit_carriers = [c for c in self._audit_carriers if c is not sender]  # free the finished carrier
        if self._stale("_rgen", self._recording_gen):
            return
        if self._guard is not None:
            self._guard.setAuditResult({"error": message})

    @Slot(str, str)
    def _on_failed(self, modality: str, message: str) -> None:
        if self._stale("_igen", self._inference_gen):
            return
        self._inference.report(f"{modality.upper()} inference failed: {message}", "", 0.0)
