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
from threading import Event

import numpy as np
from PySide6.QtCore import QCoreApplication, QObject, QThreadPool, QTimer, Slot

from biosqa.inference.onnx_runner import OnnxRunner
from biosqa.io.loaders import RecordingHandle, read_window
from biosqa.viewmodels.channel_model import ChannelEntry
from biosqa.workers.qt_threads import (
    AuditTask,
    SaliencyTask,
    InferenceTask,
    LoadResampleTask,
    SqiTask,
    SqiWorkerSignals,
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
        self._sqi_carriers: list[SqiWorkerSignals] = []
        #: span (start_sec, end_sec) of the most recent SQI request — the fallback "what is on screen"
        #: for the SQI result guard when no segment is selected (see :meth:`_live_sqi_span`).
        self._sqi_span: tuple[float, float] | None = None
        self._selection_gen = 0   # bumped on each segment select; a saliency result from an old selection is dropped
        self._load_carriers: list[LoadResampleWorkerSignals] = []
        self._stream_carriers: list[StreamWorkerSignals] = []
        self._pending: dict[str, tuple[float, int]] = {}   # modality -> (t0, n_windows)
        self._current: tuple[str, "np.ndarray"] | None = None  # (modality, resampled signal) for audit
        #: for a STREAMED (large) recording there is no in-memory signal; audit re-reads the selected
        #: window from the handle. (modality, handle, infer_channel, fs_in, fs_out).
        self._current_stream: tuple | None = None
        #: identity of what is being graded: (recording path, analyzed channel, its index in the
        #: RECORD's channel order). Every artifact — plot, bands, reviews, export — is bound to this
        #: one channel of this one recording; None = nothing analyzed.
        self._analyzed: tuple[str, str, int] | None = None
        self._plot_channels: list[str] = []   # lane order, analyzed channel first
        self._dropped_reviews = 0             # reviews an in-progress re-run of the SAME record dropped
        #: the last segmentation status line ``(text, model_version, latency_ms)``. The streaming
        #: notice APPENDS to it instead of replacing it (see :meth:`_on_notice`).
        self._last_status: tuple[str, str, float] = ("", "", 0.0)
        self._guard_reports: dict[str, dict] = {}          # modality -> latest guard report
        # Generation guards: every task carrier is stamped with the generation it was dispatched
        # under; a result whose stamp is stale (a newer open / re-run superseded it) is DROPPED, so
        # opening another file or changing a setting mid-flight can't load old data into the new view.
        self._recording_gen = 0    # bumped on every open (guards plot binding + current-signal state)
        self._inference_gen = 0    # bumped on every inference dispatch (guards intervals/guard/notice)
        #: cooperative cancel token of the IN-FLIGHT inference/stream (see :meth:`_new_cancel_token`).
        self._infer_cancel: Event | None = None
        #: set once on quit. Every AuditTask carries it, so a queued audit on the single-threaded audit
        #: pool cannot start a fresh (up to samples x timeout) ollama round trip during teardown.
        self._quitting: Event = Event()
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
        # Overlap AND refinement affect BOTH paths → full re-run/re-stream. Only the guard, the
        # recoverability pass and the bSQI threshold are streaming-disabled, so they re-run normal
        # records alone (re-streaming a multi-hour record for a setting it cannot use is pure waste).
        self._rerun_timer = QTimer(self)
        self._rerun_timer.setSingleShot(True)
        self._rerun_timer.setInterval(200)
        self._rerun_timer.timeout.connect(self._do_scheduled_rerun)
        self._pending_full = False   # a pending change that a STREAMED record must re-stream for
        for sig_name in ("windowOverlapChanged", "refineBoundariesChanged"):
            sig = getattr(settings, sig_name, None)
            if sig is not None:
                sig.connect(self._schedule_rerun)
        for sig_name in ("recoveryEnabledChanged", "guardEnabledChanged", "bsqiThresholdChanged"):
            sig = getattr(settings, sig_name, None)
            if sig is not None:
                sig.connect(self._schedule_rerun_normal)
        recordings.recordingOpened.connect(self.on_recording_opened)
        # Quit was not a synchronisation point: nothing cancelled in-flight work, and QThreadPool's
        # DESTRUCTOR then waits for it — so the window disappeared while the process lived on for the
        # rest of a streamed whole-record read or an ollama timeout (measured 32.6 s of zombie for a
        # 20 s task). Cancel first, then drain with a bound, while the app is still alive.
        qapp = QCoreApplication.instance()
        if qapp is not None:
            qapp.aboutToQuit.connect(self.shutdown)

    #: how long a quit waits for the cancelled workers to return before giving up on them (ms).
    SHUTDOWN_WAIT_MS = 3000

    @Slot()
    def shutdown(self) -> None:
        """Cancel every in-flight worker and drain the pools (bounded). Wired to ``aboutToQuit``, and
        idempotent — the cooperative tasks return at their next phase / streamed-block boundary."""
        self._quitting.set()
        self._cancel_inflight()
        self._rerun_timer.stop()
        self._pool.waitForDone(self.SHUTDOWN_WAIT_MS)
        self._audit_pool.waitForDone(self.SHUTDOWN_WAIT_MS)

    @staticmethod
    def _prune(carriers: list, attr: str, current: int) -> list:
        """Drop carriers whose generation stamp is stale (an in-flight QRunnable keeps its own carrier
        alive until it finishes, so this only frees already-superseded ones)."""
        return [c for c in carriers if getattr(c, attr, current) == current]

    @Slot()
    def _schedule_rerun(self) -> None:
        self._pending_full = True
        self._rerun_timer.start()

    @Slot()
    def _schedule_rerun_normal(self) -> None:
        self._rerun_timer.start()

    def _do_scheduled_rerun(self) -> None:
        if self._pending_full:
            self._pending_full = False
            self._rerun_inference()          # overlap/refinement changed → full re-run/re-stream
        else:
            self._rerun_inference_normal()

    def _refine_enabled(self) -> bool:
        """The "Refine boundaries" setting. Read by BOTH dispatch paths: refinement is not a
        normal-record-only feature, so a streamed record must honour the toggle too."""
        return bool(getattr(self._settings, "refineBoundaries", True)) if self._settings else True

    def _runner(self, modality: str) -> OnnxRunner:
        runner = self._runners.get(modality)
        if runner is None:
            runner = OnnxRunner(modality, self._models_dir)
            runner.load()
            self._runners[modality] = runner
        return runner

    def _stale(self, gen_attr: str, current: int) -> bool:
        """True if the emitting carrier's generation stamp (``gen_attr``) is not ``current`` — i.e. a
        newer open / re-run has superseded it, so this result must be ignored.

        FAILS CLOSED. A superseded carrier is dropped from its list (``_prune``) and then collected,
        so by the time its already-queued signal is delivered ``sender()`` can be None — under the old
        ``getattr(self.sender(), attr, current)`` that defaulted to ``current`` and the check
        degenerated to ``current != current`` -> NOT stale, i.e. exactly the superseded results the
        guard exists to reject were the ones it let through. An unidentifiable sender is stale."""
        s = self.sender()
        return s is None or getattr(s, gen_attr, None) != current

    def _stale_payload(self, payload, gen_attr: str, current: int) -> bool:
        """Like :meth:`_stale` but prefers a generation stamp CARRIED IN THE PAYLOAD, which cannot go
        missing with the carrier. Falls back to the sender for payloads that carry none."""
        gen = payload.get(gen_attr) if isinstance(payload, dict) else None
        return (gen != current) if gen is not None else self._stale(gen_attr, current)

    def _new_cancel_token(self) -> Event:
        """CANCEL whatever inference/stream is in flight and return a FRESH token for the run about
        to be dispatched.

        Cooperative: the superseded task returns at its next phase / streamed-block boundary and
        emits NOTHING AT ALL — not even ``failed`` — so it can neither burn pool threads on work
        nobody will look at (``_recoverability`` is a whole SECOND ONNX pass; a streamed job reads
        the ENTIRE record) nor clobber the live run's status line."""
        self._cancel_inflight()
        self._infer_cancel = Event()
        return self._infer_cancel

    def _cancel_inflight(self) -> None:
        """Set the in-flight run's token (if any) and forget it. Used on its own by ``_invalidate``:
        opening another recording must stop the previous one's analysis, not merely discard its
        result once it finally lands."""
        if self._infer_cancel is not None:
            self._infer_cancel.set()
            self._infer_cancel = None

    def _invalidate(self, path: str) -> None:
        """Make an open an ATOMIC state transition: drop EVERY artifact of the previously-analyzed
        recording before anything about the new one can fail.

        Runs first in ``on_recording_opened``, so a miss/failure downstream (a modality model that
        won't load) can no longer leave the previous recording's waveform, channel list, segments,
        quality bands, minimap, overview, saliency, model card, guard report or human reviews on
        screen — and exportable — under the NEW recording's name, fs and provenance."""
        self._recording_gen += 1
        self._inference_gen += 1
        self._cancel_inflight()          # stop the old analysis; don't just ignore its result
        self._current = None
        self._current_stream = None
        self._plot_channels = []
        self._last_status = ("", "", 0.0)
        self._guard_reports.clear()
        self._pending.clear()
        self._segments.load_intervals([])          # the segment model had no other reset path at all
        # The PLOT is per-recording state too: without this the previous recording's waveform stayed
        # drawn (and its handle bound, so valueAt/curveForRange kept answering with ITS samples)
        # while the title, channel list, modality and fs already named the new one.
        if self._signal_view is not None and hasattr(self._signal_view, "clear"):
            self._signal_view.clear()
        self._channels.set_channels([])            # ...and so is the channel list
        self._blank_model_card()                   # ...and the model provenance panel
        prev_path = self._analyzed[0] if self._analyzed else ""
        self._analyzed = None
        dropped = 0
        if self._selection is not None:
            if hasattr(self._selection, "set_context"):
                dropped = int(self._selection.set_context(recording=path))
            if hasattr(self._selection, "clear"):
                self._selection.clear()
        # Re-opening the SAME recording re-segments it, so its reviews are dropped too — say so on
        # the next status line rather than losing them silently (a different recording just ends the
        # previous recording's review session; that needs no notice).
        self._dropped_reviews = dropped if (prev_path and prev_path == path) else 0
        if self._guard is not None:
            self._guard.reset()   # guard/data-quality banner, saliency, attribution, SQI, audit
        rgen, igen = self._recording_gen, self._inference_gen
        self._load_carriers = self._prune(self._load_carriers, "_rgen", rgen)
        self._stream_carriers = self._prune(self._stream_carriers, "_rgen", rgen)
        self._audit_carriers = self._prune(self._audit_carriers, "_rgen", rgen)
        self._saliency_carriers = self._prune(self._saliency_carriers, "_rgen", rgen)
        self._sqi_carriers = self._prune(self._sqi_carriers, "_rgen", rgen)
        self._carriers = self._prune(self._carriers, "_igen", igen)

    def _blank_model_card(self) -> None:
        """Empty the model-card panel (a failed open must not show the PREVIOUS model's provenance
        next to the new recording's name). ModelCardModel exposes no clear(), so reset it in place."""
        mc = self._model_card
        if mc is None:
            return
        try:
            mc.beginResetModel()
            mc._rows = []
            mc._card = None
            mc.endResetModel()
            mc.cardChanged.emit()
        except Exception:  # noqa: BLE001 - a panel that won't blank must not block the open
            pass

    def _set_channel_list(self, handle, modality: str, analyzed: str = "") -> list[str]:
        """Populate the channel list with the ANALYZED channel FIRST and badged.

        The analyzed channel is the plot's primary lane, the segment inspector's trace and the
        export's annotation channel — previously the plot/export used ``channel_names[0]`` while
        inference ran on the modality-matching channel, so a record like ["RESP", "II"] (or any
        12-lead ECG, whose first channel is "I" but whose preferred channel is "II") was graded on
        one signal and had the bands drawn over another. ``analyzed=""`` means nothing was graded
        (a failed open): the channels are listed in file order, none is badged."""
        names = list(handle.channel_names)
        if analyzed and analyzed in names:
            names = [analyzed] + [n for n in names if n != analyzed]
        units = getattr(handle, "units", {})
        # Only the first (= analyzed) channel is visible by default, so the plot opens as a single
        # lane (unchanged look); toggling other channels' eye adds stacked lanes.
        self._channels.set_channels([
            ChannelEntry(name=n, modality=modality, unit=units.get(n, ""),
                         visible=(i == 0), analyzed=bool(analyzed) and n == analyzed)
            for i, n in enumerate(names)
        ])
        return names

    @Slot(str)
    def on_recording_opened(self, path: str) -> None:
        info = self._recordings.handle_for(path)
        self._invalidate(path)                     # FIRST: no early return may skip this
        if info is None:
            self._inference.report("Recording could not be opened.", "", 0.0)
            return
        handle, modality = info
        try:
            runner = self._runner(modality)
        except Exception as exc:  # noqa: BLE001
            # A coherent EMPTY state under the new recording's name: its channels exist but NOTHING
            # was graded (no analyzed badge, no plot, no segments, no card) — never the previous
            # recording's analysis wearing this recording's identity.
            self._set_channel_list(handle, modality)
            self._inference.setPrecision("")          # nothing is loaded: report unknown, not stale
            self._inference.report(f"Model load failed ({modality}): {exc}", "", 0.0)
            return
        card = runner.card
        # Read off the graph at load time, so the status bar states the precision it is actually
        # running rather than the hardcoded "FP32" it used to claim.
        self._inference.setPrecision(getattr(runner, "precision", ""))

        # model-card panel + channel list
        try:
            self._model_card.load(str(self._models_dir / f"{modality}.model_card.json"))
        except Exception:  # noqa: BLE001 - a missing card panel shouldn't block inference
            self._blank_model_card()   # ...but it must never keep showing the PREVIOUS card either

        # Heavy work (full-channel read + resample + primary-channel plot cache) goes OFF-THREAD so
        # on_recording_opened returns to the event loop immediately (no freeze). Very long records
        # take the OUT-OF-CORE path (StreamInferenceTask): block-wise plot cache + streaming
        # inference, never holding the whole signal — so multi-day / big multichannel files don't OOM.
        from biosqa.inference.streaming import LARGE_RECORD_SAMPLES, estimate_analysis_samples

        infer_ch = _preferred_channel(handle, modality)
        self._plot_channels = self._set_channel_list(handle, modality, infer_ch)
        # index in the RECORD's own channel order (NOT the re-ordered lane list) — WFDB annotations
        # are written against the record's signal index.
        self._analyzed = (path, infer_ch, list(handle.channel_names).index(infer_ch))
        rgen, igen = self._recording_gen, self._inference_gen
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
            # A streamed job reads the WHOLE record — a superseded one must stop, not finish.
            self._pool.start(StreamInferenceTask(handle, infer_ch, infer_ch,
                                                 list(self._plot_channels), modality, runner,
                                                 overlap, carrier,
                                                 refine_enabled=self._refine_enabled(),
                                                 cancel=self._new_cancel_token(), gen=rgen))
            return
        self._inference.report(f"Loading {modality.upper()}…", card.model_version, 0.0)
        carrier = LoadResampleWorkerSignals()
        carrier._rgen, carrier._igen = rgen, igen  # type: ignore[attr-defined]
        carrier.ready.connect(self._on_loaded)
        carrier.failed.connect(self._on_load_failed)
        self._load_carriers.append(carrier)
        # A superseded load must STOP (it reads + resamples the whole channel), and the generation
        # travels in the payload so the staleness check never depends on this carrier still existing.
        self._pool.start(LoadResampleTask(handle, infer_ch, infer_ch, list(self._plot_channels),
                                          float(card.fs_hz), modality, carrier,
                                          cancel=self._new_cancel_token(), gen=rgen))

    @Slot(object)
    def _on_loaded(self, payload) -> None:
        """LoadResampleTask finished off-thread: bind the plot from the precomputed cache, keep the
        resampled signal for audit/re-run, then dispatch inference — all back on the GUI thread."""
        if self._stale_payload(payload, "_rgen", self._recording_gen):
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
        if self._stale_payload(payload, "_rgen", self._recording_gen):
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
        # APPEND to the segmentation status rather than replace it: the notice lands right after
        # `_on_intervals` on the streaming path, and reporting it alone erased the segment count, the
        # model version, the latency AND the "N reviews dropped by re-segmentation" warning — so the
        # recordings whose analysis takes longest were the only ones that never reported it.
        status, version, latency = self._last_status
        self._inference.report(f"{status} · {message}" if status else message, version, latency)

    def _report_if_too_short(self, modality: str, sig, card, overlap: float) -> bool:
        """True (and the UI says so) when the signal is shorter than ONE model window.

        Such a record yields zero windows, hence zero segments — which the status line rendered as
        "EDA · 0 segments" and a user reads as "no quality problems were found". It is the opposite:
        NOTHING was analyzed. ``preprocess.window_starts`` raises ShortRecordError with the exact
        numbers; we neither depend on that (the length check is our own) nor on it not raising."""
        n = int(getattr(sig, "size", 0))
        l_m = int(getattr(card, "l_m", 0) or 0)
        if n >= max(1, l_m):
            return False
        detail = ""
        try:
            from biosqa.inference.preprocess import window_starts
            window_starts(n, card, overlap)
        except Exception as exc:  # noqa: BLE001 - the message is what we want, not the type
            detail = str(exc)
        if not detail:
            fs = float(getattr(card, "fs_hz", 0.0)) or 1.0
            detail = (f"record is {n / fs:.1f} s but one {modality} window is "
                      f"{l_m / fs:.1f} s — nothing was analyzed")
        self._segments.load_intervals([])
        self._inference.report(f"{modality.upper()} NOT ANALYSED — {detail}",
                               getattr(card, "model_version", ""), 0.0)
        return True

    def _bind_selection_context(self, modality: str) -> int:
        """Stamp the human-review/export state with the identity of the analysis that just produced
        these intervals (recording, graded channel, model, segmentation revision). Returns how many
        reviews were dropped because they belonged to a superseded analysis."""
        if self._selection is None or not hasattr(self._selection, "set_context"):
            return 0
        path, channel, ch_index = self._analyzed or ("", "", -1)
        runner = self._runners.get(modality)
        version = runner.card.model_version if runner is not None and runner.card else ""
        return int(self._selection.set_context(
            recording=path, channel=channel, channel_index=ch_index,
            model_version=version, revision=self._inference_gen))

    def _start_inference(self, modality: str, sig) -> None:
        """Dispatch sliding-window inference over an already-loaded signal using the CURRENT settings.
        Called on load-complete and again on any analysis-setting change (live re-segment, no re-read)."""
        runner = self._runners.get(modality)
        if runner is None or runner.card is None or sig is None:
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
        refine_enabled = self._refine_enabled()
        if self._report_if_too_short(modality, sig, card, overlap):
            return
        n_windows = max(0, (sig.size - card.l_m) // stride_samples + 1)
        self._inference_gen += 1
        self._carriers = self._prune(self._carriers, "_igen", self._inference_gen)
        # CANCEL the superseded run instead of only dropping its result: the generation guard already
        # made a stale result harmless, but the task itself ran to completion — including a SECOND full
        # ONNX pass in _recoverability — burning pool threads a dragged slider re-queues N times over.
        cancel = self._new_cancel_token()
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
                                       refine_enabled=refine_enabled,
                                       cancel=cancel))

    @Slot()
    def _rerun_inference_normal(self) -> None:
        """Re-segment a NORMAL (in-memory) recording after a guard/recovery/bSQI change. A streamed
        record disables those THREE features, so this is a no-op there (no wasted re-stream).
        Refinement is NOT one of them — it runs streamed too, so it goes via :meth:`_rerun_inference`."""
        if self._current is not None:
            modality, sig = self._current
            self._start_inference(modality, sig)

    @Slot()
    def _rerun_inference(self) -> None:
        """Re-segment after an OVERLAP or REFINEMENT change (both affect both paths). Normal records
        reuse the in-memory signal; a streamed record re-streams (keeping the current view — no plot
        rebuild). "Refine boundaries" used to be routed to ``_rerun_inference_normal``, which returns
        immediately when there is no in-memory signal — so the toggle silently did nothing to a
        streamed record (whose boundaries were never refined in the first place)."""
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
            self._pool.start(StreamInferenceTask(handle, infer_ch, infer_ch,
                                                 list(self._plot_channels or handle.channel_names),
                                                 modality, runner, overlap, carrier,
                                                 rebuild_plot=False,
                                                 refine_enabled=self._refine_enabled(),
                                                 cancel=self._new_cancel_token()))

    @Slot(str, str)
    def _on_load_failed(self, modality: str, message: str) -> None:
        if self._stale("_rgen", self._recording_gen):
            return
        self._inference.report(f"{modality.upper()} load failed: {message}", "", 0.0)

    @Slot(str, object)
    def _on_intervals(self, modality: str, intervals) -> None:
        if self._stale("_igen", self._inference_gen):
            return                                # a newer inference/open superseded this result
        # Bind the reviews/exports to THIS analysis before the intervals become visible: a re-run
        # re-segments the recording, so reviews anchored to the previous segmentation are dropped
        # (they no longer name a real interval) rather than re-anchored to a different span.
        dropped = self._bind_selection_context(modality) + self._dropped_reviews
        self._dropped_reviews = 0
        self._segments.load_intervals(intervals)
        t0, n_windows = self._pending.get(modality, (None, 0))
        if not n_windows:
            # STREAMED path: the window count cannot be known at dispatch (the record is scored block
            # by block), so both streaming dispatch sites record 0 and the task stamps the REAL count
            # on its carrier when it finishes. Reading it here is what keeps `latencyMs` from being a
            # flat 0.0 on exactly the recordings whose analysis takes longest.
            n_windows = int(getattr(self.sender(), "_n_windows", 0) or 0)
        latency = 0.0
        if t0 is not None and n_windows > 0:
            latency = (time.perf_counter() - t0) * 1000.0 / n_windows
        runner = self._runners.get(modality)
        version = runner.card.model_version if runner and runner.card else ""
        n = len(list(intervals)) if intervals is not None else 0
        status = f"{modality.upper()} · {n} segments"
        if dropped:
            status += (f" · {dropped} review{'s' if dropped != 1 else ''} dropped by re-segmentation")
        self._last_status = (status, version, latency)   # a streaming notice appends to this
        self._inference.report(status, version, latency)

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
        self._audit_pool.start(AuditTask(runner, window, model_grade, guard, carrier,
                                         cancel=self._quitting, **kw))

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
        # Fails closed with _stale: no identifiable sender means no identifiable selection either.
        if (sender is None or self._stale("_rgen", self._recording_gen)
                or getattr(sender, "_sgen", None) != self._selection_gen):
            return
        if self._guard is not None:
            self._guard.setSaliency(payload)

    def _on_selection_changed(self) -> None:
        """Authoritative selection change: bump the selection generation so any in-flight saliency for the
        previously-selected segment is dropped when it returns (see ``_on_saliency_ready``)."""
        self._selection_gen += 1

    @Slot(float, float)
    def _on_sqi_requested(self, start_sec: float, end_sec: float) -> None:
        """Dispatch the interpretable classical-SQI breakdown for the selected window (research3
        explainability) OFF the GUI thread and push the result to the guard when it lands.

        It used to run inline on a DIRECT connection between two GUI-thread objects — and it is not
        cheap: the whole bank runs twice (raw + band-pass-filtered) over the selected span, which on a
        long single-segment record is the whole record. Measured ~10-20 s of frozen window per
        selection. Same shape as :meth:`_on_saliency_requested` now."""
        if self._guard is None:
            return
        got = self._audit_window(start_sec, end_sec)
        if got is None:
            self._guard.setSqiBreakdown([])
            self._guard.setUsability([])          # keep both panels in lock-step (don't leave stale bands)
            return
        modality, runner, window = got
        carrier = SqiWorkerSignals()
        carrier._rgen = self._recording_gen  # type: ignore[attr-defined]  # drop if a new record opens
        # Stamp the SPAN this bank is being computed for, not a selection COUNTER. QML requests the SQI
        # from the same `selectedSegmentChanged` signal that bumps `_selection_gen`, and QML's handler
        # runs FIRST, so a counter stamped here is always one behind the live one and every
        # selection-driven result was rejected as stale (both panels permanently empty). The span is
        # order-independent: a result for the span that is on screen is fresh however the signals raced.
        carrier._span = (float(start_sec), float(end_sec))  # type: ignore[attr-defined]
        self._sqi_span = carrier._span   # most recent request (the live span when nothing is selected)
        carrier.sqiReady.connect(self._on_sqi_ready)
        self._sqi_carriers.append(carrier)
        self._pool.start(SqiTask(window, float(runner.card.fs_hz), modality, carrier))

    #: tolerance (seconds) when matching an SQI result's span against the span on screen.
    _SPAN_EPS = 1e-6

    def _live_sqi_span(self) -> tuple[float, float] | None:
        """The span the SQI panels are meant to be describing: the SELECTED segment's bounds, or —
        when nothing is selected (a direct/programmatic ``requestSqi``) — the most recent request."""
        selection = self._selection
        sel = getattr(selection, "selectedSegment", None) if selection is not None else None
        if sel is not None:
            try:
                return float(sel.startSec), float(sel.endSec)
            except Exception:  # noqa: BLE001 - a stand-in without the properties: fall through
                return None
        return self._sqi_span

    def _stale_sqi_span(self, span) -> bool:
        """True if ``span`` is not the span on screen — fails CLOSED (an unidentifiable span, or no
        span to compare it against, cannot be shown to belong to what the user is looking at)."""
        live = self._live_sqi_span()
        if span is None or live is None:
            return True
        return (abs(span[0] - live[0]) > self._SPAN_EPS
                or abs(span[1] - live[1]) > self._SPAN_EPS)

    @Slot(object)
    def _on_sqi_ready(self, payload) -> None:
        sender = self.sender()
        self._sqi_carriers = [c for c in self._sqi_carriers if c is not sender]   # free the carrier
        # Fail-closed recording + SPAN guard: a breakdown computed for a span that is no longer the
        # selected one describes a different stretch of a possibly different record.
        if (sender is None or self._stale("_rgen", self._recording_gen)
                or self._stale_sqi_span(getattr(sender, "_span", None))):
            return
        if self._guard is None:
            return
        self._guard.setSqiBreakdown(payload.get("rows", []), payload.get("filtered", []),
                                    float(payload.get("consensus", 0.0)))
        self._guard.setUsability(payload.get("usability", []))

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
