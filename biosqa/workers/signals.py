"""Qt signal definitions for worker -> UI results (Plan 2 §9).

``QRunnable`` has no signal support of its own (it isn't a ``QObject``), so
each worker owns one of these small ``QObject`` signal-carrier instances and
emits through it. Connections from these signals to ``viewmodels`` slots use
Qt's automatic cross-thread queuing (``Qt.AutoConnection`` resolves to
``Qt.QueuedConnection`` when emitter and receiver live on different
threads), which is what keeps the receiving slot's code running safely back
on the GUI thread.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal


class InferenceWorkerSignals(QObject):
    """Emitted by a sliding-window ONNX inference ``QRunnable`` (Plan 2 §7.2)."""

    progress = Signal(str, float)  # (modality, fraction_complete)
    intervalsReady = Signal(str, object)  # (modality, list[QualityInterval])
    guardReady = Signal(str, object)  # (modality, false-clean guard report dict)
    dataQualityReady = Signal(str, object)  # (modality, record data-quality report dict)
    failed = Signal(str, str)


class LoadResampleWorkerSignals(QObject):
    """Emitted by a :class:`~biosqa.workers.qt_threads.LoadResampleTask` — the off-thread
    full-channel read + resample + plot-cache build that used to block the GUI thread on open."""

    #: one dict payload: modality, handle, sig (resampled inference signal), fs_in, fs_out,
    #: plot_channels, plot_fs, full_t, full_y, trace_lo, trace_hi, n_samples_primary.
    ready = Signal(object)
    failed = Signal(str, str)  # (modality, error_message)


class ChannelCacheWorkerSignals(QObject):
    """Off-thread build of a non-primary channel's plot cache — so toggling an extra lane never
    blocks the GUI on a whole-channel read (P1), including on large streamed records."""

    ready = Signal(str, object, object, float, float)   # (channel, full_t, full_y, lo, hi)
    failed = Signal(str, str)


class StreamWorkerSignals(QObject):
    """Emitted by a :class:`~biosqa.workers.qt_threads.StreamInferenceTask` — the out-of-core path
    for very long recordings (block-wise plot cache + streaming inference, no full signal in RAM)."""

    plotReady = Signal(object)            # plot-cache payload dict (like LoadResampleWorkerSignals)
    intervalsReady = Signal(str, object)  # (modality, list[QualityInterval])
    notice = Signal(str)                  # a user-facing note (e.g. "streaming mode: guard off")
    failed = Signal(str, str)


class AuditWorkerSignals(QObject):
    """Emitted by an on-demand LLM-audit ``QRunnable`` (off the decision path)."""

    auditReady = Signal(object)  # (judgment dict, or {"error": ...} if the LLM was unreachable)
    failed = Signal(str)


class SaliencyWorkerSignals(QObject):
    """Emitted by an on-demand occlusion-saliency ``QRunnable`` (XAI 'what is the model looking at')."""

    saliencyReady = Signal(object)  # {"map": [float,...] downsampled 0..1, "n": int}


class TranscodeWorkerSignals(QObject):
    """Emitted by a foreign-format -> Zarr transcode + pyramid-build ``QRunnable`` (Plan 2 §6.1)."""

    progress = Signal(str, float)  # (recording_path, fraction_complete)
    finished = Signal(str)  # (recording_path)
    failed = Signal(str, str)
