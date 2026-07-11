"""Top-bar model status pill: version/quant/latency (design spec (b)/(d))."""

from __future__ import annotations

from PySide6.QtCore import Property, QObject, Signal, Slot


class InferenceStatusController(QObject):
    """Live-updated from `inference.onnx_runner` once a session is loaded/running."""

    statusTextChanged = Signal()
    latencyMsChanged = Signal()
    modelVersionChanged = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._status_text = "No model loaded"
        self._latency_ms = 0.0
        self._model_version = ""

    def _get_status_text(self) -> str:
        return self._status_text

    statusText = Property(str, _get_status_text, notify=statusTextChanged)

    def _get_latency_ms(self) -> float:
        return self._latency_ms

    latencyMs = Property(float, _get_latency_ms, notify=latencyMsChanged)

    def _get_model_version(self) -> str:
        return self._model_version

    modelVersion = Property(str, _get_model_version, notify=modelVersionChanged)

    @Slot(str, str, float)
    def report(self, status_text: str, model_version: str, latency_ms: float) -> None:
        """Update the pill; called from `OnnxRunner`/inference-worker result handlers.

        TODO(Plan2 §7.1): call this after every `InferenceTask` completes
        with the real measured latency, once workers are wired up.
        """
        self._status_text = status_text
        self._model_version = model_version
        self._latency_ms = latency_ms
        self.statusTextChanged.emit()
        self.modelVersionChanged.emit()
        self.latencyMsChanged.emit()
