"""Top-bar model status pill: version/quant/latency (design spec (b)/(d))."""

from __future__ import annotations

from PySide6.QtCore import Property, QObject, Signal, Slot


class InferenceStatusController(QObject):
    """Live-updated from `inference.onnx_runner` once a session is loaded/running.

    Every field here is a FACT ABOUT THE MODEL, so each has an explicit "not measured"
    value that the pill renders as an omitted clause rather than a plausible number:
    ``latencyMs == 0.0`` (never measured) and ``precision == ""`` (unknown). The top bar
    used to paper over both with a hardcoded "FP32 · 2.1 ms/win".
    """

    statusTextChanged = Signal()
    latencyMsChanged = Signal()
    modelVersionChanged = Signal()
    precisionChanged = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._status_text = "No model loaded"
        self._latency_ms = 0.0
        self._model_version = ""
        self._precision = ""

    def _get_status_text(self) -> str:
        return self._status_text

    statusText = Property(str, _get_status_text, notify=statusTextChanged)

    def _get_latency_ms(self) -> float:
        return self._latency_ms

    #: measured ms/window; 0.0 means NOTHING has been measured yet (the pill omits the clause).
    latencyMs = Property(float, _get_latency_ms, notify=latencyMsChanged)

    def _get_model_version(self) -> str:
        return self._model_version

    modelVersion = Property(str, _get_model_version, notify=modelVersionChanged)

    def _get_precision(self) -> str:
        return self._precision

    #: numeric precision of the loaded ONNX graph ("FP32"/"INT8"); "" until a producer
    #: reports it, in which case the pill says nothing rather than guessing.
    precision = Property(str, _get_precision, notify=precisionChanged)

    @Slot(str, str, float)
    @Slot(str, str, float, str)
    def report(self, status_text: str, model_version: str, latency_ms: float,
               precision: str | None = None) -> None:
        """Update the pill; called from `OnnxRunner`/inference-worker result handlers.

        `precision` is optional and STICKY: a 3-arg caller (which doesn't know it) leaves the
        last reported value alone rather than clearing it. Never substitute a default -- an
        unmeasured latency or an assumed quantization is an invented claim about the model.
        """
        self._status_text = status_text
        self._model_version = model_version
        self._latency_ms = latency_ms
        self.statusTextChanged.emit()
        self.modelVersionChanged.emit()
        self.latencyMsChanged.emit()
        if precision is not None:
            self.setPrecision(precision)

    @Slot(str)
    def setPrecision(self, precision: str) -> None:  # noqa: N802
        """Report the loaded graph's precision on its own (it is known at model-load time,
        i.e. before the first `report()` carries a measured latency)."""
        if precision != self._precision:
            self._precision = precision
            self.precisionChanged.emit()
