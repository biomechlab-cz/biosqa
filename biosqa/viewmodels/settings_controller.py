"""Persisted application settings (appearance + analysis + guard + LLM audit).

Single QML-facing source of truth (`settings` context property), backed by ``QSettings`` so choices
survive restarts. QML reads bindings (``settings.themeDark``) and writes through the ``setXxx`` slots
(``settings.setThemeDark(true)``); every setter coerces + range-clamps, persists, and emits its notify
signal. ``QSettings`` round-trips everything as strings on some platforms, so reads are explicitly
coerced back to bool/int/float.

Consumers: ``Theme`` (appearance, synced from ``Main.qml``), the ``Coordinator`` (window overlap, guard
enable + bSQI threshold, LLM audit enable/host/model/samples).
"""

from __future__ import annotations

from PySide6.QtCore import Property, QObject, QSettings, Signal, Slot

_ALLOWED_OVERLAP = (0.0, 0.25, 0.5, 0.75, 0.9)

_DEFAULTS: dict[str, object] = {
    "appearance/themeDark": True,
    "appearance/accent": "#35D0BA",
    "appearance/colorBlindTiers": False,
    "view/segmentation": "table",   # Quality Segmentation Table/Grid choice, remembered across runs
    "analysis/windowOverlap": 0.5,  # default 50% overlap → finer segment boundaries
    "analysis/recoveryEnabled": True,  # second filtered pass → "recoverable" advisory overlay
    "analysis/refineBoundaries": True,  # localize poor segments to the actual artefact (fine SQI)
    "guard/enabled": True,
    "guard/bsqiThreshold": 0.72,
    "audit/enabled": True,
    "audit/ollamaHost": "http://localhost:11434",
    "audit/model": "gemma4:latest",  # fast default; qwen3:32b is more accurate but much slower
    "audit/samples": 1,              # single pass by default (self-consistency 3-5 = 3-5x slower)
}


def _as_bool(v: object) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes", "on")
    return bool(v)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class SettingsController(QObject):
    """Exposed to QML as ``settings``. ``backend`` is injectable for isolated tests."""

    themeDarkChanged = Signal()
    accentChanged = Signal()
    colorBlindTiersChanged = Signal()
    segmentationViewChanged = Signal()
    windowOverlapChanged = Signal()
    recoveryEnabledChanged = Signal()
    refineBoundariesChanged = Signal()
    guardEnabledChanged = Signal()
    bsqiThresholdChanged = Signal()
    auditEnabledChanged = Signal()
    ollamaHostChanged = Signal()
    auditModelChanged = Signal()
    auditSamplesChanged = Signal()

    def __init__(self, backend: QSettings | None = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._s = backend if backend is not None else QSettings("BioSQA", "BioSQA Studio")
        self._theme_dark = _as_bool(self._raw("appearance/themeDark"))
        self._accent = str(self._raw("appearance/accent"))
        self._color_blind = _as_bool(self._raw("appearance/colorBlindTiers"))
        self._seg_view = str(self._raw("view/segmentation")) or "table"
        self._overlap = self._snap_overlap(self._raw_float("analysis/windowOverlap"))
        self._recovery_enabled = _as_bool(self._raw("analysis/recoveryEnabled"))
        self._refine_boundaries = _as_bool(self._raw("analysis/refineBoundaries"))
        self._guard_enabled = _as_bool(self._raw("guard/enabled"))
        self._bsqi = _clamp(self._raw_float("guard/bsqiThreshold"), 0.5, 0.9)
        self._audit_enabled = _as_bool(self._raw("audit/enabled"))
        self._ollama_host = str(self._raw("audit/ollamaHost"))
        self._audit_model = str(self._raw("audit/model"))
        self._audit_samples = int(_clamp(self._raw_float("audit/samples"), 1, 5))

    # ---- backend read helpers ---------------------------------------------
    def _raw(self, key: str) -> object:
        return self._s.value(key, _DEFAULTS[key])

    def _raw_float(self, key: str) -> float:
        try:
            return float(self._s.value(key, _DEFAULTS[key]))
        except (TypeError, ValueError):
            return float(_DEFAULTS[key])  # type: ignore[arg-type]

    def _store(self, key: str, value: object) -> None:
        self._s.setValue(key, value)
        self._s.sync()

    @staticmethod
    def _snap_overlap(v: float) -> float:
        return min(_ALLOWED_OVERLAP, key=lambda a: abs(a - v))

    # ---- appearance -------------------------------------------------------
    def _g_theme_dark(self) -> bool:
        return self._theme_dark

    themeDark = Property(bool, _g_theme_dark, notify=themeDarkChanged)

    @Slot(bool)
    def setThemeDark(self, v: bool) -> None:  # noqa: N802
        v = _as_bool(v)
        if v != self._theme_dark:
            self._theme_dark = v
            self._store("appearance/themeDark", v)
            self.themeDarkChanged.emit()

    def _g_accent(self) -> str:
        return self._accent

    accent = Property(str, _g_accent, notify=accentChanged)

    @Slot(str)
    def setAccent(self, v: str) -> None:  # noqa: N802
        v = str(v)
        if v != self._accent:
            self._accent = v
            self._store("appearance/accent", v)
            self.accentChanged.emit()

    def _g_color_blind(self) -> bool:
        return self._color_blind

    colorBlindTiers = Property(bool, _g_color_blind, notify=colorBlindTiersChanged)

    @Slot(bool)
    def setColorBlindTiers(self, v: bool) -> None:  # noqa: N802
        v = _as_bool(v)
        if v != self._color_blind:
            self._color_blind = v
            self._store("appearance/colorBlindTiers", v)
            self.colorBlindTiersChanged.emit()

    def _g_seg_view(self) -> str:
        return self._seg_view

    #: Quality-Segmentation Table/Grid choice, persisted so the last-used view is restored.
    segmentationView = Property(str, _g_seg_view, notify=segmentationViewChanged)

    @Slot(str)
    def setSegmentationView(self, v: str) -> None:  # noqa: N802
        v = "grid" if str(v).lower() == "grid" else "table"
        if v != self._seg_view:
            self._seg_view = v
            self._store("view/segmentation", v)
            self.segmentationViewChanged.emit()

    # ---- analysis ---------------------------------------------------------
    def _g_overlap(self) -> float:
        return self._overlap

    windowOverlap = Property(float, _g_overlap, notify=windowOverlapChanged)

    @Slot(float)
    def setWindowOverlap(self, v: float) -> None:  # noqa: N802
        v = self._snap_overlap(float(v))
        if v != self._overlap:
            self._overlap = v
            self._store("analysis/windowOverlap", v)
            self.windowOverlapChanged.emit()

    def _g_recovery_enabled(self) -> bool:
        return self._recovery_enabled

    recoveryEnabled = Property(bool, _g_recovery_enabled, notify=recoveryEnabledChanged)

    @Slot(bool)
    def setRecoveryEnabled(self, v: bool) -> None:  # noqa: N802
        v = _as_bool(v)
        if v != self._recovery_enabled:
            self._recovery_enabled = v
            self._store("analysis/recoveryEnabled", v)
            self.recoveryEnabledChanged.emit()

    def _g_refine_boundaries(self) -> bool:
        return self._refine_boundaries

    refineBoundaries = Property(bool, _g_refine_boundaries, notify=refineBoundariesChanged)

    @Slot(bool)
    def setRefineBoundaries(self, v: bool) -> None:  # noqa: N802
        v = _as_bool(v)
        if v != self._refine_boundaries:
            self._refine_boundaries = v
            self._store("analysis/refineBoundaries", v)
            self.refineBoundariesChanged.emit()

    # ---- integrity guard --------------------------------------------------
    def _g_guard_enabled(self) -> bool:
        return self._guard_enabled

    guardEnabled = Property(bool, _g_guard_enabled, notify=guardEnabledChanged)

    @Slot(bool)
    def setGuardEnabled(self, v: bool) -> None:  # noqa: N802
        v = _as_bool(v)
        if v != self._guard_enabled:
            self._guard_enabled = v
            self._store("guard/enabled", v)
            self.guardEnabledChanged.emit()

    def _g_bsqi(self) -> float:
        return self._bsqi

    bsqiThreshold = Property(float, _g_bsqi, notify=bsqiThresholdChanged)

    @Slot(float)
    def setBsqiThreshold(self, v: float) -> None:  # noqa: N802
        v = round(_clamp(float(v), 0.5, 0.9), 3)
        if v != self._bsqi:
            self._bsqi = v
            self._store("guard/bsqiThreshold", v)
            self.bsqiThresholdChanged.emit()

    # ---- LLM audit --------------------------------------------------------
    def _g_audit_enabled(self) -> bool:
        return self._audit_enabled

    auditEnabled = Property(bool, _g_audit_enabled, notify=auditEnabledChanged)

    @Slot(bool)
    def setAuditEnabled(self, v: bool) -> None:  # noqa: N802
        v = _as_bool(v)
        if v != self._audit_enabled:
            self._audit_enabled = v
            self._store("audit/enabled", v)
            self.auditEnabledChanged.emit()

    def _g_ollama_host(self) -> str:
        return self._ollama_host

    ollamaHost = Property(str, _g_ollama_host, notify=ollamaHostChanged)

    @Slot(str)
    def setOllamaHost(self, v: str) -> None:  # noqa: N802
        v = str(v).strip()
        if v != self._ollama_host:
            self._ollama_host = v
            self._store("audit/ollamaHost", v)
            self.ollamaHostChanged.emit()

    def _g_audit_model(self) -> str:
        return self._audit_model

    auditModel = Property(str, _g_audit_model, notify=auditModelChanged)

    @Slot(str)
    def setAuditModel(self, v: str) -> None:  # noqa: N802
        v = str(v).strip()
        if v != self._audit_model:
            self._audit_model = v
            self._store("audit/model", v)
            self.auditModelChanged.emit()

    def _g_audit_samples(self) -> int:
        return self._audit_samples

    auditSamples = Property(int, _g_audit_samples, notify=auditSamplesChanged)

    @Slot(int)
    def setAuditSamples(self, v: int) -> None:  # noqa: N802
        v = int(_clamp(int(v), 1, 5))
        if v != self._audit_samples:
            self._audit_samples = v
            self._store("audit/samples", v)
            self.auditSamplesChanged.emit()

    # ---- reset ------------------------------------------------------------
    @Slot()
    def reset(self) -> None:
        """Restore every setting to its default (persisted) and notify QML."""
        self.setThemeDark(_DEFAULTS["appearance/themeDark"])
        self.setAccent(_DEFAULTS["appearance/accent"])
        self.setColorBlindTiers(_DEFAULTS["appearance/colorBlindTiers"])
        self.setSegmentationView(_DEFAULTS["view/segmentation"])
        self.setWindowOverlap(_DEFAULTS["analysis/windowOverlap"])
        self.setRecoveryEnabled(_DEFAULTS["analysis/recoveryEnabled"])
        self.setRefineBoundaries(_DEFAULTS["analysis/refineBoundaries"])
        self.setGuardEnabled(_DEFAULTS["guard/enabled"])
        self.setBsqiThreshold(_DEFAULTS["guard/bsqiThreshold"])
        self.setAuditEnabled(_DEFAULTS["audit/enabled"])
        self.setOllamaHost(_DEFAULTS["audit/ollamaHost"])
        self.setAuditModel(_DEFAULTS["audit/model"])
        self.setAuditSamples(_DEFAULTS["audit/samples"])
