"""SettingsController: defaults, QSettings persistence + type round-trip, clamping/snapping, reset."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from biosqa.viewmodels.settings_controller import SettingsController  # noqa: E402

_app = QApplication.instance() or QApplication([])


def _backend(tmp_path, name="s.ini"):
    return QSettings(str(tmp_path / name), QSettings.Format.IniFormat)


def test_defaults(tmp_path):
    s = SettingsController(backend=_backend(tmp_path))
    assert s.themeDark is True and s.accent == "#35D0BA" and s.colorBlindTiers is False
    assert s.windowOverlap == 0.5 and s.guardEnabled is True and s.bsqiThreshold == 0.72
    assert s.auditEnabled is True and s.ollamaHost == "http://localhost:11434"
    assert s.auditModel == "gemma4:latest" and s.auditSamples == 1
    assert s.segmentationView == "table"


def test_segmentation_view_persists(tmp_path):
    s = SettingsController(backend=_backend(tmp_path))
    s.setSegmentationView("grid")
    assert s.segmentationView == "grid"
    s.setSegmentationView("nonsense")                 # anything not 'grid' normalizes to 'table'
    assert s.segmentationView == "table"
    s.setSegmentationView("grid")
    # a fresh controller on the same backing file restores the last-used view
    assert SettingsController(backend=_backend(tmp_path)).segmentationView == "grid"


def test_setters_persist_and_roundtrip_types(tmp_path):
    s = SettingsController(backend=_backend(tmp_path))
    s.setThemeDark(False); s.setAccent("#5B9CFF"); s.setColorBlindTiers(True)
    s.setWindowOverlap(0.5); s.setGuardEnabled(False); s.setBsqiThreshold(0.8)
    s.setAuditEnabled(False); s.setOllamaHost("http://x:1"); s.setAuditModel("gemma4:latest")
    s.setAuditSamples(5)
    # reload from the SAME backing file -> everything persisted with correct Python types
    s2 = SettingsController(backend=_backend(tmp_path))
    assert s2.themeDark is False and isinstance(s2.themeDark, bool)
    assert s2.accent == "#5B9CFF" and s2.colorBlindTiers is True
    assert s2.windowOverlap == 0.5 and isinstance(s2.windowOverlap, float)
    assert s2.guardEnabled is False and abs(s2.bsqiThreshold - 0.8) < 1e-9
    assert s2.auditEnabled is False and s2.ollamaHost == "http://x:1"
    assert s2.auditModel == "gemma4:latest"
    assert s2.auditSamples == 5 and isinstance(s2.auditSamples, int)


def test_clamping_and_overlap_snapping(tmp_path):
    s = SettingsController(backend=_backend(tmp_path))
    s.setBsqiThreshold(0.99); assert s.bsqiThreshold == 0.9      # clamp high
    s.setBsqiThreshold(0.10); assert s.bsqiThreshold == 0.5      # clamp low
    s.setAuditSamples(9); assert s.auditSamples == 5             # clamp high
    s.setAuditSamples(0); assert s.auditSamples == 1             # clamp low
    s.setWindowOverlap(0.30); assert s.windowOverlap == 0.25     # snap to nearest allowed
    s.setWindowOverlap(0.80); assert s.windowOverlap == 0.75     # 75% option
    s.setWindowOverlap(0.90); assert s.windowOverlap == 0.9      # 90% option (added)
    s.setWindowOverlap(2.0); assert s.windowOverlap == 0.9       # snaps to the max allowed


def test_reset_restores_defaults(tmp_path):
    s = SettingsController(backend=_backend(tmp_path))
    s.setThemeDark(False); s.setBsqiThreshold(0.9); s.setAuditSamples(5); s.setGuardEnabled(False)
    s.reset()
    assert s.themeDark is True and s.bsqiThreshold == 0.72 and s.auditSamples == 1 and s.guardEnabled is True


def test_notify_only_on_change(tmp_path):
    s = SettingsController(backend=_backend(tmp_path))
    fired = []
    s.themeDarkChanged.connect(lambda: fired.append(1))
    s.setThemeDark(False)   # change -> 1 emit
    s.setThemeDark(False)   # no change -> no emit
    assert fired == [1]


def test_bool_string_coercion_from_backend(tmp_path):
    # QSettings/INI can hand back bools as strings; the controller must coerce, not treat "false" as truthy.
    b = _backend(tmp_path)
    b.setValue("appearance/themeDark", "false")
    b.setValue("guard/enabled", "true")
    b.sync()
    s = SettingsController(backend=_backend(tmp_path))
    assert s.themeDark is False and s.guardEnabled is True
