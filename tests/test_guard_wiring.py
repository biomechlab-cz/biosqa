"""Unit tests for the guard / data-quality / LLM-audit wiring (no network, no GUI loop)."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication  # GUI app (superset) so it coexists with the e2e test's QApplication

from biosqa.viewmodels.guard_controller import GuardController
from biosqa.workers import qt_threads
from biosqa.workers.signals import AuditWorkerSignals, InferenceWorkerSignals

_app = QApplication.instance() or QApplication([])


# ---- GuardController property logic ---------------------------------------
def test_guard_banner_and_warning():
    g = GuardController()
    assert not g.hasWarning
    g.setGuard("ecg", {"prefiltered": True, "reasons": ["hf suppressed"], "n_overridden": 3})
    assert g.prefiltered and g.nOverridden == 3 and g.hasWarning
    assert "pre-filtered" in g.bannerText and "re-flagged" in g.bannerText


def test_guard_data_quality_triggers_warning():
    g = GuardController()
    g.setDataQuality("ecg", {"completeness": 0.7, "usable": False, "flags": ["12% clipped"]})
    assert g.completeness == 0.7 and not g.dataQualityUsable
    assert g.dataQualityFlags == ["12% clipped"] and g.hasWarning


def test_guard_audit_success_then_error():
    g = GuardController()
    g.setAuditResult({"grade": 1, "agreement": 1.0, "rationale": "pre-filtered + low bSQI"})
    assert g.hasAudit and not g.auditError and g.auditGrade == 1 and "low bSQI" in g.auditText
    g.setAuditResult({"error": "timeout"})
    assert g.auditError and "unavailable" in g.auditText.lower()


def test_guard_request_audit_emits_and_pending():
    g = GuardController()
    captured = []
    g.auditRequested.connect(lambda s, e, t, c: captured.append((s, e, t, c)))
    g.requestAudit(1.0, 3.0, "Q3", 0.9)
    assert g.auditPending and captured == [(1.0, 3.0, "Q3", 0.9)]


def test_guard_reset_clears_state():
    g = GuardController()
    g.setGuard("ecg", {"prefiltered": True, "n_overridden": 2})
    g.setAuditResult({"grade": 0, "rationale": "x"})
    g.reset()
    assert not g.hasWarning and not g.hasAudit and not g.prefiltered


# ---- worker: guard override applied to tiers + reports emitted ------------
class _Head:
    class_order = ["Q0_unacceptable", "Q1_poor", "Q2_acceptable", "Q3_excellent"]


class _Card:
    primary_head = _Head()
    artifact_head = None
    fs_hz = 250.0


class _Pred:
    primary = np.array([[0.05, 0.05, 0.10, 0.80], [0.10, 0.10, 0.10, 0.70]])  # both argmax Q3

    def get(self, _k):
        return None


class _Runner:
    card = _Card()
    modality = "ecg"

    def run_sliding_window_multihead(self, _sig, overlap=0.0):
        return _Pred()

    def guard_record(self, _sig, _pred=None, overlap=0.0, bsqi_corrupt=0.72):
        return {"prefiltered": True, "reasons": ["hf"], "n_overridden": 1,
                "override_mask": np.array([True, False]), "score": 0.9}


def test_inference_task_applies_override_and_emits_reports():
    sig = np.zeros(5000, np.float32)
    signals = InferenceWorkerSignals()
    ivs, guards, dqs = [], [], []
    signals.intervalsReady.connect(lambda m, x: ivs.append(list(x)))
    signals.guardReady.connect(lambda m, g: guards.append(g))
    signals.dataQualityReady.connect(lambda m, d: dqs.append(d))
    qt_threads.InferenceTask(_Runner(), sig, 20.0, 20.0, signals).run()
    assert guards and guards[0]["n_overridden"] == 1 and guards[0]["prefiltered"]
    assert dqs and "completeness" in dqs[0]                      # RecordQuality -> dict
    # window 0 (argmax Q3) was integrity-overridden to the worst tier Q0
    assert ivs[0][0].tier == "Q0"


def test_inference_task_skips_guard_when_disabled():
    sig = np.zeros(5000, np.float32)
    signals = InferenceWorkerSignals()
    guards, ivs = [], []
    signals.guardReady.connect(lambda m, g: guards.append(g))
    signals.intervalsReady.connect(lambda m, x: ivs.append(list(x)))
    qt_threads.InferenceTask(_Runner(), sig, 20.0, 20.0, signals, guard_enabled=False).run()
    assert ivs and not guards            # segmentation ran; guard skipped -> no guardReady
    assert ivs[0][0].tier == "Q3"        # no override applied -> window 0 keeps its model tier


def test_inference_task_threads_overlap_into_inference():
    seen = {}

    class R(_Runner):
        def run_sliding_window_multihead(self, _sig, overlap=0.0):
            seen["overlap"] = overlap
            return _Pred()

    qt_threads.InferenceTask(R(), np.zeros(5000, np.float32), 10.0, 20.0,
                             InferenceWorkerSignals(), overlap=0.5, guard_enabled=False).run()
    assert seen.get("overlap") == 0.5


def test_audit_task_emits_judgment(monkeypatch):
    monkeypatch.setattr(qt_threads, "audit_segment",
                        lambda *a, **k: {"grade": 2, "rationale": "ok", "agreement": 1.0})
    got = []
    sig = AuditWorkerSignals()
    sig.auditReady.connect(lambda j: got.append(j))
    qt_threads.AuditTask(_Runner(), np.zeros(500, np.float32), {"grade": 3}, {"prefiltered": True}, sig).run()
    assert got and got[0]["grade"] == 2


def test_audit_task_graceful_when_llm_unreachable():
    # real audit_segment against an unreachable host -> {"error": ...}, never raises
    from biosqa.inference.llm_audit import audit_segment
    out = audit_segment(np.zeros(500, np.float32), 250.0, "ecg", host="http://127.0.0.1:1", samples=1, timeout=2.0)
    assert "error" in out
