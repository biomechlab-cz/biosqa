"""Guard / data-quality / LLM-audit state -> QML (the false-clean + integrity surfaces).

Single QML-facing controller (`guard`) for the three off-the-happy-path signals the model alone
can't give:
  * the FALSE-CLEAN guard (pre-filter detector + filter-robust bSQI integrity override) -> a warning
    banner + a count of windows re-flagged;
  * the record DATA-QUALITY report (completeness / flatline / clipping / dropout gaps);
  * an on-demand LLM AUDIT of the selected segment (off the decision path — a human-readable second
    opinion, never a gate; graceful when the local LLM is unavailable).

The Coordinator feeds the guard/data-quality dicts in (from the inference worker) and services
``requestAudit`` by launching an ``AuditTask`` and routing the judgment back to :meth:`setAuditResult`.
"""

from __future__ import annotations

from PySide6.QtCore import Property, QObject, Signal, Slot

_TIER_LABEL = {"Q0": "unacceptable", "Q1": "poor", "Q2": "acceptable", "Q3": "excellent"}


class GuardController(QObject):
    """Exposed to QML as ``guard`` (context property)."""

    guardChanged = Signal()
    dataQualityChanged = Signal()
    auditChanged = Signal()
    sqiChanged = Signal()
    saliencyChanged = Signal()
    #: (startSec, endSec, tier, confidence) — the Coordinator connects this to launch an AuditTask.
    auditRequested = Signal(float, float, str, float)
    #: (startSec, endSec) — the Coordinator computes the interpretable SQI breakdown for that window.
    sqiRequested = Signal(float, float)
    #: (startSec, endSec) — the Coordinator launches a SaliencyTask (occlusion XAI) for that window.
    saliencyRequested = Signal(float, float)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._prefiltered = False
        self._reasons: list[str] = []
        self._n_overridden = 0
        self._completeness = 1.0
        self._dq_usable = True
        self._dq_flags: list[str] = []
        self._dsi = 0.0
        self._regime_flags: list[str] = []
        self._novelty_frac = 0.0
        self._audit_pending = False
        self._audit_text = ""
        self._audit_grade = -1
        self._audit_agreement = 0.0
        self._audit_error = False
        self._has_audit = False
        self._saliency: list = []
        self._saliency_pending = False
        self._grade_attribution: list = []      # [{group, phi, share}] — group-Shapley grade attribution
        self._grade_narrative = ""              # unified plain-language explanation (where + why + what)
        self._explain_ctx: dict = {}            # segment context stashed at request time for the narrative
        self._sqi: list = []
        self._sqi_filtered: list = []
        self._sqi_consensus: float = 0.0
        self._usability: list = []

    # ---- interpretable SQI breakdown (per selected segment) ----------------
    def _get_sqi(self) -> list:
        return self._sqi

    sqiBreakdown = Property("QVariant", _get_sqi, notify=sqiChanged)

    def _get_sqi_filtered(self) -> list:
        return self._sqi_filtered

    #: the SAME bank recomputed on a band-pass-filtered copy of the window (Raw/Filtered toggle).
    sqiBreakdownFiltered = Property("QVariant", _get_sqi_filtered, notify=sqiChanged)

    def _get_sqi_consensus(self) -> float:
        return self._sqi_consensus

    #: fused 0..1 quality consensus over the RAW bank (higher = cleaner); drives the model-vs-SQI banner.
    sqiConsensus = Property(float, _get_sqi_consensus, notify=sqiChanged)

    @Slot(float, float)
    def requestSqi(self, start_sec: float, end_sec: float) -> None:  # noqa: N802
        """QML entry point (on segment select): ask the Coordinator for the classical-SQI breakdown."""
        self.sqiRequested.emit(float(start_sec), float(end_sec))

    @Slot("QVariant")
    @Slot("QVariant", "QVariant", float)
    def setSqiBreakdown(self, rows, filtered=None, consensus: float = 0.0) -> None:  # noqa: N802
        self._sqi = list(rows) if rows else []
        self._sqi_filtered = list(filtered) if filtered else []
        self._sqi_consensus = float(consensus)
        self.sqiChanged.emit()
        if self._saliency or self._saliency_pending or self._grade_attribution or self._grade_narrative:
            self._saliency = []                             # a new segment invalidates the old explanation
            self._saliency_pending = False
            self._grade_attribution = []
            self._grade_narrative = ""
            self.saliencyChanged.emit()

    def _get_usability(self) -> list:
        return self._usability

    #: per-modality "usable for what" verdicts for the selected segment — EEG per-band (δ/θ/α/β/γ),
    #: EDA tonic/phasic; ``[]`` for ECG/PPG (their rate verdict is the rate-usable card). Each is
    #: ``{label, usable, detail}``.
    usabilityVerdicts = Property("QVariant", _get_usability, notify=sqiChanged)

    @Slot("QVariant")
    def setUsability(self, verdicts) -> None:  # noqa: N802
        self._usability = list(verdicts) if verdicts else []
        self.sqiChanged.emit()

    # ---- occlusion saliency (on-demand XAI: 'what is the model looking at?') ----
    def _get_saliency(self) -> list:
        return self._saliency

    #: per-sample occlusion-saliency map (0..1, downsampled) for the selected segment — the UI paints it
    #: as an importance heatmap over the trace. Empty until 'Explain' is run. A perturbation-based ESTIMATE.
    saliencyMap = Property("QVariant", _get_saliency, notify=saliencyChanged)

    def _get_saliency_pending(self) -> bool:
        return self._saliency_pending

    saliencyPending = Property(bool, _get_saliency_pending, notify=saliencyChanged)

    def _get_grade_attribution(self) -> list:
        return self._grade_attribution

    #: group-Shapley feature attribution for the selected segment's grade — a ranked list of
    #: ``{group, phi, share}`` ("which signal-quality property drives the grade"), the *why* complement to
    #: the *where* heatmap. ``[]`` for ECG (no fused SQI vector) or before 'Explain'. φ>0 pushes toward
    #: unusable, φ<0 toward usable; ``share`` = |φ|/Σ|φ|. A perturbation-based ESTIMATE, not ground truth.
    gradeAttribution = Property("QVariant", _get_grade_attribution, notify=saliencyChanged)

    def _get_grade_narrative(self) -> str:
        return self._grade_narrative

    #: one plain-language sentence unifying the three XAI signals (where / why / what) for the selected
    #: segment — built when 'Explain' completes. Empty before then. An honest summary, not a claim of truth.
    gradeNarrative = Property(str, _get_grade_narrative, notify=saliencyChanged)

    @Slot(float, float)
    @Slot(float, float, str, str, "QVariant")
    def requestSaliency(self, start_sec: float, end_sec: float,
                        tier_code: str = "", tier_label: str = "", artifacts=None) -> None:  # noqa: N802
        """QML entry ('Explain'): ask the Coordinator to run occlusion saliency + feature attribution. The
        segment context (tier + artifacts) is stashed so the unified narrative can be assembled when the
        saliency/attribution return (they carry only the map + group Shapley values)."""
        self._explain_ctx = {
            "tier_code": str(tier_code or ""),
            "tier_label": str(tier_label or ""),
            "seg_dur_s": max(0.0, float(end_sec) - float(start_sec)),
            "artifacts": list(artifacts) if artifacts else [],
        }
        self._saliency_pending = True
        self.saliencyChanged.emit()
        self.saliencyRequested.emit(float(start_sec), float(end_sec))

    @Slot("QVariant")
    def setSaliency(self, payload) -> None:  # noqa: N802
        p = payload or {}
        attribution = p.get("attribution")
        self._saliency = list(p.get("map", []))
        self._grade_attribution = list(attribution.get("groups", []) if attribution else [])
        self._saliency_pending = False
        self._grade_narrative = self._build_narrative(p.get("map", []), attribution)
        self.saliencyChanged.emit()

    def _build_narrative(self, saliency_map, attribution) -> str:
        """Assemble the unified where/why/what sentence from the returned map + attribution and the stashed
        segment context. Advisory only — never let a failure here break the (already-populated) heatmap."""
        ctx = self._explain_ctx
        if not ctx.get("tier_code"):
            return ""
        try:
            from biosqa.inference.narrative import build_grade_narrative
            return build_grade_narrative(
                tier_code=ctx["tier_code"], tier_label=ctx["tier_label"], seg_dur_s=ctx["seg_dur_s"],
                saliency_map=saliency_map, attribution=attribution, artifacts=ctx.get("artifacts"))
        except Exception:  # noqa: BLE001
            return ""

    # ---- false-clean guard banner -----------------------------------------
    def _get_prefiltered(self) -> bool:
        return self._prefiltered

    prefiltered = Property(bool, _get_prefiltered, notify=guardChanged)

    def _get_n_overridden(self) -> int:
        return self._n_overridden

    nOverridden = Property(int, _get_n_overridden, notify=guardChanged)

    def _get_reasons(self) -> list:
        return self._reasons

    guardReasons = Property(list, _get_reasons, notify=guardChanged)

    def _get_banner(self) -> str:
        parts = []
        if self._prefiltered:
            parts.append("Input looks pre-filtered — reported quality may be optimistic (false-clean).")
            if self._n_overridden > 0:
                parts.append(f"{self._n_overridden} window(s) re-flagged by the integrity guard.")
        if self._dq_flags:
            parts.append("Data quality: " + "; ".join(self._dq_flags) + ".")
        return " ".join(parts)

    bannerText = Property(str, _get_banner, notify=guardChanged)

    def _get_has_warning(self) -> bool:
        return self._prefiltered or bool(self._dq_flags)

    hasWarning = Property(bool, _get_has_warning, notify=guardChanged)

    # ---- record data-quality ----------------------------------------------
    def _get_completeness(self) -> float:
        return self._completeness

    completeness = Property(float, _get_completeness, notify=dataQualityChanged)

    def _get_dq_usable(self) -> bool:
        return self._dq_usable

    dataQualityUsable = Property(bool, _get_dq_usable, notify=dataQualityChanged)

    def _get_dq_flags(self) -> list:
        return self._dq_flags

    dataQualityFlags = Property(list, _get_dq_flags, notify=dataQualityChanged)

    # ---- on-demand LLM audit ----------------------------------------------
    def _get_audit_pending(self) -> bool:
        return self._audit_pending

    auditPending = Property(bool, _get_audit_pending, notify=auditChanged)

    def _get_audit_text(self) -> str:
        return self._audit_text

    auditText = Property(str, _get_audit_text, notify=auditChanged)

    def _get_audit_grade(self) -> int:
        return self._audit_grade

    auditGrade = Property(int, _get_audit_grade, notify=auditChanged)

    def _get_audit_agreement(self) -> float:
        return self._audit_agreement

    auditAgreement = Property(float, _get_audit_agreement, notify=auditChanged)

    def _get_audit_error(self) -> bool:
        return self._audit_error

    auditError = Property(bool, _get_audit_error, notify=auditChanged)

    def _get_has_audit(self) -> bool:
        return self._has_audit

    hasAudit = Property(bool, _get_has_audit, notify=auditChanged)

    # ---- slots fed by the Coordinator / worker ----------------------------
    @Slot(str, "QVariant")
    def setGuard(self, modality: str, report) -> None:  # noqa: N802
        report = report or {}
        self._prefiltered = bool(report.get("prefiltered", False))
        self._reasons = [str(r) for r in report.get("reasons", [])]
        self._n_overridden = int(report.get("n_overridden", 0))
        self.guardChanged.emit()

    @Slot(str, "QVariant")
    def setDataQuality(self, modality: str, report) -> None:  # noqa: N802
        report = report or {}
        self._completeness = float(report.get("completeness", 1.0))
        self._dq_usable = bool(report.get("usable", True))
        self._dq_flags = [str(f) for f in report.get("flags", [])]
        self._dsi = float(report.get("dsi", 0.0))
        self._regime_flags = [str(f) for f in report.get("regime_flags", [])]
        self._novelty_frac = float(report.get("novelty_frac", 0.0))
        self.dataQualityChanged.emit()
        self.guardChanged.emit()  # bannerText/hasWarning also depend on data-quality flags

    def _get_dsi(self) -> float:
        return self._dsi

    def _get_novelty_frac(self) -> float:
        return self._novelty_frac

    #: fraction of windows whose interpretable SQI vector is novel vs the training set (feature-space
    #: Mahalanobis OOD). Elevated (≫ the calibrated ~1%) → possible new device/cohort. Explained in regimeFlags.
    noveltyFraction = Property(float, _get_novelty_frac, notify=dataQualityChanged)

    #: Domain-Shift Index 0..1 (research3 cross-dataset weakness): how far the recording's SPECTRUM sits
    #: from the model's expected acquisition regime (narrow band / aliasing). High → trust scores less.
    domainShiftIndex = Property(float, _get_dsi, notify=dataQualityChanged)

    def _get_regime_flags(self) -> list:
        return self._regime_flags

    regimeFlags = Property(list, _get_regime_flags, notify=dataQualityChanged)

    @Slot(float, float, str, float)
    def requestAudit(self, start_sec: float, end_sec: float, tier: str, confidence: float) -> None:  # noqa: N802
        """QML entry point ('Audit this segment'): mark pending + ask the Coordinator to run it."""
        self._audit_pending = True
        self._audit_error = False
        self._has_audit = True
        self._audit_text = "Auditing… (local LLM second opinion)"
        self.auditChanged.emit()
        self.auditRequested.emit(float(start_sec), float(end_sec), str(tier), float(confidence))

    @Slot("QVariant")
    def setAuditResult(self, judgment) -> None:  # noqa: N802
        """Called by the Coordinator when the AuditTask finishes."""
        self._audit_pending = False
        self._has_audit = True
        judgment = judgment or {}
        if "error" in judgment:
            self._audit_error = True
            self._audit_grade = -1
            self._audit_agreement = 0.0
            self._audit_text = "LLM audit unavailable — is a local ollama server running? " \
                               f"({judgment.get('error', '')})"
        else:
            self._audit_error = False
            g = judgment.get("grade")
            self._audit_grade = int(g) if g is not None else -1
            self._audit_agreement = float(judgment.get("agreement", 0.0))
            label = _TIER_LABEL.get(f"Q{self._audit_grade}", "")
            rationale = judgment.get("rationale", "") or "(no rationale returned)"
            head = f"LLM second opinion: Q{self._audit_grade} ({label})" if self._audit_grade >= 0 else "LLM second opinion"
            if self._audit_agreement:
                head += f" · {self._audit_agreement:.0%} self-consistency"
            self._audit_text = f"{head}\n{rationale}"
        self.auditChanged.emit()

    @Slot()
    def reset(self) -> None:
        """Clear all guard/audit state (called when a new recording is opened)."""
        self._prefiltered = False
        self._reasons = []
        self._n_overridden = 0
        self._completeness = 1.0
        self._dq_usable = True
        self._dq_flags = []
        self._dsi = 0.0
        self._regime_flags = []
        self._novelty_frac = 0.0
        self._saliency = []
        self._saliency_pending = False
        self._grade_attribution = []
        self._grade_narrative = ""
        self._explain_ctx = {}
        self._audit_pending = False
        self._audit_text = ""
        self._audit_grade = -1
        self._audit_agreement = 0.0
        self._audit_error = False
        self._has_audit = False
        self._sqi = []
        self._sqi_filtered = []
        self._sqi_consensus = 0.0
        self._usability = []
        self.guardChanged.emit()
        self.dataQualityChanged.emit()
        self.auditChanged.emit()
        self.sqiChanged.emit()
        self.saliencyChanged.emit()
