"""Unified plain-language EXPLANATION of a segment's grade — one sentence that ties together the three XAI
signals the app already computes: the occlusion-saliency map (*where* in time the model looks), the group-
Shapley attribution (*which* quality property drives the grade), and the artifact-type head (*what* kind of
corruption). Pure string assembly over already-computed inputs — no model calls — so it is cheap and unit-
testable. Composes gracefully from whatever is available (ECG has no attribution; a clean segment has no
salient region), and never overstates: it defers to the honest per-signal caveats shown alongside it.
"""
from __future__ import annotations

import numpy as np

__all__ = ["build_grade_narrative"]

# a saliency map (absolute-scaled 0..1) needs a peak this strong before we name a "region" — below it the
# grade isn't localized in time and we say so rather than inventing a spurious location.
_PEAK_MIN = 0.35
# a group's |phi| must clear this (P(unusable) units) to be called a driver — below it nothing stands out.
_PHI_MIN = 0.05


def _where_clause(saliency_map, seg_dur_s: float) -> "str | None":
    m = np.asarray(saliency_map, dtype=np.float64).reshape(-1)
    if m.size == 0 or not np.isfinite(m).any():
        return None
    mx = float(m.max())
    if mx < _PEAK_MIN:
        return None                                        # diffuse / faint — no clear region to point at
    frac_hot = float((m > 0.5 * mx).mean())                # how spread the important region is
    if frac_hot > 0.45:
        return "the model responds across much of the window"
    t = float(np.argmax(m)) / max(1, m.size) * max(0.0, seg_dur_s)
    return f"the model focuses on a region ≈{t:.1f} s in"


def _why_clause(attribution) -> "str | None":
    if not attribution:
        return None                                        # ECG (no fused SQI vector) or not computed
    groups = attribution.get("groups") or []
    if not groups:
        return None
    top = groups[0]
    if abs(float(top.get("phi", 0.0))) < _PHI_MIN:
        return "no single quality property stands out"     # clean — nothing drives it away from the tier
    lead = "driven mainly by the" if float(top["phi"]) >= 0 else "held up mainly by the"
    clause = f"the grade is {lead} {top['group']} quality features"
    if len(groups) > 1 and float(groups[1].get("share", 0.0)) >= 0.22:
        clause += f" (with {groups[1]['group']})"
    return clause


def _what_clause(artifacts) -> "str | None":
    tags = [str(a).strip() for a in (artifacts or []) if str(a).strip()]
    if not tags:
        return None
    shown = tags[:2]
    return "tagged as " + (" and ".join(shown) if len(shown) == 2 else shown[0])


def build_grade_narrative(*, tier_code: str, tier_label: str, seg_dur_s: float,
                          saliency_map=None, attribution=None, artifacts=None) -> str:
    """One- or two-sentence explanation. First sentence = the grade; second = the available *where / why /
    what* clauses joined naturally. Returns just the grade sentence if no explanation signals are present."""
    dur = f"{max(0.0, float(seg_dur_s)):.1f} s"
    head = f"Graded {tier_label} ({tier_code}) over this {dur} window."
    clauses = [c for c in (_where_clause(saliency_map, float(seg_dur_s)),
                           _why_clause(attribution),
                           _what_clause(artifacts)) if c]
    if not clauses:
        return head
    # capitalize the first clause, comma-join the rest, end with a period
    body = clauses[0][0].upper() + clauses[0][1:]
    if len(clauses) > 1:
        body += ", " + ", ".join(clauses[1:])
    return f"{head} {body}."
