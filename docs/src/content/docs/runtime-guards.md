---
title: "Runtime guards: pre-filtering, false-clean & recoverability"
description: "Why raw matters, the false-clean integrity guard, the recoverability pass, and boundary refinement."
---

Runtime guards are advisory passes that run alongside the model's grade to catch the cases raw scoring alone can miss. They never silently change the decision. The **raw model grade stays authoritative** everywhere, and every guard only annotates, re-flags, or narrows what you already see.

## Why raw matters

Each modality's model is trained on **raw signals**. A signal that has already been filtered upstream can be scored **clean** even when it is corrupt, because filtering strips the high-frequency cues the model keys on. The evidence of the artifact is gone before the model ever sees it.

> Feed raw. A pre-filtered-but-corrupt input tends to be **under-flagged**: bad windows that look good. See [Opening recordings](/biosqa/docs/opening-recordings/) for how the app ingests signals.

## False-clean integrity guard

When the input looks pre-filtered, the app runs a **filter-robust beat-detector-agreement cue** (**bSQI**) that does not depend on the high-frequency content filtering removes. bSQI votes against a confidently-clean grade: where it **disagrees** with the model, those windows are re-flagged to the **worst tier** so they surface on the track rather than hiding.

The bSQI threshold is adjustable from **0.5 to 0.9** in [Settings](/biosqa/docs/settings/). A **record-level banner** explains what was detected as pre-filtered and why specific windows were re-flagged, so the re-flagging is never opaque.

> The guard re-flags, it does not re-score. The raw model grade is still recorded; the guard only forces suspect windows into view.

## Recoverability pass

The recoverability pass is a **second scoring pass on a band-pass-filtered copy** of the signal. It identifies **poor windows that a standard filter would likely lift to usable**: signal that is currently below the usable line but plausibly salvageable with routine preprocessing. These are shown with a **↺ marker**.

For **ECG and PPG**, the recoverable flag is **bSQI-corroborated** before it is shown. The pass **never re-grades**: the raw tier stays authoritative, and ↺ is an annotation on top of it, not a replacement.

## Rate-usable advisory

Some segments have **poor morphology but reliable beats**: the waveform shape is too degraded for morphological analytics, yet the beat timing is trustworthy. The rate-usable advisory marks these so **heart/pulse rate stays usable**, and shows the **bpm** for the segment.

## Boundary refinement

The model grades on **coarse windows**, so a short artifact burst smears its poor grade across the whole window, and window **overlap makes the flagged region wider, not narrower**. Boundary refinement computes a **fine per-bin badness score** to localize a poor segment down to its actual **artifact core**.

Refinement is conservative by construction:

- It **only ever shrinks** poor regions. It never grows them or moves them into clean signal.
- If it **cannot localize a core**, it leaves the segment **untouched**.

## Data-quality report

Alongside the grades, the app produces a data-quality report covering **completeness** plus **flatline**, **clipping**, **dropout**, and **NaN** flags, with a **not-usable** indicator for affected regions.

> A **non-finite input window fails safe to Q0 with zero confidence**. The app never exports a NaN. A window it cannot trust becomes unacceptable, not silently clean.

See [Quality scale](/biosqa/docs/quality-scale/) for what Q0–Q3 mean and [Exporting](/biosqa/docs/exporting/) for how flags travel with results.

## Advisory-only and streaming

All of these passes are **advisory**. To keep memory bounded on multi-hour recordings, guards, recoverability, and boundary refinement are **disabled for out-of-core streamed (very large) records**.

| Pass | Streamed (out-of-core) | Effect on grade |
|---|---|---|
| False-clean guard | disabled | re-flags to worst tier (advisory) |
| Recoverability (↺) | disabled | annotation only, never re-grades |
| Boundary refinement | disabled | only shrinks poor regions |

> The **raw grade is authoritative everywhere**: with or without the guards, streamed or in-memory. The guards add context; they do not overrule the model.
