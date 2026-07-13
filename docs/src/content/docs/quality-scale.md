---
title: "The Q0–Q3 quality scale"
description: "The four ordinal grades, run-length bands, confidence/uncertainty, and the artifact types."
---

BioSQA Studio grades every window of a signal on a four-level ordinal scale from Q0 (unacceptable) to Q3 (excellent). This page explains what each grade means, how per-window scores become the bands you see on the trace, and the confidence, uncertainty, and artifact-type information attached to each segment.

## The four grades

The scale is **ordinal**: higher grades support more downstream analysis. "**Usable**" is shorthand for **Q2 + Q3** — signal you can compute something meaningful from.

| Grade | Name | Glyph | Supports |
|-------|------|-------|----------|
| **Q3** | Excellent | ✓ | All analytics, including fine morphology and HRV |
| **Q2** | Acceptable | ✓ | Rate and coarse features — not fine morphology |
| **Q1** | Poor | ⚠ | Partially corrupted; use with caveats |
| **Q0** | Unacceptable | ⊘ | Dominated by artifact; discard |

The grade tells you what a segment is *good enough for*, not just whether it is "clean." A Q2 segment of ECG is fine for a heart-rate estimate but not for beat-morphology work; a Q0 segment should be dropped from analysis entirely.

## Ordinal, not binary

The four grades form a gradient, not four unrelated buckets. Adjacent grades are "closer" than distant ones — mistaking Q2 for Q3 is a smaller error than mistaking Q2 for Q0. During model development this is measured with **quadratic-weighted kappa**, which penalizes distant confusions more than near ones, alongside **macro-F1** and **Cohen's kappa**.

> As a user, the practical takeaway is to treat the scale as a **gradient**: a boundary between Q1 and Q2 is a soft transition, not a hard cliff, and a recording that hovers around a grade boundary deserves a closer look. See [Manual review](/biosqa/docs/manual-review/) when you disagree with a call.

## From per-window scores to run-length bands

The model does not grade the whole recording at once. It scores **overlapping, fixed-length windows** as it streams across the signal. Where adjacent windows share the same grade, those windows are **run-length-encoded** into a single contiguous **segment**, and each segment is painted as a band on the trace.

The **window length is fixed by the model card**, not by the UI — for example, ECG at 10 s and 250 Hz. You cannot change it from the interface; it is a property of the trained model. This is why segment boundaries fall on window-aligned positions. For how these bands are rendered and navigated, see [Segmentation](/biosqa/docs/segmentation/).

## The palette and the color-blind-safe alternate

Grade is never communicated by color alone. Every segment carries three redundant cues:

- a **color**,
- a **glyph** (✓ / ⚠ / ⊘), and
- a **mono code** (Q0..Q3).

Q0 is additionally **hatched** so it stands out even in grayscale. A **blue↔orange color-blind-safe palette** toggle is available in [Settings](/biosqa/docs/settings/) for viewers who need it.

## Confidence and uncertainty

Each segment carries two numbers derived from the model's **temperature-calibrated** output distribution:

| Quantity | Definition |
|----------|------------|
| **Confidence** | The calibrated **max-softmax** — how strongly the model backs its chosen grade |
| **Uncertainty** | The **normalized softmax entropy** — how spread the distribution is across grades |

Because both come from the *calibrated* distribution, they are meant to be read as honest self-assessments rather than raw logits. **Uncertainty is surfaced in the interface** and turns **amber** when a segment is shaky — a cue to inspect it manually rather than trust the grade at face value. The [Segment inspector](/biosqa/docs/segment-inspector/) shows these values per segment.

## Artifact types flagged

For modalities that include an **artifact head**, the model additionally produces a **multilabel** tagging of the *kind* of corruption present in a segment — several types can co-occur. Names are plain-language:

| Type | Type |
|------|------|
| muscle / EMG | dropout |
| powerline | electrode |
| baseline wander | spike |
| motion | noise |
| clipping / flatline | burst / transient |

> Not every modality has an artifact head. Where one is absent, segments carry grades, confidence, and uncertainty, but no per-type artifact tags. See [Models](/biosqa/docs/models/) for which modality provides what.

Feed **raw** signals: a pre-filtered but still-corrupt recording can be under-flagged, because filtering can hide the very artifacts the model looks for. BioSQA Studio is a research and engineering tool, **not a medical device**.
