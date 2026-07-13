---
title: "Recording overview (quality dashboard)"
description: "The KPI dashboard: usable share, tier fractions, artifact bars, and the model-card panel."
---

The recording overview is a whole-recording summary you read once to decide whether a signal is fit for downstream analysis. It condenses the per-window assessment into a handful of headline numbers, a tier distribution, and a per-modality picture of where and how quality degrades.

## A bento KPI dashboard

The overview is laid out as a **bento dashboard** — a grid of self-contained tiles, each answering one question about the recording as a whole. Rather than scrolling the signal, you glance at the tiles to judge overall fitness for purpose, then drill into the [segmentation](/biosqa/docs/segmentation/) and [segment inspector](/biosqa/docs/segment-inspector/) views for detail.

## Headline KPIs

The top tiles carry the numbers you act on:

| KPI | Meaning |
| --- | --- |
| **Total duration** | Length of the recording as loaded. |
| **Usable share** | The combined **Q3 + Q2** percentage — the fraction graded acceptable or excellent. |
| **Per-tier fractions** | The share of the recording in each grade, Q0 through Q3. |

**Usable share** is the single number most often used to accept or reject a recording. Recall that **usable = Q2 + Q3**: Q2 (acceptable) supports rate and coarse analytics, Q3 (excellent) supports all analytics. See the [quality scale](/biosqa/docs/quality-scale/) for what each tier permits.

## Distribution

A **donut** renders the Q0..Q3 tier fractions as a ring. Each arc carries both **color and glyph** (⊘ Q0, ⚠ Q1, ✓ Q2, ✓ Q3), never color alone, and the ring honors the **color-blind-safe palette toggle** so the distribution stays legible under either palette.

> Color is never load-bearing on its own here. The mono codes Q0..Q3 and the per-tier glyphs carry the same information as the arcs, so the donut reads correctly regardless of palette or vision.

## Per-modality timeline & artifact bars

For each modality present, the overview shows a **quality-ribbon timeline** — a compressed strip that colors the recording by grade across its full duration, so you can see whether poor quality is a transient artifact burst or sustained degradation across the record.

Alongside it, **artifact-type bars** summarise which artifact classes dominate the recording. Where the timeline tells you *when* quality drops, the artifact bars tell you *what kind* of corruption is responsible.

## Model Card panel

The **Model Card** panel shows the parsed contents of the modality's `model_card.json`, so you can see exactly which model and settings produced the assessment on screen:

```
fs              sampling rate the model expects
window length   analysis window the grades are computed over
class order     the Q0..Q3 label ordering
normalization   input normalization applied before inference
version         model version identifier
```

Because the app consumes only `models/<modality>.onnx` plus its `models/<modality>.model_card.json`, this panel is the authoritative record of what ran. For how these fields are used and where the files live, see [Models](/biosqa/docs/models/).

## Quality Inspector indicators

When relevant, the dock also surfaces **record-level guard and data-quality flags** alongside the KPIs. Two indices describe how far the recording sits from the model's training distribution:

- a **domain-shift index**, and
- a **novelty fraction** — how much of the signal's SQI signature is unlike anything seen in training.

> A high domain-shift index or novelty fraction means the grades are extrapolations: the model is scoring signal whose SQI signature it was not trained on, so treat the assessment with corresponding caution.

These indicators come from the same runtime layer that raises the other flags — see [Runtime guards](/biosqa/docs/runtime-guards/) for what each guard checks and how to read it.
