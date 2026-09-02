---
title: "Segment inspector & reshaping"
description: "The single-segment deep dive, the SQI breakdown, and reshaping boundaries."
---

The segment inspector is a full-page view of one selected segment: its waveform, its grade and rationale, an interpretable bank of classical signal-quality indices, and the tools to edit segmentation boundaries directly.

## The single-segment deep dive

Selecting a segment opens a full-page view of that one window. It shows a **zoomed waveform** of the segment's real samples alongside its metadata:

- a **tier badge**: the ordinal grade (Q0..Q3) as color + glyph + mono code
- **start/end times** and **duration**
- the **modality** (ECG, PPG, EEG, or EDA)
- the **rationale** and **artifact cards** describing why the grade was assigned
- a **confidence gauge**

This is the place to look closely at a single window before deciding whether to trust, relabel, or reshape it. Grades are ordinal, see [the quality scale](/biosqa/docs/quality-scale/) for what Q0..Q3 mean.

## Signal Quality Indices breakdown

Below the waveform is an interpretable bank of **classical SQIs** computed on the selected window. The exact indices depend on the modality, and may include measures such as **bSQI**, **template correlation**, **rhythm regularity**, **kurtosis**, **HF-noise**, and **spectral / aperiodic** measures.

Each index is rendered as a **quality-fill bar**: a full bar means good, so all bars read in the same direction regardless of the underlying metric's sign.

> These indices are **explanatory only**. They do not change the grade the model assigned; they exist to help you interpret it.

### Raw / Filtered toggle

A **Raw / Filtered** toggle recomputes the same bank on a band-pass-filtered copy of the window. Comparing the two tells you what a filter **would**: or would **not**: clean up. Remember that the model itself expects [raw signals](/biosqa/docs/opening-recordings/): a pre-filtered but corrupt signal can be under-flagged.

### Discordance banner

When the **model grade is clean but the classical indices disagree**, a **discordance banner** appears. It flags windows where the learned grade and the interpretable measures point in different directions. A cue to inspect the segment yourself rather than take either at face value.

## Explainability

An on-demand **"Explain this grade"** action produces a heatmap, feature attribution, and a plain-language summary for the selected segment. See [explainability](/biosqa/docs/explainability/) for what these views show and how to read them.

## Reshaping the segmentation

The inspector lets you edit the segmentation itself:

- **trim** a boundary
- **split** a segment
- **merge** two segments
- **reclassify** a tier

These edits mutate the overlay, the segment table, and exports **live**.

> **Reshaping is a hard boundary edit**: it changes where segments begin and end. This is distinct from a **soft relabel**, which preserves the model's original grade for audit. Choose the [manual review](/biosqa/docs/manual-review/) soft-relabel path when you want to record a human judgment without discarding what the model said.
