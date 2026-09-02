---
title: "Explainability (XAI)"
description: "On-demand saliency heatmap, group-Shapley feature attribution, and a plain-language explanation, all gradient-free and on-device."
---

Explainability answers a narrow question: *why did the model grade this segment the way it did?* Because BioSQA Studio runs its models under **ONNX Runtime with no autograd at inference**, every explanation here is **gradient-free**: computed purely from forward passes on perturbed inputs.

That has a consequence worth stating up front: these views are **estimates** of what the model attends to, not ground truth about its internals. The UI says so wherever it shows them.

## Explain this grade

In the [Segment Inspector](/biosqa/docs/segment-inspector/), an on-demand **"Explain this grade"** action runs three complementary views for the currently selected segment. The computation runs **off the GUI thread**, so the interface stays responsive while the perturbation passes complete.

The three views answer three different questions:

| View | Question | Applies to |
|------|----------|------------|
| **Saliency heatmap** | *Where* in the window? | all modalities |
| **Feature attribution** | *Which property* of the signal? | fusion models: EDA here, plus PPG and EEG if you supply weights |
| **Plain-language summary** | *What's the takeaway?* | all modalities |

## Saliency heatmap (where)

An **occlusion-based** importance band painted over the zoomed waveform. The method slides a **clean occluder** across the window and measures how much each region changes the grade, regions that move the grade the most are marked as the most important.

**Amber** highlights the time regions that most drive the grade. The band is **absolute-scaled**: a genuinely clean window stays faint rather than having its low-level fluctuations stretched to fill the color range. In other words, faintness is informative. It means no region strongly moves the grade.

## Feature attribution (which property)

For the **fusion models**, whose grade fuses a hand-crafted **SQI + dynamics vector**, the app computes an **exact group-Shapley attribution**. It ranks roughly **5–6 interpretable feature groups** by how much each pushes the grade toward *unusable* versus *usable*, relative to a clean-tier reference. Of the bundled models this applies to **EDA**; PPG and EEG are fusion models too and light up as soon as you supply weights for them. ECG is excluded by architecture, not by availability: its grade head reads the raw branch alone.

The groups are interpretable by construction, for example:

| Group | What it captures |
|-------|------------------|
| Noise / HF | High-frequency noise content |
| Spectral | Frequency-domain shape |
| Morphology | Waveform shape features |
| Complexity / dynamics | Signal complexity measures |
| … | (remaining groups) |

> **ECG shows the heatmap but not these bars.** The ECG grade reads the **raw + spectral channels** directly, not the SQI+dynamics vector, so there is no feature-group vector to attribute over. The heatmap still applies.

## Plain-language summary (the takeaway)

One sentence that ties the three signals together into a reading, for example:

```
Graded Poor (Q1) over this 8.0 s window. The model focuses on a
region ~3.1 s in, the grade is driven mainly by the Noise/HF
quality features, tagged as motion.
```

It names the [grade](/biosqa/docs/quality-scale/), the region the heatmap points to, and, for fusion models, the dominant feature group, so you can move from *"where"* and *"which property"* to a single takeaway without reading the two visual panels in detail.

## Honest limits

> **These are estimates, not proof.** Perturbation-based attributions have limited and, in general, **unverifiable faithfulness**: they approximate what the model does, and that approximation cannot be independently confirmed for a given window.

Two design choices follow from that:

- **Raw attention maps are the least faithful class of explanation, and are deliberately not used.** The app relies on perturbation views instead.
- The estimate is **silent about anything the grade reads directly from the raw waveform**: attribution only covers the perturbations the method applies, not every path through the model.

Treat these views as one interpretability surface among several. The classical **SQI breakdown** in the [Segment Inspector](/biosqa/docs/segment-inspector/) and the **novelty / domain-shift indicators** in the Quality Inspector are complementary, read them alongside the XAI views rather than in place of them.
