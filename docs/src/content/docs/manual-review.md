---
title: "Manual review: relabel, notes & flag-for-review"
description: "The researcher decides. The MODEL SAID vs YOU SET trail, corrections, notes, and the training queue."
---

BioSQA Studio grades quality automatically, but the grade is a proposal, not a verdict. Manual review is where you accept, override, annotate, and queue segments, and where every decision is recorded so it stays auditable.

## The researcher decides

The model proposes; you dispose. When you disagree with an automatic grade, you set your own, but the correction never erases what the model said. Every override **preserves the model's original grade** alongside yours, so the trail of who decided what is intact after the fact.

> This is a research and engineering tool, not a medical device. The automatic grade is an estimate on a raw signal; your judgement is the final word on a segment, and the app keeps both on the record.

## MODEL SAID vs YOU SET

Each reviewed segment carries a two-part verdict trail:

- **MODEL SAID**: the tier the model assigned automatically.
- **YOU SET**: the tier you assigned by hand, if you overrode it.

Both travel through to export. An exported segment carries an **effective tier** (the grade in force, yours if you set one, otherwise the model's) *and* the **`model_tier`** (the model's original grade). Because both are present, any relabel is fully auditable: a reader can always recover what the model said and what you changed it to.

## Relabel or accept

You have two responses to an automatic grade:

- **Accept**: keep the AI tier as the effective tier.
- **Override**: set the effective tier yourself to **Q3**, **Q2**, **Q1**, or **Q0** (see the [quality scale](/biosqa/docs/quality-scale/)).

An override here is a **soft relabel**: it changes the tier assigned to a segment, but it does not move the segment's boundaries. Reshaping where a segment begins and ends is a distinct, harder edit. The [segment inspector](/biosqa/docs/segment-inspector/) handles that.

## Review note and flag for review

Beyond the tier, you can attach two things to a segment:

- a free-text **review note**: context, rationale, an observation about the artifact;
- a **review flag**: a marker that the segment needs another look.

Either or both can be attached, and both **persist with the correction**: they are part of the recorded decision, not a transient UI state.

## Save correction, the training queue

Saving a correction appends it to a durable **JSONL** log. An active-learning **reverse channel** back toward model training. This is a queue you accumulate over a review session and can later feed into a training pipeline.

```
one JSON object per line, appended, never rewritten
```

> The app itself never trains and never imports training code. The reverse channel is a durable record you can hand to the ML side; BioSQA Studio only writes it.

## What exports carry

Corrections do not stay locked inside the app. When you [export](/biosqa/docs/exporting/), the following travel with the data:

- your **corrections** (the effective tier);
- the **`model_tier`**: the model's original grade;
- any **notes** and **review flags**;
- **model provenance**: which model produced the automatic grades.

Because the effective tier and the model tier are both present, together with notes, flags, and provenance, an exported record is self-describing: a downstream reader can reconstruct both what the model said and what you set.
