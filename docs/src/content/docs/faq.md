---
title: "FAQ"
description: "Common questions: medical use, GPU/internet, formats, odd scores, your own model, window length, corrections, datasets."
---

Short answers to the questions that come up most often. Each links to
the page with the full detail where one exists.

## Is it a medical device?

No. BioSQA Studio is a **research and engineering tool**, not a
certified medical device. It is not validated or approved for
clinical or diagnostic use, and nothing it reports should drive
patient care.

## Do I need a GPU or internet?

No. Inference runs on the **CPU via ONNX Runtime**, fully local — no
GPU, no cloud, no network call. The only feature that reaches out is
the optional [LLM audit](/biosqa/docs/llm-audit/), and even that
talks to a **local Ollama server**, not a remote service.

## What formats can I open, and how long a recording?

The app reads **WFDB**, the **EDF / BDF / GDF / BrainVision / EEGLAB /
FIF** family, and **Parquet**. Files open **header-only**, so a
recording of any length opens instantly, and multi-hour records
**stream out-of-core** block by block — the whole signal is never held
in memory. See [Opening recordings](/biosqa/docs/opening-recordings/).

## Why did a clean, pre-filtered signal score oddly?

The models score **raw signals**. Pre-filtering strips the very cues
they key on, so a filtered-but-corrupt signal can be **under-flagged**
— it looks clean while the underlying artifact is gone from view.

> Feed raw signals. The **false-clean guard** exists precisely for
> this case. See [Runtime guards](/biosqa/docs/runtime-guards/).

## Can I use my own model, or a new modality?

Yes. Export an **ONNX model** plus a **conforming model card** and drop
both into `models/`. The app consumes those two files and nothing
else — it never trains or imports training code. See
[Models](/biosqa/docs/models/).

## Why can't I change the window length?

The window length is **fixed by the model card**. It is part of how
the model was trained, not a UI setting, so changing it would break
the correspondence between what the model saw during training and what
it scores now.

## What do corrections do?

They **never change the model**. A correction records an auditable
relabel or note and appends it to a **JSONL active-learning queue** —
a durable log for later retraining or review, not a live edit to
scores. See [Manual review](/biosqa/docs/manual-review/).

## Which datasets were used, and are they redistributable?

The models were trained on **public biosignal datasets under the
PhysioNet umbrella**. Some sources — for example **TUAR / TUH EEG** —
require **credentialed access** and are **not redistributed** with the
app. See [Contributing](/biosqa/docs/contributing/).
