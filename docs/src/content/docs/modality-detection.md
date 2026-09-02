---
title: "Automatic modality detection"
description: "How ECG/PPG/EEG/EDA is inferred, and how to correct it."
---

BioSQA Studio grades signals with a **per-modality model**: ECG, PPG, EEG and EDA each have their own independent ONNX model and model card. Before it can grade a recording it has to decide which modality you opened, so on open it runs a lightweight detector and auto-loads the matching model. This release bundles weights for **ECG and EDA**; detection still works for all four, and an EEG or PPG recording opens and plots normally but reports that no model is bundled for it (see [Models](/biosqa/docs/models/)).

## How it works

Detection is a **vote** over three pieces of header evidence:

| Evidence | What it is |
| --- | --- |
| Channel **units** | the physical units declared per channel |
| Channel **names** | the channel labels in the header |
| **Sample rate** | the acquisition rate |

The three signals are combined into a single modality decision (ECG/PPG/EEG/EDA), and the app loads the corresponding `models/<modality>.onnx` + `models/<modality>.model_card.json` and runs inference. See [Models](/biosqa/docs/models/) for what each model consumes.

## Confidence and low-confidence guessing

The vote returns a **confidence** alongside the modality. When units and names contribute, confidence is higher. When those provide nothing and only the **sample-rate tie-break** fired, confidence is low (around **0.35**) and the app treats the result as a *guess*.

> In that case the app **warns you to verify** rather than silently grading with an arbitrary model. A sample rate alone is weak evidence, several modalities overlap in rate, so the honest move is to flag it, not to commit to a grade you might have to walk back.

## Background verification

You can [force a modality](/biosqa/docs/opening-recordings/) when you know what the file is. Even then, the detector **still runs in the background**. If the header **confidently disagrees** with your forced choice, you get a **mismatch warning**: a guard against opening an ECG file but forcing EDA, which would otherwise grade the whole recording with the wrong model without complaint.

## Correcting the modality after opening

Detection is not a one-shot commitment. You can **re-tag** an already-open recording's modality, from the Open menu or the post-open correction control. Re-tagging **reloads the matching model and re-runs inference live**, so the grades update in place without reopening the file.

## Ambiguous cases

Generic numeric formats, for example a plain **Parquet of numbers**, carry **no units and no channel names** to vote on. That leaves detection uncertain.

> For these files, **force the modality** yourself. The header simply doesn't contain enough to decide, and forcing removes the ambiguity while still keeping background verification active.

Remember that models expect **raw** signals, see [Opening recordings](/biosqa/docs/opening-recordings/). A pre-filtered but corrupt signal can be under-flagged regardless of how the modality was detected.
