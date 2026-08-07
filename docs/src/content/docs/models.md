---
title: "Models & the model-card contract"
description: "The two-artifact contract per modality, card-driven preprocessing, and bring-your-own-model."
---

Every score BioSQA Studio produces comes from a **trained ONNX model** paired with a **model card** that tells the app exactly how to feed it. The card is not optional metadata. It drives preprocessing, defines the grade scale, and is validated at load so you always know precisely what scored your signal.

## Two artifacts per modality

Each modality is defined by two files in `models/`:

```
models/<modality>.onnx              # float32[1, 1, L] input
models/<modality>.model_card.json   # the contract
```

**Two models ship with the app, `ecg` and `eda`**, each its own independent pair. The ONNX graph takes a single raw window shaped `float32[1, 1, L]`; the card supplies everything the app needs to build that window and interpret the output.

All four signal types are fully supported for loading, plotting and export, but **`eeg` and `ppg` weights are not bundled**. Those models were trained on cohorts with credentialed access (MIMIC-III-Ext-PPG), non-commercial terms (WESAD) or a signed data use agreement (TUAR / TUH EEG), none of which an openly-licensed weight release can honour. Opening an EEG or PPG recording shows the signal and reports that no model is bundled for that modality. Drop in your own conforming `<modality>.onnx` + `<modality>.model_card.json` and it works with no code change, see [Bring your own model](#bring-your-own-model). The per-dataset reasoning is in [`LICENSE-MODELS`](https://github.com/sokolmarek/biosqa/blob/main/LICENSE-MODELS).

## What the card declares

The card is the single source of truth for how a model is run and read. At minimum it declares:

| Field | Meaning |
|---|---|
| `fs` | Sample rate the model expects |
| `L` | Window length (samples) |
| `class_order` | Grades Q0..Q3, ordered **worst→best** |
| `normalization` | How the raw window is normalized |
| model version | Identifies the exported model |

Cards may also carry optional blocks that richer models rely on:

- **calibration temperatures**: for probability calibration.
- a **conformal (APS)** block, for prediction sets.
- a **novelty (Mahalanobis)** reference, for out-of-distribution flagging.
- a **feature-attribution** reference, used by fusion models.
- **provenance & licence**: `training_data_provenance` (training store, cohorts, corpus digest where one was recorded) and `license` (the terms the weights inherit from their training data). Documentation only: the app ignores both. See [Contributing & license](/biosqa/docs/contributing/).

The grade scale itself is covered on [Quality scale](/biosqa/docs/quality-scale/); the card's `class_order` is what binds a model's output heads to Q0..Q3.

## Card-driven preprocessing (never hard-coded)

Normalization, window length, and any second model input all come from the card, none of it is baked into the app. This is what lets different models coexist without code changes. Models fall into three input shapes:

| Type | Inputs | Example |
|---|---|---|
| **Single-input** | raw window only | none shipped |
| **Dual-branch** | raw + spectral channels | ECG |
| **Fusion** | raw + hand-crafted SQI + dynamics feature vector | PPG, EEG, EDA |

For dual-branch and fusion models, the host computes the **second input exactly as the card specifies**: the spectral channels or the feature vector are constructed to the card's declaration, not to a hard-coded recipe.

> **EEG grade note.** EEG signal-quality grading is inherently the hardest of the four modalities, and it is also the one whose expert-annotated cohort (TUAR) carries a data use agreement, which is why no EEG weights are bundled here. If you supply your own, treat the EEG grade as advisory and lean on the usable gate and the artifact chips.

> Feed **raw** signals. Preprocessing is the model's job, described by its card. A pre-filtered-but-corrupt signal can be under-flagged because the artifacts the model is trained to catch have already been partly smoothed away.

## Fail-loud contract

At load time the app validates the pair and refuses anything inconsistent:

- input **shape vs `L`**: the graph's input must match the declared window length.
- each **head's output width vs its `class_order` length**.
- the **card modality vs the filename**.
- for fusion models, the **feature-vector names and width vs the card**.

> A missing or mismatched card is a **hard error at load, never a silent fallback**. This is a feature, not a limitation: the app will not guess, so you always know exactly which model and which preprocessing scored your signal.

## Bring your own model

Because everything is card-driven, you can score a new dataset or an entirely new modality without touching app code. Export your own **ONNX + a conforming card** and drop both into `models/`.

Two things to keep in mind:

- **Window length is fixed by the card, not the UI.** Whatever `L` you declare is what the app windows to.
- Your card must pass the same fail-loud validation above, shapes, head widths, modality, and (for fusion) feature names and width all have to line up.

For app configuration and options, see [Settings](/biosqa/docs/settings/); to score a recording end-to-end, see [Getting started](/biosqa/docs/getting-started/).
