---
title: "Opening recordings & formats"
description: "Supported formats, header-only opening, and why you should feed raw signals."
---

BioSQA Studio opens a recording by reading its header, detecting the modality, and scoring the samples against the matching ONNX model. This page covers what formats it reads, how opening stays instant regardless of length, and why the signal you feed it matters.

## Supported formats

The app reads the common biosignal container families directly:

| Family | Formats |
|---|---|
| **WFDB** | `.dat` / `.hea` |
| **MNE family** | EDF/EDF+, BDF/BDF+, GDF, BrainVision (`.vhdr`), EEGLAB (`.set`), FIF |
| **Columnar** | Parquet |

> **In progress.** Zarr and richer Parquet ingestion are being built out and are not yet complete. Treat them as partial until the docs say otherwise.

## Header-only opening

On open, only the **header** is read, not the sample data. A recording of **any length** therefore opens instantly, whether it is a few seconds or many hours. Samples are read **lazily** as you view and analyse, which is what lets the app stream multi-hour recordings out-of-core without ever holding the whole signal in memory.

## Auto-detect vs forcing a modality

By default the **modality is auto-detected**: see [Modality detection](/biosqa/docs/modality-detection/). Each modality has its own independent model, so the detected type determines which model scores the recording. All four types open, plot and export; grading requires a model in `models/`, and this release bundles **ECG and EDA** only, opening an EEG or PPG recording displays the signal and reports that no model is bundled for it. See [Models](/biosqa/docs/models/) for why, and for how to supply your own.

You can also **force a modality** from the **Open** menu (`ecg`, `eeg`, `ppg`, `eda`) to score a recording against a specific model, useful when you already know the signal type or want to test one model deliberately.

> When auto-detect returns a **low-confidence** result, the app surfaces a warning asking you to verify the signal type before trusting the grades. Force the correct modality if the guess is wrong.

## Recent recordings

Files you have opened are kept in a **recent recordings** list for quick re-open. Behind that, a bounded **LRU cache** of open file handles keeps a few recordings ready for fast access and closes the rest, so re-opening a recent file is cheap without holding every handle open.

## Feed raw signals

> **Feed raw signals.** Models score the signal **as provided**. A signal that has already been pre-filtered or cleaned but is still corrupt can be **under-flagged**: the filtering hides the artifact from the model without removing the underlying problem. For how the runtime treats what it is given, see [Runtime guards](/biosqa/docs/runtime-guards/).

## Troubleshooting a failed open

A malformed or unsupported **header** produces a **clear error** rather than a crash. The app declines to open the file and tells you why. If a recording you expect to be valid will not open, see [Troubleshooting](/biosqa/docs/troubleshooting/).
