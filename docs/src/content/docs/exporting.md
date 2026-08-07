---
title: "Exporting results"
description: "What is exported and every format: CSV, TSV (BIDS), JSON, Parquet, WFDB annotation, and MATLAB .mat."
---

Exporting turns an analysed recording into a file you can carry into your own pipeline, share, or archive. What leaves the app is the quality assessment. The interval structure, your edits, and the provenance to trace where each grade came from.

## What is exported

An export captures the **quality intervals** the model produced, together with anything you added on top of them:

- **Your corrections**: any relabels applied during [manual review](/biosqa/docs/manual-review/).
- **Notes and flags** attached to intervals.
- **Model provenance**: which model and which version scored the signal, read from the model card.

Run inference first. There is nothing to export until a recording has been analysed. The interval list, and therefore the export, does not exist until the model has run.

## Effective vs model tier

Every interval carries **two** grades, not one:

- the **effective tier**: the grade in force after any relabel you applied;
- the original **model_tier**: the grade the model assigned before you touched it.

> Both are written to every export. A relabel is therefore fully auditable: a downstream reader can see what the model said, what you changed it to, and reconstruct the difference. Nothing is overwritten silently.

## Formats

The same interval data can be written in several forms. Pick the one that matches where the results are going.

| Format | Extension | Shape / use |
|---|---|---|
| **CSV** | `.csv` | Flat interval schema. One row per interval (`start_sec`, `end_sec`, `tier`, …). |
| **TSV** | `_events.tsv` | BIDS events file, `onset` / `duration` / `trial_type`. |
| **JSON** | `.json` | Richest, schema-versioned form; round-trips back into the training / active-learning channel. |
| **Parquet** | `.parquet` | Columnar, for analysis pipelines. |
| **WFDB annotation** | `.qual` | Sample-indexed annotation; the tier is carried in the annotation **subtype**. |
| **MATLAB** | `.mat` | MATLAB workspace file. |

A few notes on the less obvious ones:

- **JSON** is the only form that round-trips. Because it is schema-versioned and carries the full interval record, an export can be read back into the training / active-learning channel, corrected labels become data.
- **BIDS TSV** maps the interval onto the neuroimaging events convention: `onset` (interval start), `duration` (its length), and `trial_type` (the tier). It is the right choice when the recording already lives in a BIDS dataset.
- **WFDB annotation** stays in the signal's own sample index rather than seconds, so the `.qual` file lines up directly against the waveform in WFDB tooling. The quality tier rides in the annotation subtype field.

## Export the filtered selection

You do not have to export the whole recording. If you have narrowed the segment list, by tier or by any other filter in the [segment view](/biosqa/docs/segmentation/), you can export **just the currently-filtered segments**. The export then contains exactly the intervals you are looking at, in whichever format you choose above.
