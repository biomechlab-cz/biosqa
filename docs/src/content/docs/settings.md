---
title: "Settings reference"
description: "Appearance, analysis, integrity guard, LLM audit, and persistence."
---

Settings control how BioSQA Studio looks, how finely it segments a recording, and which optional checks run. Every setting described here is persisted with **QSettings** and remembered across runs.

## Appearance

Appearance settings are cosmetic and never change the underlying grades or scores.

| Setting | Options |
| --- | --- |
| Theme | Light / Dark |
| Accent color | Teal / Blue / Purple / Amber |
| Color-blind-safe tiers | Off / On (blue↔orange palette) |

The **color-blind-safe tiers** toggle swaps the quality-tier palette to a blue↔orange scheme. Grades are never conveyed by color alone — each tier also carries a glyph and a mono code (Q0..Q3), and Q0 is additionally hatched. See the [quality scale](/biosqa/docs/quality-scale/) for the full encoding.

## Analysis

These settings govern how the open recording is segmented. **Changing any of them re-segments the current recording live** — you do not need to close and re-open the file.

| Setting | Options | Default |
| --- | --- | --- |
| Window overlap | 0% / 25% / 50% / 75% / 90% | 50% |
| Recoverability pass | Off / On | — |
| Boundary refinement | Off / On | — |

**Window overlap** trades localization against speed. Higher overlap places windows closer together, so quality transitions are located more finely, but more windows means more inference work per recording. Lower overlap is faster and coarser.

The **recoverability pass** is a second analysis pass; **boundary refinement** sharpens the start/end of graded segments. Both can be turned off to reduce work. See [segmentation](/biosqa/docs/segmentation/) for how these shape the segment list.

## Integrity guard

The integrity guard is the **false-clean guard** — it flags signals that look clean but may be corrupt or pre-filtered.

| Setting | Options |
| --- | --- |
| False-clean guard | Enable / Disable |
| bSQI threshold | 0.5 – 0.9 |

The **bSQI threshold** sets how aggressively the guard fires. Details of what the guard checks and how it surfaces its findings are in [runtime guards](/biosqa/docs/runtime-guards/).

> Feed BioSQA Studio **raw** signals. A signal that has already been filtered but is still corrupt can be under-flagged; the integrity guard exists to catch some of these cases, but it is not a substitute for supplying unprocessed input.

## LLM audit

The optional LLM audit sends selected material to a local **Ollama** instance for a natural-language second opinion. It is off unless enabled here.

| Setting | Options |
| --- | --- |
| LLM audit | Enable / Disable |
| Ollama host | Host address |
| Model name | Ollama model identifier |
| Self-consistency samples | 1 – 5 |

**Self-consistency samples** controls how many independent responses are gathered and reconciled — more samples give a more stable verdict at higher cost. See [LLM audit](/biosqa/docs/llm-audit/) for the full workflow.

## Remembered UI

Beyond the settings above, the app remembers your **Table / Grid** segmentation view choice, so the recording opens in the layout you last used.

## Speed vs resolution

A short orientation for tuning throughput on long recordings:

- **Lower the window overlap** and **disable the recoverability pass** — these are the main speed levers.
- **Raise the window overlap** to improve boundary resolution, at the cost of more inference per recording.

Because analysis runs block-wise out-of-core, these levers matter most on multi-hour recordings, where the number of windows dominates processing time.
