---
title: "Quality segmentation (Table & Grid)"
description: "The run-length timeline, table and grid views, filters, and jump-to-poor."
---

After inference, BioSQA Studio turns the recording's per-window grades into a compact, navigable index of where quality holds and where it breaks down. This page covers the two views of that index, Table and Grid, plus the filters and navigation that make a multi-hour recording tractable.

## A table of contents by quality

Inference produces an ordinal grade (Q0–Q3) for each analysis window. Rather than leave you scrolling through thousands of windows, the app **collapses contiguous windows of the same grade into run-length segments**: one row per span of consistent quality. The segmentation view is that collapsed structure: a **table of contents for the recording, organized by quality**.

Each segment carries a time range and its tier, so you can read a whole session's quality profile at a glance and navigate straight to the parts that matter. Grades are ordinal, Q0 unacceptable (⊘), Q1 poor (⚠), Q2 acceptable (✓), Q3 excellent (✓), with **"usable" meaning Q2+Q3**. See [Quality scale](/biosqa/docs/quality-scale/) for the full grade definitions.

## Table view

The **Table view** lists segments as dense rows, one per run-length segment:

| Column | Meaning |
|---|---|
| **Time range** | Start–end of the contiguous segment |
| **Tier** | The segment's quality grade, shown as a badge (color + glyph + mono code) |
| **Confidence** | The model's confidence for the segment |
| **Artifact tags** | Artifact types associated with the segment |
| **Recoverable** | A ↺ marker on segments flagged as recoverable |

Numeric fields are set in **monospace** so columns align and scan cleanly. The layout favors density. You can survey many segments without scrolling.

> Tier is never conveyed by color alone. Each badge combines color, glyph, and the mono `Q0`..`Q3` code, and a color-blind-safe palette toggle is available. See [Quality scale](/biosqa/docs/quality-scale/).

## Grid view

The **Grid view** presents the same segments as clickable **card thumbnails**. Each card renders a **mini-waveform drawn from the segment's real samples**, alongside the same tier badge, confidence, and artifact tags as the table. This makes it easy to visually recognize the morphology of a dropout, a saturation, or a clean segment without opening each one.

Clicking a card selects that segment. The **Table/Grid choice is remembered across runs**, so the app reopens in whichever view you last used.

## Filter pills

A row of **filter pills** narrows the segment list to a quality band:

| Pill | Shows |
|---|---|
| **All** | Every segment |
| **Only Q0** | Unacceptable segments only |
| **Poor** | Q0 + Q1 |
| **Usable** | Q2 + Q3 |
| **Recoverable** | Segments flagged recoverable (↺) |

Filters apply to both the Table and Grid views, so you can, for instance, page through only the Poor segments as cards.

## Jump to next poor region

For long recordings, **Jump to next poor region** steps directly to the next Q0/Q1 segment, **selecting it and centering it** rather than making you hunt for it. Repeated use walks you through every poor region in order. The selection is shared with the rest of the app; see the [Workspace](/biosqa/docs/workspace/) for how selection and centering behave across views.

## Export the selection

Whatever the filter pills are currently showing defines the working set, and you can **export just the currently-filtered segments**: for example, all Usable segments, or only the recoverable ones. See [Exporting](/biosqa/docs/exporting/) for the available formats and what each export contains.

> This is a research and engineering tool, not a medical device. Feed raw signals: a pre-filtered but corrupt signal can be under-flagged, so segments marked usable still warrant inspection.
