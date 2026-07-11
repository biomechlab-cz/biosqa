---
title: "The workspace: signal view & navigation"
description: "The docked layout, reading the overlay, navigation, and out-of-core behavior."
---

The workspace is where you read a recording against its quality grades: the raw trace, the quality-band overlay, and the tools to move through hours of signal. This page covers the layout, how to read the overlay, navigation, and how the app stays responsive on multi-hour records.

## Layout

The workspace uses a **docked, resizable layout** with three regions:

- The **file/channel tree** on the left — the open recording and its channels.
- The **waveform plot canvas** in the center, carrying the trace and the **quality-band overlay**.
- The **AI Quality Inspector** on the right.

An **activity rail** switches between full-bleed views: **Workspace**, **Overview**, **Segmentation**, and **Segment inspector**. The dock boundaries are resizable, so you can trade space between the tree, the canvas, and the inspector as the task demands.

> The center canvas is the reference surface. The Overview, Segmentation, and Segment inspector views (reached from the activity rail) present the same recording at different granularities — see [Recording overview](/biosqa/docs/recording-overview/), [Segmentation](/biosqa/docs/segmentation/), and [Segment inspector](/biosqa/docs/segment-inspector/).

## Reading the overlay

Each **quality segment** is drawn as a translucent, **tier-colored band** over the trace. The band co-renders the **tier glyph** and the **mono code** (Q0..Q3) so the grade is legible without relying on color:

| Grade | Meaning | Glyph | Code |
|-------|---------|-------|------|
| Q0 | unacceptable (discard) | ⊘ | Q0 |
| Q1 | poor | ⚠ | Q1 |
| Q2 | acceptable | ✓ | Q2 |
| Q3 | excellent | ✓ | Q3 |

Band **opacity scales gently with confidence** — lower-confidence grades read fainter. **Q0 bands are hatched** in addition to their color and glyph, so the discard tier stands out even in a color-blind-safe palette. For the full grade definitions, see [Quality scale](/biosqa/docs/quality-scale/).

## Navigation

Three tools move you through the trace:

- **Pan** — drag the canvas along the time axis.
- **Zoom** — drag-box a region, or use the mouse wheel.
- **Measure** — read a **Δ-time** between two points.

A **minimap navigator** gives an hours-scale overview strip carrying a **quality ribbon** and a **draggable viewport** box. Move the viewport to jump the main canvas to any point in the recording.

## Multi-channel lanes

Toggle channels on as **stacked, per-channel-colored lanes**. A header reports **visible / total** channel counts, so you always know how much of the recording is currently drawn versus available.

## Hover tooltip

Hovering the trace shows a tooltip with the **timestamp**, the **sample value**, and the **tier + confidence** at the cursor, accompanied by a **playhead line** marking the cursor position.

> The tier lookup at the cursor is **O(log N)** over the sorted segments, so the tooltip stays responsive even on long recordings with many segments.

## Click-to-select, synced everywhere

Selecting a segment in **any** view — the plot, the minimap, the run-length track, the table, or the inspector — **highlights it everywhere** and can **center the view** on it. Selection is a single shared state across the app, so you can pick a segment where it is easiest to see and read it where it is most useful.

## Out-of-core behavior

Multi-hour records are analysed **block-wise**, and the plot draws from a **decimated cache** rather than the full samples. The **whole signal is never held in memory**, which is what lets the workspace open recordings that would not fit at full resolution.

> In this streamed mode, **some advisory guards are disabled**. See [Runtime guards](/biosqa/docs/runtime-guards/) for which guards run and when.
