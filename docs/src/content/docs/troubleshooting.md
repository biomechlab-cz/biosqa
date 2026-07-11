---
title: "Troubleshooting"
description: "Recording won't open, model fails to load, slow inference, nothing to export, LLM audit, rendering quirks."
---

Symptoms you may hit while running BioSQA Studio, each with its likely cause and the fix. Most are deliberate guards, not defects — the app prefers a clear error over a silently-wrong grade.

## A recording won't open

**Cause.** An unsupported or malformed file — typically a header the app cannot parse.

**Fix.** The app reports a clear error rather than crashing. Confirm the file is in a **supported format** (see [Opening recordings](/biosqa/docs/opening-recordings/)). For a generic numeric file with no self-describing header, the app cannot infer the signal type, so **force a modality** on open.

## The model fails to load

This is **by design**. Each modality has its own `models/<modality>.onnx` paired with a `models/<modality>.model_card.json`. At load time the app checks the card against the ONNX graph — modality, input shape, and feature-vector definition must agree.

> A missing or mismatched model card is a **hard error at load**. This is intentional: it guarantees you never receive grades from a model that doesn't match its declared contract.

**Fix.** Correct the card/model pairing in `models/` so the card describes the graph it sits beside. See [Models](/biosqa/docs/models/) for the contract the app enforces.

## Inference is slow

**Cause.** High window overlap and the recoverability pass both add per-window work. Very large records also stream block-wise, which is slower than an in-memory pass but keeps memory bounded.

**Fix.** In [Settings](/biosqa/docs/settings/), **lower the window overlap** and/or **disable the recoverability pass**. For multi-hour recordings, expect the streaming path to trade speed for bounded memory.

## There's nothing to export

**Cause.** Export is only meaningful after a recording has been analysed and segmented — there are no results to write yet.

**Fix.** **Run inference first**: open and analyse a recording so segmentation produces per-segment grades, then export. See [Exporting](/biosqa/docs/exporting/).

## The LLM audit doesn't respond

**Cause.** The audit calls a **local Ollama server** running the configured model. If Ollama is not running, unreachable, or the configured model is unavailable, the call cannot complete.

**Fix.** Start the local Ollama server with the configured model, then retry.

> The audit has a **bounded timeout** and degrades to an error if Ollama is unreachable. The model's own verdict is unaffected and still stands. See [LLM audit](/biosqa/docs/llm-audit/).

## Rendering / GPU quirks

**Cause.** Inference runs entirely on the **CPU**, but the front end uses the **GPU for drawing**. A driver or compositor hiccup can leave the window rendering oddly.

**Fix.** Try **toggling the theme** or **restarting the app**. Neither affects inference results.

## Where to file an issue

If a problem persists or looks like a defect, report it on the project's **GitHub Issues** tracker.
