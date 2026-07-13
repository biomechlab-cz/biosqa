---
title: "Install & run"
description: "Requirements, installing from source, first run with the bundled sample data, and the five workspaces at a glance."
---

BioSQA Studio is a desktop app: you install it from source into a Python virtual environment and launch a Qt window. Inference runs on the CPU via ONNX Runtime — **no GPU, no cloud, and no training dependencies** are required.

## Requirements

- **Python ≥ 3.10** (3.12 recommended).
- A desktop OS with a windowing system. The standalone **frozen build is validated on Windows**; macOS and Linux are supported **from source** (their release builds are not yet proven).
- **No GPU** and **no internet** are needed for inference. The optional [LLM audit](/biosqa/docs/llm-audit/) needs a local Ollama server, but nothing else does.

The app deliberately has a small dependency set — PySide6 (Qt 6), ONNX Runtime, NumPy/SciPy, `wfdb`, and MNE-Python — and never imports any model-training code.

## Install from source

```
cd app
python -m venv .venv
# activate the venv:
#   Windows:  .venv\Scripts\activate
#   macOS/Linux:  source .venv/bin/activate
pip install -e .
```

## Run

```
biosqa-studio
# or, equivalently:
python -m biosqa.main
```

No recording of your own is required — the repository ships a `dummy_data/` folder with short synthetic ECG/PPG/EEG/EDA records that exercise the whole pipeline.

> **Feed raw signals.** The models assess the signal **as provided**. A pre-filtered-but-still-corrupt signal can be *under*-flagged, because filtering strips the high-frequency cues the model keys on. Open raw recordings whenever you can — the [runtime guards](/biosqa/docs/runtime-guards/) exist precisely to catch this case.

## The five workspaces at a glance

The activity rail on the left switches between full-bleed views:

- **Workspace** — the docked layout: file/channel tree, the waveform canvas with the quality overlay, and the AI Quality Inspector.
- **Overview** — a KPI dashboard summarising the whole recording (usable share, tier fractions, artifact bars, the model card).
- **Segmentation** — every quality segment as a filterable table or grid.
- **Segment inspector** — a single-segment deep dive with a zoomed waveform, the classical SQI breakdown, and on-demand explainability.

## A five-minute first pass

1. **Open** `dummy_data/test_ecg_3min.hea` (or drop in your own recording). It opens instantly — only the header is read.
2. The modality is **detected** and the matching model loads automatically.
3. Inference runs and the trace is painted with **Q0–Q3 quality bands**.
4. Click **Jump to next poor region** to step through the artifact-corrupted segments, or select any segment to inspect it.
5. **Export** the segments (CSV/TSV/JSON/Parquet/WFDB/.mat) once you're happy.

## Frozen build

A standalone Windows build (no Python install needed) is produced with PyInstaller and validated in CI. macOS and Linux frozen builds are a work in progress; on those platforms, run from source.
