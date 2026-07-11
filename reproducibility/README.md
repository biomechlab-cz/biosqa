# BioSQA — ML experiment reproducibility

Reference code for the neural **signal-quality-assessment (SQA)** models behind BioSQA Studio: the model
architectures, the training loop, the evaluation harness, and the ONNX export. This folder is the
paper-facing subset of the research engine — enough to read and reproduce the ML experiments. The desktop
app (a separate, torch-free package) only *consumes* the exported `<modality>.onnx` + `model_card.json`.

> **Scope.** This is the model + training + eval + export code. It does **not** ship any dataset, and it
> omits the dataset-specific raw-format loaders (the data is acquired separately, and several sources are
> access-restricted). The Q0–Q3 harmonization mapping and the store schema are documented in
> [§ Data](#data) so the pipeline is reproducible once you have the data.

---

## What the models are

Four per-modality quality models — **ECG, PPG, EEG, EDA** — each scoring a fixed-length window on the
ordinal scale **Q0 (unacceptable) · Q1 (poor) · Q2 (acceptable) · Q3 (excellent)**. A shared design with
per-modality specialisation:

- **Adapter** — a per-modality patchifier (`models/adapters.py`): the raw 1-D window is split into
  overlapping patches (`patch_len`/`stride` absorb the sampling rate → ~32–64 tokens/window).
- **Backbone** — a shared trunk (`models/backbones/`): **PatchTST** (patch transformer, default) or
  **CNN1D**; a pure-PyTorch **Mamba** selective-scan is included as a research arm.
- **Heads** (`models/heads.py`) — an **ordinal** grade head (Q0–Q3, ordered-logit / CORN option), a
  **binary** usable head, and (ECG/EEG) a **multilabel** artifact-type head.
- **Second input** (`models/fusion.py`, `models/frontends.py`) — ECG fuses precomputed **spectral
  channels**; PPG/EEG/EDA fuse a hand-crafted **SQI + dynamics feature vector** (`data/sqa_features.py`).
- **Domain generalization** — optional **MixStyle** (`models/mixstyle.py`) for cross-cohort robustness.

Per-model architecture specifics (input shapes, heads, fusion, rates) are in **[ARCHITECTURES.md](ARCHITECTURES.md)**.

---

## Repo layout

```
reproducibility/
  biosqa/
    models/        adapters · backbones/{cnn1d,patchtst,mamba} · frontends · fusion · heads · mixstyle · pooling · model
    train/         loop (train/val, AMP, early-stop) · losses (CE/focal/CORN/ordinal + DG) · multitask
    eval/          metrics (macro-F1, κ, QWK, AUROC) · protocols (splits, LODO) · segment  [FROZEN harness]
    export/        to_onnx (legacy exporter, parity check, INT8)
    data/          windows · sqa_features · sqi · augment · synthetic · feature banks (ecg_sqi, nonlinear, …)
    utils/         config · seed · paths
  configs/         base.yaml + experiment/*.yaml
  scripts/         run_experiment.py (entrypoint) · export_*.py
  requirements.txt
```

The **eval harness is frozen**: every run calls `eval.protocols.evaluate(...)`; metrics are never
re-implemented in experiment code, so historical comparisons stay valid.

---

## Environment

- Python **3.12**. GPU training was done on an RTX 5090 (Blackwell, sm_120) with a **CUDA 12.8** PyTorch
  wheel (`torch 2.11.0+cu128`); CPU-only reproduction works with the default-index torch.

```
python -m venv .venv && . .venv/bin/activate      # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
# GPU (Blackwell/sm_120): install torch from the CUDA 12.8 index instead of the default wheel:
#   pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Reproducibility knobs: `utils.seed.seed_everything()`, a config hash + data-manifest hash logged per run,
and every run tracked to **MLflow** (`./mlruns`, SQLite backend).

---

## Reproduce

Config = `configs/base.yaml` deep-merged with `configs/experiment/<name>.yaml`, then CLI dotlist overrides.

```
# smoke — synthetic data, fully self-contained, no dataset or data-loader needed:
python scripts/run_experiment.py --experiment dummy_smoke

# a real run — requires the data-acquisition layer (see Data) plus a built store:
python scripts/run_experiment.py --experiment ecg_store --set train.lr=1e-4 seed=1

# export a trained model to ONNX + model card:
python scripts/export_all_modalities.py          # or the per-modality export_*.py
```

The `dummy_smoke` run is self-contained (`data/synthetic.py`); the `store` / `cinc2011` data sources need
the omitted loader layer (below).

The training recipe (defaults in `configs/base.yaml`): PatchTST `d_model=128`, depth 3, 8 heads; 30 epochs,
AdamW `lr=1e-3` / `weight_decay=1e-2`, 5 % warmup, batch 256, AMP, early-stop on macro-F1. Report
**macro-F1 + Cohen's κ** (nominal) **and quadratic-weighted κ** (ordinal cost).

---

## Data

The models are trained on public biosignal datasets (per-modality below), harmonized to the shared Q0–Q3
scale and windowed into a unified **Zarr store** (per-modality `[N, 1, L_m]` arrays indexed by `zarr_row`,
plus a `segments.parquet` of `modality, dataset, subject_id, label_harmonized, split`).

> **The data-acquisition layer is omitted here** — the per-dataset raw-format loaders, the Q0–Q3
> harmonization, and the store builder/registry (`loaders.py`, `harmonize.py`, `store.py`) handle the raw
> and access-restricted datasets and are not part of this reference release. What IS included is the
> modality-agnostic pipeline the models actually consume: windowing (`data/windows.py`), the SQI + dynamics
> feature banks (`data/sqa_features.py`, `sqi.py`, `ecg_sqi.py`, `nonlinear_features.py`), augmentation,
> and the synthetic data source that drives `dummy_smoke`. To reproduce on real data, provide a store in
> the schema above (or wire your own loaders) and point `data.store_dir` at it.

**Datasets (each retains its own license/terms — acquire from the source and respect its access terms):**

| Modality | Datasets | Access |
|---|---|---|
| ECG | BUT QDB · European ST-T · MIT-BIH VFDB/SVDB/NSTDB · PTB-XL · PhysioNet/CinC-2011 | open (CC-BY / ODC-BY) |
| PPG | BUT PPG · WESAD (wrist) · PPG-DaLiA | open / research-use |
| EEG | Motion-Artifact fNIRS+EEG · PhysioMotion (ds006386) · Phantom-EEG (ds004784) · Mind-in-Motion · TUAR (TUH) | mostly CC0 / ODC-BY; **TUAR requires a Temple DUA** |
| EDA | EDABE · EDA-Artifact-Detection (UTD + AWW) · WESAD (wrist) | open / research-use |

> **Access terms matter.** Some sources are access-restricted (e.g. TUAR/TUH EEG needs a signed Temple data
> use agreement). Obtain each dataset under its own terms; this repo redistributes **no data**. A
> Q0–Q3 harmonization mapping per dataset (native label → ordinal grade) is applied in the (not-shipped)
> loader layer; the mapping rules are documented in the paper's supplement.

---

## License

Code: MIT (see the project root). Datasets and any trained weights retain the terms of their respective
data sources — see the access column above before redistributing a model.
