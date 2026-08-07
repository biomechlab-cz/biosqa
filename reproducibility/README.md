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
  scripts/         run_experiment.py (entrypoint) · export_*.py · sync_from_src.py
  pyproject.toml · requirements.txt
```

The **eval harness is frozen**: every run calls `eval.protocols.evaluate(...)`; metrics are never
re-implemented in experiment code, so historical comparisons stay valid.

### `biosqa/` and `scripts/` are GENERATED — do not edit them here

Both trees are mechanically derived from the research monorepo by
**`scripts/sync_from_src.py`** (`--check` reports drift and exits 1; no argument rewrites):

| tree | source | transform |
|---|---|---|
| `biosqa/**.py` | `src/biosqa/<same path>` | none — **verbatim byte copy** |
| `scripts/*.py` | `<monorepo>/scripts/<same name>` | one line: the `sys.path` bootstrap points at `<root>` instead of `<root>/src`, plus a GENERATED banner |

Deliberate exceptions: `biosqa/utils/paths.py` (its repo root must resolve to *this* folder) and
`scripts/sync_from_src.py` itself (reproducibility-only). To change anything else, edit the
monorepo copy and re-run the sync — an edit made here is silently reverted by the next sync.

Hand-copying is what let this package rot twice: first `eval/protocols.py` fell behind and lost the
cluster-bootstrap CI helpers, then nine modules **and both export scripts** shipped pre-fix code
(an inflated artifact-type `pos_weight`, an un-masked `class_weights`) after the engine had already
been fixed — i.e. the package reproduced the *buggy* engine. `app/tests/test_reproducibility_sync.py`
now fails the app test suite whenever the snapshot drifts, so this cannot rot silently again.

---

## Environment

- Python **3.12**. GPU training was done on an RTX 5090 (Blackwell, sm_120) with a **CUDA 12.8** PyTorch
  wheel (`torch 2.11.0+cu128`); CPU-only reproduction works with the default-index torch.

```
python -m venv .venv && . .venv/bin/activate      # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
pip install -e .                                  # puts `biosqa` on the path
# GPU (Blackwell/sm_120): install torch from the CUDA 12.8 index instead of the default wheel:
#   pip install torch --index-url https://download.pytorch.org/whl/cu128
```

`pip install -e .` is what makes `import biosqa` work; `requirements.txt` only installs the
dependencies. (The scripts also insert this folder on `sys.path` themselves, so they run from a
bare checkout too.)

Reproducibility knobs: `utils.seed.seed_everything()` and a **config hash** logged per run.
`scripts/run_experiment.py` also logs to **MLflow** (`./mlruns`, SQLite backend) when it is
installed. Two honest caveats about the wider research engine this folder is drawn from: no
*data-manifest* hash is computed per run (corpus-level SHA-256 digests exist per built store, in
that store's `snapshot_manifest.json`, not per run), and most published campaign results come from
standalone experiment scripts that do **not** go through this entrypoint and are not in MLflow —
they are recorded as JSON result files plus a running research log.

---

## Reproduce

Config = `configs/base.yaml` deep-merged with `configs/experiment/<name>.yaml`, then CLI dotlist overrides.

```
# smoke — synthetic data, fully self-contained, no dataset or data-loader needed:
python scripts/run_experiment.py --experiment dummy_smoke

# a real run — requires the data-acquisition layer (see Data) plus a built store:
python scripts/run_experiment.py --experiment ecg_store --set train.lr=1e-4 seed=1

# export a trained model to ONNX + model card — REFERENCE CODE, see the caveat below:
python scripts/export_all_modalities.py          # or the per-modality export_*.py
```

The `dummy_smoke` run is self-contained (`data/synthetic.py`) — its data source is imported lazily, so
it runs against this package alone. The `store` / `cinc2011` sources need the omitted loader layer (below).

> **The `export_*.py` scripts are reference code, not runnable as shipped.** They import
> `biosqa.data.harmonize`, `biosqa.data.store` and `biosqa.xdomain` (calibration + conformal
> abstention) at module level, and none of those are part of this subset — the first two are the
> omitted data-acquisition layer, and `xdomain/` is outside the model+train+eval+export scope. They
> are included because they are the exact, byte-synced recipe that produced the shipped
> `<modality>.onnx` + `model_card.json` (architecture, loss weighting, temperature-scaling and
> conformal-threshold procedure, card fields), which is what a reader needs to audit. To execute
> them, take them from the full research monorepo.

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
>
> **Three of those modules are readable but not importable as shipped**, because they need the omitted
> `data/harmonize.py` (the Q0–Q3 mapping table) at import time: `data/windows.py`,
> `data/artifact_labels.py` and `data/artifact_synth.py`. The feature banks, augmentation and
> `data/synthetic.py` — everything `dummy_smoke` and the SQI/fusion inputs touch — import cleanly.
> `app/tests/test_reproducibility_sync.py` pins this list, so it cannot grow unnoticed.

**Datasets (each retains its own license/terms — acquire from the source and respect its access terms):**

| Modality | Datasets | Access |
|---|---|---|
| ECG | BUT QDB · European ST-T · MIT-BIH VFDB/SVDB/NSTDB · PTB-XL · PhysioNet/CinC-2011 | open (CC-BY / ODC-BY) |
| PPG | BUT PPG · PPG-DaLiA · WESAD (wrist) · MIMIC-III-Ext-PPG | open / research-use; **WESAD is non-commercial**, **MIMIC-III-Ext-PPG needs PhysioNet credentialed access** |
| EEG | Motion-Artifact fNIRS+EEG · PhysioMotion (ds006386) · Phantom-EEG (ds004784) · Mind-in-Motion · TUAR (TUH) | mostly CC0 / ODC-BY; **TUAR requires a Temple DUA** |
| EDA | EDABE · EDA-Artifact-Detection (UTD + AWW) · WESAD (wrist) | open / research-use |

> **Access terms matter.** Some sources are access-restricted (e.g. TUAR/TUH EEG needs a signed Temple data
> use agreement). Obtain each dataset under its own terms; this repo redistributes **no data**. A
> Q0–Q3 harmonization mapping per dataset (native label → ordinal grade) is applied in the (not-shipped)
> loader layer; the mapping rules are documented in the paper's supplement.

---

## License

Code: MIT (see the project root). **Datasets and trained weights are not covered by that grant** —
they retain the terms of their respective data sources. Two of those terms are restrictive and apply
to the weights shipped with the app: **WESAD** is research-use only (no commercial use, PPG + EDA),
and **MIMIC-III-Ext-PPG** and **TUAR/TUH EEG** require credentialed access / a signed data use
agreement (PPG and EEG respectively). Per-model provenance and inherited terms are in
[`../LICENSE-MODELS`](../LICENSE-MODELS). This is a provenance statement, not legal advice: check
each source's current terms before redistributing a model.
