# Model architectures

All four models share the **adapter → backbone → heads** skeleton (`biosqa/models/model.py`), differing in
sampling rate, window length, second input, and which heads are present. Grade is always the ordinal
Q0–Q3 head; "usable" is the binary Q2+Q3 head; "artifact" is a multilabel artifact-type head.

Shared trunk (defaults, `configs/base.yaml`):
- **Adapter** (`models/adapters.py`) — patchify the raw window: `patch_len`/`stride` per modality → ~32–64
  tokens; linear patch embedding to `d_model`.
- **Backbone** (`models/backbones/patchtst.py`) — PatchTST: `d_model=128`, depth 3, 8 attention heads,
  dropout 0.1. (`cnn1d.py` and the research `mamba_backbone.py` are drop-in alternatives.)
- **Pooling** (`models/pooling.py`) → **Heads** (`models/heads.py`).
- **Fusion / spectral second input** (`models/fusion.py`, `models/frontends.py`): the raw-signal trunk is
  combined with a per-modality second input before the heads.
- Optional **MixStyle** (`models/mixstyle.py`) for cross-cohort domain generalization.

| Model | fs (Hz) | Window L | Second input | Heads | Grade decoder | Version tag |
|---|---|---|---|---|---|---|
| **ECG** | 250 | 2500 (10 s) | **spectral channels** (`x_spec`) — dual-branch | grade · usable · artifact | ordinal | `v3-dualbranch-spectral` |
| **PPG** | 64 | 640 (10 s) | **SQI + dynamics vector** (`x_feat`) — fusion | grade · usable | ordinal ordered-logit | `v4-4cohort-dalia-ordlogit` |
| **EEG** | 256 | 1280 (5 s) | **SQI + dynamics vector** (`x_feat`) — fusion | grade · usable · artifact | ordinal ordered-logit | `v4-multihead-grade-usable-type` |
| **EDA** | 8 | 480 (60 s) | **SQI + dynamics vector** (`x_feat`) — fusion | grade · usable | ordinal ordered-logit | `v3-combined-fusion-ordlogit` |

## Routing (per model card)

- **ECG** — `grade <- raw branch; usable + artifact-type <- raw + spectral fused`. The spectral second
  input is precomputed band-power channels (host-side at inference; see the app's `spectral.py`).
- **PPG / EEG / EDA** — `grade + usable (+ type for EEG) <- raw trunk FUSED with the host "combined
  SQI + dynamics" vector`, grade decoded as an **ordered logit**. The fused vector is `combined_vector(...)`
  from `data/sqa_features.py` (per-modality interpretable SQIs + advanced dynamics: spectral kurtosis,
  recurrence-quantification, ordinal-pattern, dispersion-entropy, …).

## Export

Each model is exported to a **static per-modality ONNX graph** (`export/to_onnx.py`): the modality
embedding is folded in as a constant, so the graph is a clean `adapter → backbone → head(s)` chain plus the
second input. Export uses the **legacy exporter** (`dynamo=False`) for correct dynamic-batch `Reshape`;
FP32 is the source of truth, with an optional INT8 dynamic-quantization pass. Parity vs the PyTorch model is
checked at export. The exported artifact ships with a `model_card.json` declaring `fs_hz`, `l_m`,
`class_order` (Q0..Q3), normalization, the heads, and (fusion models) the fused-feature names + a
temperature-calibration block — the contract the app validates at load.
