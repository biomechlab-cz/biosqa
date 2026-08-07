# models/ (Plan 1 artifacts — not produced here)

This folder is a **drop-in target**, not a build output of this app. It is
populated by **Plan 1** (`plans/01_ML_AI_ENGINEERING_PLAN.md`, the
`biosqa` training package at the repo root) and consumed read-only by
`biosqa.model.model_card` / `biosqa.inference.onnx_runner`.

Do **not** add training code, checkpoints, or `torch` state dicts here —
the app's dependency set is deliberately disjoint from Plan 1's training
stack (see `app/pyproject.toml`).

## Expected layout

```
models/
  ecg.onnx
  ecg.model_card.json
  eda.onnx
  eda.model_card.json
```

`eeg.*` and `ppg.*` follow the same contract and the app supports both signal
types, but their weights are **not shipped here** — see
[Licensing of the shipped weights](#licensing-of-the-shipped-weights). Drop your
own pair in and the modality works with no code change.

## `model_card.json` schema (Plan 2 §11 handshake)

Minimal (legacy single-head) card:

```json
{
  "modality": "ecg",
  "L_m": 2048,
  "fs_hz": 250,
  "class_order": ["Q0", "Q1", "Q2", "Q3"],
  "normalization": { "method": "zscore", "mean": 0.0, "std": 1.0 },
  "training_data_hash": "sha256:...",
  "model_version": "0.1.0"
}
```

`biosqa/model/model_card.py` validates this shape on load and refuses
to run inference (loud failure, not a silent fallback) if a field is missing or
the ONNX model's declared input shape doesn't match `[batch, 1, L_m]` (the batch
dim is a dynamic axis, so it may be `1` or symbolic; channel and `L_m` are
checked strictly).

### v2 multi-head cards (Plan 1 §12.1)

A card MAY additionally declare a `heads` array so the app can surface the three
levels of output the model exports. The ONNX graph then has **multiple named
outputs** (input unchanged: `float32[batch, 1, L_m]`):

```json
{
  "modality": "ecg", "L_m": 2500, "fs_hz": 250.0,
  "class_order": ["Q0_unacceptable", "Q1_poor", "Q2_acceptable", "Q3_excellent"],
  "normalization": { "method": "zscore", "mean": 0.0, "std": 1.0 },
  "training_data_hash": "sha256:...", "model_version": "0.2.0",
  "heads": [
    { "name": "grade",    "output_name": "q_logits",    "kind": "ordinal",
      "activation": "softmax", "class_order": ["Q0_unacceptable","Q1_poor","Q2_acceptable","Q3_excellent"] },
    { "name": "usable",   "output_name": "bin_logits",  "kind": "binary",
      "activation": "softmax", "class_order": ["BAD", "OK"] },
    { "name": "artifact", "output_name": "type_logits", "kind": "multilabel",
      "activation": "sigmoid", "threshold": 0.5,
      "class_order": ["clean","baseline_wander","motion","muscle","electrode","powerline"] }
  ]
}
```

Rules (all validated loudly on load):

- Exactly one `ordinal` head; its `class_order` **must equal** the card's
  top-level `class_order` (it is the primary Q-grade head, and `n_classes`
  still derives from it). When `heads` is absent the app synthesizes this head
  from `class_order` and output name `q_logits`, so legacy cards keep working.
- `binary` heads use `softmax` and exactly two classes (`["BAD","OK"]`).
- `multilabel` heads (artifact-TYPE) use `sigmoid` (artifacts co-occur) and
  require a `threshold` in `[0, 1]`; each window emits the labels whose sigmoid
  prob ≥ `threshold` (the `clean` class is never emitted as a tag), and these
  tags flow into the segment table's **artifacts** column.
- Every declared `output_name` must exist in the ONNX graph outputs, or load
  fails (card/graph mismatch).

This directory ships two FP32 models (`ecg` and `eda`, each an `.onnx` +
`.model_card.json`) in version control; the app loads `<modality>.onnx`
directly. Any additional `.onnx`/`.model_card.json` files are build/release
artifacts.

## Licensing of the shipped weights

The repository's MIT `LICENSE` covers the **app source code**. It does **not**
grant rights over the `.onnx` weights here, which are derived from third-party
datasets. Both shipped models were trained exclusively on sources whose terms
are open and attribution-based (ODC-BY / CC BY 4.0 / BSD 3-Clause), verified
individually against each provider's licence page, so both may be redistributed
and used commercially **provided the dataset citations travel with them**.

`eeg.onnx` and `ppg.onnx` are deliberately **not** in this directory. Their
training corpora include credentialed-access (MIMIC-III derivatives),
non-commercial (WESAD) and data-use-agreement (TUAR) material whose terms an
openly-licensed weight release would not honour. Per-model provenance and the
full reasoning are in [`../LICENSE-MODELS`](../LICENSE-MODELS), and
machine-readably in each card:

```json
"training_data_provenance": { "store": "...", "cohorts": [...], "digest": "sha256:..." | null },
"license": { "code_license": "...", "weights_terms": "app/LICENSE-MODELS",
             "redistribution": "attribution-required" | "review-required" | "restricted",
             "commercial_use": "...", "source_terms": [...] }
```

Both blocks are documentation only — `model_card.py` ignores unknown keys, so
they carry no runtime behaviour. `training_data_hash` is a real SHA-256 digest
only where the training store recorded a snapshot manifest, and **neither
shipped model has one**: `ecg` was exported from `store_v2` and `eda` from
`store_v3`, neither of which wrote a manifest, so both carry a training-run
nickname and say so in `training_data_provenance.digest_note`. (The withdrawn
`eeg` model was the only one with a real corpus digest.) `onnx_sha256` is
unaffected — it is a digest of the shipped graph itself and is enforced at load.
