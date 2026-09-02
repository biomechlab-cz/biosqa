---
title: "Contributing & license"
description: "MIT license, third-party licenses, project layout, running tests, how to contribute, and dataset attribution."
---

BioSQA Studio is open source under a permissive license, with a deliberately narrow dependency footprint on the app side. This page covers licensing, the app↔engine boundary, where the code lives, and how to help.

## License

The **app source code** is released under the **MIT License**, © Marek Sokol. You are free to use, modify, and redistribute the code subject to the license terms and the retention of the copyright and permission notice.

> BioSQA Studio is a research and engineering tool, **not a medical device**. The MIT license disclaims warranty; nothing here constitutes clinical validation.

### The model weights are not MIT

The MIT grant covers the code, **not the trained `.onnx` weights** shipped in `models/`. Those are derived from third-party biosignal datasets whose terms the weights inherit. Only models whose entire lineage is open and attribution-based are shipped:

| Model | Shipped? | Training-data terms |
| --- | --- | --- |
| `ecg.onnx` | **yes** | seven open-access PhysioNet cohorts, ODC-BY v1.0 / CC BY 4.0; attribution required, no DUA, no non-commercial clause |
| `eda.onnx` | **yes** | EDABE (CC BY 4.0) + EDA-Artifact-Detection UTD/AWW (BSD 3-Clause); both permissive, both permit derived works |
| `ppg.onnx` | **no** | **MIMIC-III-Ext-PPG**: PhysioNet Credentialed Health Data License 1.5.0, signed DUA, access may not be shared onward; **WESAD**: "scientific, non-commercial purposes" only |
| `eeg.onnx` | **no** | **TUAR / TUH EEG**: signed Neural Engineering Data Consortium agreement restricting redistribution of the data and silent on derived weights |

So the two shipped models **may be redistributed and used commercially, provided the dataset citations travel with them**. The two withheld ones are not in this repository at all; the app still opens EEG and PPG recordings and will use your own weights if you supply them.

Per-model provenance is in [`LICENSE-MODELS`](https://github.com/biomechlab-cz/biosqa/blob/main/LICENSE-MODELS) in the repository root, and machine-readably in the `license` block of each model card. It is a provenance statement, **not legal advice**: check each source's current terms before redistributing the weights or using them commercially.

## Third-party licenses

The app bundles and builds on several open-source projects. Their licenses apply to their respective components, and we acknowledge them with thanks:

| Component | License |
| --- | --- |
| **PySide6 / Qt 6** | LGPLv3 |
| **ONNX Runtime** | MIT |
| **wfdb** | MIT |
| **MNE-Python** | BSD-3 |
| **NumPy / SciPy** | Per their respective licenses |
| Bundled fonts | Per their respective font licenses |

## The app↔engine contract

BioSQA Studio is one half of a two-part project. The other half is the training and research engine, which produces the models. The two are joined by a single, deliberately narrow contract.

The app consumes **only** two files per modality:

```
models/<modality>.onnx
models/<modality>.model_card.json
```

It **never** imports training code, and it carries **no deep-learning training stack**: the app and the engine have deliberately **disjoint dependency sets**. Inference runs on the CPU via ONNX Runtime. This boundary is a feature, not an accident: it keeps the app small, auditable, and free of the heavy dependencies that model training requires.

See [Models & model cards](/biosqa/docs/models/) for how the app loads and validates these files.

## Project layout

A short map of the repository:

| Path | Contents |
| --- | --- |
| app package | inference, viewmodels, workers, ui, io, model |
| `models/` | per-modality `.onnx` + `.model_card.json` |
| `tests/` | pytest suite |
| `docs/` | this documentation site |

The app package separates concerns: the **inference** and **model** layers load and run ONNX; **workers** handle out-of-core streaming off the UI thread; **viewmodels** bridge to the QML **ui**; and **io** handles reading recordings.

## Running tests

The framework-agnostic pieces, the parts that don't require a running Qt event loop, are unit-tested with **pytest**. CI runs the suite on both **Linux and Windows**.

```
pytest
```

## How to contribute

Contributions are welcome. Help is most valuable in two areas:

- **More input formats**: ingestion for **Zarr** and **Parquet** sources, extending the range of recordings the app can open.
- **Frozen builds**: proving out and hardening the **macOS and Linux** packaged builds.

> Whatever you add, keep the app's **no-training-deps boundary** intact. The app must not gain a dependency on a deep-learning training stack; if a change would pull one in, it belongs in the engine, not the app.

## Datasets & attribution

The **shipped** models were trained on these datasets, for ECG: CinC-2011, BUT QDB, European ST-T, MIT-BIH VFDB/NSTDB/SVDB and PTB-XL; for EDA: EDABE and EDA-Artifact-Detection (UTD + AWW). Every one is open access under an attribution licence. The full citation list is in the repository [README](https://github.com/biomechlab-cz/biosqa#datasets), and carrying it forward is a **condition** of redistributing the weights, not a courtesy: ODC-BY, CC BY 4.0 and BSD 3-Clause all require attribution, so an MIT notice alone is not sufficient.

The wider research corpus behind the project also covers PPG (BUT PPG, PPG-DaLiA, WESAD, MIMIC-III-Ext-PPG) and EEG (TUAR, PhysioMotion, Phantom-EEG, Mind-in-Motion, Motion-Artifact fNIRS+EEG). Some of those, **TUAR / TUH EEG**, **MIMIC-III-Ext-PPG**: require credentialed access or a signed agreement, and **WESAD** permits non-commercial use only. No dataset is redistributed here, and no model derived from those sources is shipped.

We cite dataset provenance openly. If you build on or extend the models, please carry that attribution forward and respect the access terms of each source.
