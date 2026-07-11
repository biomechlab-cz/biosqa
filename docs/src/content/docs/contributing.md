---
title: "Contributing & license"
description: "MIT license, third-party licenses, project layout, running tests, how to contribute, and dataset attribution."
---

BioSQA Studio is open source under a permissive license, with a deliberately narrow dependency footprint on the app side. This page covers licensing, the app↔engine boundary, where the code lives, and how to help.

## License

BioSQA Studio is released under the **MIT License**, © Marek Sokol. You are free to use, modify, and redistribute the app subject to the license terms and the retention of the copyright and permission notice.

> BioSQA Studio is a research and engineering tool, **not a medical device**. The MIT license disclaims warranty; nothing here constitutes clinical validation.

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

It **never** imports training code, and it carries **no deep-learning training stack** — the app and the engine have deliberately **disjoint dependency sets**. Inference runs on the CPU via ONNX Runtime. This boundary is a feature, not an accident: it keeps the app small, auditable, and free of the heavy dependencies that model training requires.

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

The framework-agnostic pieces — the parts that don't require a running Qt event loop — are unit-tested with **pytest**. CI runs the suite on both **Linux and Windows**.

```
pytest
```

## How to contribute

Contributions are welcome. Help is most valuable in two areas:

- **More input formats** — ingestion for **Zarr** and **Parquet** sources, extending the range of recordings the app can open.
- **Frozen builds** — proving out and hardening the **macOS and Linux** packaged builds.

> Whatever you add, keep the app's **no-training-deps boundary** intact. The app must not gain a dependency on a deep-learning training stack; if a change would pull one in, it belongs in the engine, not the app.

## Datasets & attribution

The per-modality models were trained on public biosignal datasets, primarily under the **PhysioNet** umbrella. Some of these sources — for example **TUAR / TUH EEG** — require **credentialed access** and are therefore **not redistributed** with the app.

We cite dataset provenance openly. If you build on or extend the models, please carry that attribution forward and respect the access terms of each source.
