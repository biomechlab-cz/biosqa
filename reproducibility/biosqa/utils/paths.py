"""Canonical repository paths.

Resolved relative to this file so scripts work regardless of CWD. Layout maps
Plan 1 §4 onto the user's existing top-level folders (data/, experiments/,
scripts/, notebooks/, utils/, models/).

NOTE — this is the one module of the reproducibility tree that is NOT a verbatim
copy of ``src/biosqa`` (see ``scripts/sync_from_src.py``). Here the package sits
at ``<root>/biosqa/`` rather than ``<root>/src/biosqa/``, one directory shallower,
so the root is found by walking up to the folder that actually owns
``configs/base.yaml`` instead of by a hard-coded ``parents[N]``.
"""
from __future__ import annotations

import os
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    """Nearest ancestor of ``start`` that contains ``configs/base.yaml``."""
    for p in (start, *start.parents):
        if (p / "configs" / "base.yaml").is_file():
            return p
    return start.parents[1]  # <root>/biosqa/utils/paths.py -> <root>


REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)

SRC = REPO_ROOT / "biosqa"
CONFIGS = REPO_ROOT / "configs"
DATA = REPO_ROOT / "data"                 # processed data + inventory + manifests
EXPERIMENTS = REPO_ROOT / "experiments"   # experiment scripts + xdomain probes
SCRIPTS = REPO_ROOT / "scripts"
NOTEBOOKS = REPO_ROOT / "notebooks"
MODELS = REPO_ROOT / "models"             # exported *.onnx + model_card.json (-> app)
RESEARCH = REPO_ROOT / "research"
RESEARCH_LOG = REPO_ROOT / "research_log.md"

# Derived data locations (created on demand).
STORE_DIR = DATA / "store"                # unified Zarr store + segments.parquet
SEGMENTS_INDEX = STORE_DIR / "segments.parquet"
ZARR_STORE = STORE_DIR / "waveforms.zarr"
RUNS_DIR = REPO_ROOT / "mlruns"           # MLflow local tracking

# Raw datasets live outside the repo (large, read-only). This module is a
# PKG_EXCEPTIONS entry in scripts/sync_from_src.py — it is deliberately NOT
# regenerated from the monorepo, because REPO_ROOT must resolve to this package
# rather than to src/biosqa. That exemption is also why the author's own drive
# letter sat here unnoticed in a public repository named in the manuscript's
# data-availability statement: no drift test can ever flag this file. Set
# BIOSQA_RAW_DATASETS to wherever the datasets were downloaded; the fallback is
# the original author's layout and is meaningless on any other machine.
RAW_DATASETS = Path(os.environ.get("BIOSQA_RAW_DATASETS", "").strip()
                    or "D:/Quality Datasets")


def ensure_dirs() -> None:
    """Create the derived-data directories if missing (idempotent)."""
    for d in (DATA, MODELS, STORE_DIR):
        d.mkdir(parents=True, exist_ok=True)
