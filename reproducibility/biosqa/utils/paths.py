"""Canonical repository paths.

Resolved relative to this file so scripts work regardless of CWD. The repo root
is the parent of ``src/``. Layout maps Plan 1 §4 onto the user's existing
top-level folders (data/, experiments/, scripts/, notebooks/, utils/, models/).
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

SRC = REPO_ROOT / "src"
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

# Raw datasets live outside the repo (large, read-only).
RAW_DATASETS = Path("D:/Quality Datasets")


def ensure_dirs() -> None:
    """Create the derived-data directories if missing (idempotent)."""
    for d in (DATA, MODELS, STORE_DIR):
        d.mkdir(parents=True, exist_ok=True)
