"""Config loading/merging + content hashing (Plan 1 §3.2 experiment hygiene).

An experiment config is ``configs/base.yaml`` deep-merged with an experiment
override file (``configs/experiment/<name>.yaml``) and optional CLI dotlist
overrides. The merged, resolved config is hashed so every run records exactly
what it ran (git SHA + config hash + data-manifest hash, per §3.2).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

from omegaconf import DictConfig, OmegaConf

from .paths import CONFIGS

__all__ = ["load_config", "config_hash", "to_container"]


def load_config(
    experiment: str | Path | None = None,
    *,
    base: str | Path = "base.yaml",
    overrides: Sequence[str] | None = None,
) -> DictConfig:
    """Load and deep-merge base + experiment + CLI dotlist overrides.

    Parameters
    ----------
    experiment : experiment name (``foo`` -> ``configs/experiment/foo.yaml``) or
        a path. If None, only base is loaded.
    base : base config filename under ``configs/`` (or a path).
    overrides : OmegaConf dotlist entries, e.g. ``["train.lr=1e-4", "seed=1"]``.
    """
    base_path = _resolve(base, CONFIGS)
    cfg = OmegaConf.load(base_path)

    if experiment is not None:
        exp_path = _resolve(experiment, CONFIGS / "experiment", suffix=".yaml")
        cfg = OmegaConf.merge(cfg, OmegaConf.load(exp_path))

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))

    OmegaConf.resolve(cfg)
    return cfg  # type: ignore[return-value]


def to_container(cfg: DictConfig) -> dict:
    """Plain nested dict (for JSON/MLflow logging)."""
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def config_hash(cfg: DictConfig, length: int = 12) -> str:
    """Stable short hash of the resolved config (order-independent)."""
    payload = json.dumps(to_container(cfg), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:length]


def _resolve(name: str | Path, root: Path, *, suffix: str = "") -> Path:
    p = Path(name)
    if p.is_absolute() and p.exists():
        return p
    if suffix and p.suffix == "":
        p = p.with_suffix(suffix)
    cand = root / p
    if cand.exists():
        return cand
    if p.exists():
        return p
    raise FileNotFoundError(f"Config not found: tried {cand} and {p}")
