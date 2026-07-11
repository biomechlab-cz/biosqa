"""Zarr v3 store access for the canonical out-of-core waveform layout (Plan 2 §6.1).

Layout on disk (one Zarr store per transcoded recording)::

    recording.zarr/
      .zgroup
      raw/<channel>            # 1-D array, chunked along time (>=1MB chunks, sharded)
      pyramid/<channel>/<factor>/min
      pyramid/<channel>/<factor>/max
      .zattrs                  # fs_hz, channel names/units, recording start time, ...

TODO(Plan2 §6.1): wire the actual sharding config (~1GB shards of ~1MB
chunks) once a real transcode pipeline exists; this module currently
implements the read/write primitives with a conservative single-chunk
default so it is exercisable without tuning for a specific dataset yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import zarr
except ImportError:  # pragma: no cover - zarr is a required app dep; guarded for import-time safety
    zarr = None  # type: ignore[assignment]

#: Plan 2 §6.2 concurrency caution: dask_threads * zarr_async_concurrency can
#: reach hundreds of concurrent ops and thrash disk. Cap it explicitly rather
#: than trusting zarr's default.
DEFAULT_ASYNC_CONCURRENCY = 8


@dataclass
class ChannelMeta:
    """Per-channel bookkeeping the view controller needs (Plan 2 §6.3)."""

    name: str
    fs_hz: float
    n_samples: int
    unit: str = ""
    chunk_size: int = 1_000_000
    pyramid_factors: tuple[int, ...] = field(default_factory=tuple)


class RecordingStore:
    """Thin wrapper around a single recording's Zarr v3 group.

    TODO(Plan2 §6.1): implement transcode-on-open for foreign formats
    (wfdb/edf) via ``io.loaders`` -> this store, including pyramid build
    (``io.pyramid.build_minmax_pyramid``) as a one-time, cancellable cost.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._group: Any = None

    @classmethod
    def create(cls, path: str | Path, channels: dict[str, ChannelMeta]) -> "RecordingStore":
        """Create a new, empty Zarr v3 store with `raw/` arrays for each channel.

        TODO(Plan2 §6.1): configure sharding (~1GB shards of ~1MB chunks) and
        Blosc compression once dataset-specific chunk-size tuning is decided;
        this uses a single conservative chunk size for now.
        """
        if zarr is None:
            raise RuntimeError("zarr is required to create a RecordingStore (see app/pyproject.toml)")

        store = cls(path)
        group = zarr.open_group(str(store.path), mode="w")
        raw_group = group.create_group("raw")
        group.create_group("pyramid")
        for meta in channels.values():
            raw_group.create_array(
                meta.name,
                shape=(meta.n_samples,),
                chunks=(min(meta.chunk_size, max(meta.n_samples, 1)),),
                dtype="float32",
            )
            raw_group[meta.name].attrs["fs_hz"] = meta.fs_hz
            raw_group[meta.name].attrs["unit"] = meta.unit
        store._group = group
        return store

    def open(self) -> "RecordingStore":
        """Open an existing store read/write. TODO(Plan2 §6.2): cap async concurrency."""
        if zarr is None:
            raise RuntimeError("zarr is required to open a RecordingStore (see app/pyproject.toml)")
        self._group = zarr.open_group(str(self.path), mode="r+")
        return self

    def read_window(self, channel: str, start_idx: int, end_idx: int) -> np.ndarray:
        """Read a raw sample window ``[start_idx, end_idx)`` for one channel.

        Only the covering chunks are touched -- callers wanting overview-scale
        data should go through ``io.pyramid`` levels instead of calling this
        with a huge range.
        """
        if self._group is None:
            self.open()
        array = self._group["raw"][channel]
        return np.asarray(array[start_idx:end_idx])

    def write_pyramid_level(self, channel: str, factor: int, y_min: np.ndarray, y_max: np.ndarray) -> None:
        """Persist one precomputed pyramid level for ``channel`` (Plan 2 §3.1)."""
        if self._group is None:
            self.open()
        pyramid_group = self._group["pyramid"].require_group(channel).require_group(str(factor))
        pyramid_group["min"] = y_min
        pyramid_group["max"] = y_max

    def read_pyramid_level(self, channel: str, factor: int) -> tuple[np.ndarray, np.ndarray]:
        """Load a precomputed pyramid level back from disk."""
        if self._group is None:
            self.open()
        pyramid_group = self._group["pyramid"][channel][str(factor)]
        return np.asarray(pyramid_group["min"]), np.asarray(pyramid_group["max"])
