"""Multi-resolution min/max decimation pyramid (Plan 2 §3.1).

For each channel, precompute a min/max pyramid at fixed decimation factors
so the overview (and any zoom level short of raw) never has to touch more
than ``~2 * plot_width_px`` samples. Levels are stored per-bin ``(min, max)``
pairs so spikes/artifacts survive downsampling -- the precomputed analog of
PyQtGraph's live ``mode='peak'`` clipping, moved off the hot path.

This module is pure numpy (no Zarr/Qt dependency) so the core math is
cheaply unit-testable; ``io.store`` is responsible for persisting/loading
these arrays to/from the Zarr ``pyramid/`` group.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Decimation factors precomputed per channel (Plan 2 §3.1): raw, then
#: successive x8 levels. Chosen so `samples_per_pixel` at typical zoom levels
#: (whole-recording overview down to a few-second window) lands close to one
#: of these without needing an enormous number of stored levels.
PYRAMID_FACTORS: tuple[int, ...] = (1, 8, 64, 512, 4096)


@dataclass(frozen=True)
class PyramidLevel:
    """One precomputed decimation level for a single channel."""

    factor: int
    y_min: np.ndarray
    y_max: np.ndarray


def build_minmax_pyramid(
    y: np.ndarray, factors: tuple[int, ...] = PYRAMID_FACTORS
) -> dict[int, PyramidLevel]:
    """Build a min/max pyramid for one channel's raw samples.

    Level ``factor == 1`` is the raw signal itself (``y_min == y_max == y``),
    included so downstream level-selection code can treat "raw" uniformly
    with the coarser levels.

    Args:
        y: 1-D array of raw samples for one channel.
        factors: decimation factors to build, e.g. ``(1, 8, 64, 512, 4096)``.

    Returns:
        Mapping ``factor -> PyramidLevel``. Levels whose factor exceeds
        ``len(y)`` are skipped (nothing meaningful to bin).
    """
    y = np.asarray(y)
    n = y.shape[0]
    levels: dict[int, PyramidLevel] = {}

    for factor in sorted(factors):
        if factor == 1:
            levels[1] = PyramidLevel(factor=1, y_min=y, y_max=y)
            continue
        if factor > max(n, 1):
            continue

        n_bins = -(-n // factor)  # ceil division: keep a short trailing bin
        pad = n_bins * factor - n
        if pad:
            padded = np.pad(y, (0, pad), mode="edge")
        else:
            padded = y
        reshaped = padded.reshape(n_bins, factor)
        levels[factor] = PyramidLevel(
            factor=factor,
            y_min=reshaped.min(axis=1),
            y_max=reshaped.max(axis=1),
        )

    return levels


def select_pyramid_level(
    samples_per_pixel: float, factors: tuple[int, ...] = PYRAMID_FACTORS
) -> int:
    """Pick the pyramid factor whose decimation best matches the view.

    Design spec (a)/(§3.1): ``samples_per_pixel = visible_samples /
    plot_width_px``; choose the *finest* available level that is still
    coarse enough not to under-decimate (i.e. the largest factor
    ``<= samples_per_pixel``), falling back to the coarsest level if the
    view is zoomed out further than any precomputed level covers, or to raw
    (``factor == 1``) if fully zoomed in.

    Args:
        samples_per_pixel: current view's samples-per-pixel ratio.
        factors: available precomputed factors (ascending or not).

    Returns:
        The selected factor (one of ``factors``).
    """
    if samples_per_pixel <= 1:
        return min(factors)

    candidates = sorted(f for f in factors if f <= samples_per_pixel)
    if candidates:
        return max(candidates)
    # Zoomed out further than any precomputed level covers well: use the
    # coarsest level available rather than under-decimating.
    return max(factors)


def pyramid_slice(
    level: PyramidLevel, start_idx: int, end_idx: int
) -> tuple[np.ndarray, np.ndarray]:
    """Slice a precomputed level's (min, max) arrays to a bin-index window.

    ``start_idx``/``end_idx`` are indices into the *decimated* level, not
    raw sample indices -- callers convert raw sample range -> bin range via
    ``raw_index // level.factor`` before calling this.
    """
    return level.y_min[start_idx:end_idx], level.y_max[start_idx:end_idx]
