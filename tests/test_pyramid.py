"""Unit tests for the decimation-pyramid math (Plan 2 §3.1) -- pure numpy, no Qt."""

from __future__ import annotations

import numpy as np

from biosqa.io.pyramid import (
    PYRAMID_FACTORS,
    build_minmax_pyramid,
    pyramid_slice,
    select_pyramid_level,
)


def test_build_minmax_pyramid_raw_level_is_identity():
    y = np.arange(100, dtype=np.float32)
    levels = build_minmax_pyramid(y, factors=(1, 8))
    assert np.array_equal(levels[1].y_min, y)
    assert np.array_equal(levels[1].y_max, y)


def test_build_minmax_pyramid_preserves_spikes():
    # A single large spike must survive an x8 decimation as a max, per Plan 2
    # §3.1 ("so spikes/artifacts survive downsampling").
    y = np.zeros(64, dtype=np.float32)
    y[5] = 100.0
    levels = build_minmax_pyramid(y, factors=(8,))
    assert levels[8].y_max[0] == 100.0
    assert levels[8].y_min[0] == 0.0


def test_build_minmax_pyramid_skips_factors_larger_than_signal():
    y = np.zeros(10, dtype=np.float32)
    levels = build_minmax_pyramid(y, factors=(1, 8, 64, 512))
    assert set(levels.keys()) == {1, 8}


def test_select_pyramid_level_picks_finest_covering_level():
    assert select_pyramid_level(0.5, PYRAMID_FACTORS) == 1
    assert select_pyramid_level(10, PYRAMID_FACTORS) == 8
    assert select_pyramid_level(100, PYRAMID_FACTORS) == 64
    assert select_pyramid_level(100_000, PYRAMID_FACTORS) == max(PYRAMID_FACTORS)


def test_pyramid_slice_returns_matching_windows():
    y = np.arange(64, dtype=np.float32)
    levels = build_minmax_pyramid(y, factors=(8,))
    y_min, y_max = pyramid_slice(levels[8], 0, 2)
    assert y_min.shape == (2,)
    assert y_max.shape == (2,)
