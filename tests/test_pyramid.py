"""Unit tests for the decimation-pyramid math (Plan 2 §3.1) -- pure numpy, no Qt."""

from __future__ import annotations

import numpy as np

from biosqa.io.pyramid import (
    PYRAMID_FACTORS,
    build_minmax_pyramid,
    minmax_envelope_indices,
    pyramid_slice,
    samples_per_bucket,
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


# ---- the envelope the plot caches decimate with ---------------------------
def test_minmax_envelope_keeps_a_spike_naive_striding_drops():
    y = np.zeros(1000, dtype=np.float64)
    y[517] = 9.0                                    # a single-sample artifact, off any stride grid
    spb = samples_per_bucket(1000, 100)             # 100-point budget -> 50 buckets of 20 samples
    assert spb == 20
    assert y[::spb].max() == 0.0                    # naive raw[::stride] loses it entirely
    idx = minmax_envelope_indices(y, spb)
    assert 517 in idx.tolist()                      # the envelope keeps it as its bucket's max
    assert idx.size == 100                          # ...at exactly the same point budget
    assert np.all(np.diff(idx) >= 0)                # non-decreasing -> x stays monotone for the plot


def test_minmax_envelope_emits_both_extremes_in_time_order():
    y = np.array([0.0, -3.0, 1.0, 2.0, 5.0, 0.5], dtype=np.float64)
    idx = minmax_envelope_indices(y, 3)             # buckets [0,-3,1] and [2,5,0.5]
    # bucket 0: min@1 (-3) before max@2 (1); bucket 1: max@4 (5) before min@5 (0.5) — TIME order,
    # not min-then-max, so the plotted x never goes backwards.
    assert idx.tolist() == [1, 2, 4, 5]


def test_minmax_envelope_does_not_let_a_nan_win_every_bucket():
    # A NaN is not an extremum: it must not be picked over real samples (it would shred the trace),
    # but an all-NaN bucket still yields its NaN so a genuine data gap stays visible.
    y = np.array([1.0, np.nan, 3.0, np.nan, np.nan, np.nan], dtype=np.float64)
    idx = minmax_envelope_indices(y, 3)
    assert np.isfinite(y[idx[:2]]).all() and set(idx[:2].tolist()) == {0, 2}
    assert np.isnan(y[idx[2:]]).all()


def test_samples_per_bucket_is_one_under_the_cap():
    assert samples_per_bucket(100, 400) == 1        # fits: no decimation at all
    assert samples_per_bucket(1_000_000, 400_000) == 5
