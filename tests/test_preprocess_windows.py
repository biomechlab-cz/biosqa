"""Windowing contract: a record shorter than one window must NOT grade to silence, and the
trailing partial window must NOT go ungraded (both read to a user as "no problems found")."""
from pathlib import Path

import numpy as np
import pytest

from biosqa.inference.preprocess import (
    ShortRecordError,
    make_windows,
    ungraded_tail_samples,
    window_starts,
)
from biosqa.model.model_card import ModelCard, Normalization

# the SHIPPED EDA card: L_m=480 @ 8 Hz == a 60 SECOND window (the worst case for both defects)
EDA = ModelCard(
    modality="eda", l_m=480, fs_hz=8.0, class_order=("Q0", "Q1", "Q2", "Q3"),
    normalization=Normalization(method="none"), training_data_hash="sha256:test",
    model_version="test", source_path=Path("eda.model_card.json"),
)


def _ramp(n: int) -> np.ndarray:
    return np.arange(n, dtype=np.float32)      # every sample distinguishable -> slices are checkable


def test_short_record_raises_instead_of_returning_empty():
    """A 25 s EDA record cannot fill the 60 s window: it must raise with the facts, never return an
    empty array (zero windows -> zero segments -> the UI says '0 segments' = 'nothing is wrong')."""
    with pytest.raises(ShortRecordError) as exc:
        make_windows(_ramp(200), EDA)
    err = exc.value
    assert err.n_samples == 200 and err.l_m == 480
    assert err.record_sec == pytest.approx(25.0) and err.required_sec == pytest.approx(60.0)
    assert "25.0 s" in str(err) and "60.0 s" in str(err)
    assert isinstance(err, ValueError)         # catchable by the workers' existing broad handlers


def test_empty_signal_raises_too():
    with pytest.raises(ShortRecordError):
        make_windows(np.zeros(0, dtype=np.float32), EDA)


def test_tail_of_a_3_9_window_record_is_graded():
    """3.9 windows: the old grid dropped the last 432 samples (54 s, 23% of the record). The final
    window is end-anchored so every sample lands in at least one graded window."""
    n = 1872                                    # 3.9 * 480
    sig = _ramp(n)
    wins, starts = make_windows(sig, EDA, return_starts=True)
    assert starts.tolist() == [0, 480, 960, 1392]           # 3 grid windows + the end-anchored tail
    assert wins.shape == (4, 480)
    assert np.array_equal(wins[-1], sig[n - 480:])          # the tail window IS the record's tail
    assert ungraded_tail_samples(n, EDA) == 0
    covered = np.zeros(n, dtype=bool)
    for s in starts:
        covered[s:s + 480] = True
    assert covered.all()                                    # no ungraded sample anywhere


def test_tail_window_is_emitted_with_overlap_too():
    n = 1500                                                # step=240 at overlap=0.5
    starts = window_starts(n, EDA, overlap=0.5)
    assert starts.tolist() == [0, 240, 480, 720, 960, 1020]  # last one end-anchored (1500-480)
    assert ungraded_tail_samples(n, EDA, overlap=0.5) == 0


def test_exact_multiple_is_bit_for_bit_unchanged():
    """No-op guard: a record that tiles evenly must produce exactly the legacy grid -- no extra
    window, identical samples (the end-anchored tail only exists when there IS a tail)."""
    for overlap in (0.0, 0.5):
        n = 480 * 4
        sig = _ramp(n)
        step = max(1, int(round(480 * (1.0 - overlap))))
        legacy_n = (n - 480) // step + 1
        legacy = np.stack([sig[i * step:i * step + 480] for i in range(legacy_n)]).astype(np.float32)
        wins = make_windows(sig, EDA, overlap=overlap)
        assert wins.shape == legacy.shape
        assert np.array_equal(wins, legacy)
        assert window_starts(n, EDA, overlap=overlap).tolist() == [i * step for i in range(legacy_n)]
        assert ungraded_tail_samples(n, EDA, overlap=overlap) == 0


def test_cover_tail_false_reports_the_ungraded_samples():
    """The legacy drop-the-tail grid stays available, but the ungraded tail is now countable, so a
    caller that opts out can mark it not-analyzed instead of leaving it invisible."""
    n = 1872
    wins = make_windows(_ramp(n), EDA, cover_tail=False)
    assert wins.shape == (3, 480)
    assert ungraded_tail_samples(n, EDA, cover_tail=False) == 432      # 54 s @ 8 Hz
    assert make_windows(_ramp(n), EDA).shape == (4, 480)               # ... the DEFAULT still covers it


def test_overlap_bounds_still_validated():
    with pytest.raises(ValueError):
        make_windows(_ramp(1000), EDA, overlap=1.0)
