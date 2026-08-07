"""Regressions for the Zarr opener, which had no tests at all until this file.

The Zarr path is de-advertised (`app/README.md`, `docs/opening-recordings.md`: "In progress ...
not yet complete") and unreachable through the shipped OpenFile dialog, so this is library
hygiene, not a user-facing safety gap. The defect was still real and silent: `open_zarr` read one
sampling rate off the GROUP and fell back to 250.0, and never looked at an array's rank. Pointed at
`data/store_v8/waveforms.zarr` it therefore SUCCEEDED -- returning the four `[N,1,L]` modality
arrays as four channels of one recording, 250 Hz for all (31.25x wrong for 8 Hz EDA), `n_samples`
set to the window count (72.632 s against a true 181,580 s), and a transposed 4-D `read_window`.
"""
from pathlib import Path

import numpy as np
import pytest

from biosqa.io.loaders import guess_format, open_recording, read_window

zarr = pytest.importorskip("zarr")


def _flat_store(tmp_path, channels: dict, group_attrs: dict | None = None) -> Path:
    """A recording-shaped store: top-level 1-D arrays, ``{name: (data, attrs)}``."""
    p = tmp_path / "rec.zarr"
    g = zarr.open_group(str(p), mode="w")
    for name, (data, attrs) in channels.items():
        data = np.asarray(data, dtype="float32")
        a = g.create_array(name, shape=data.shape, dtype="float32")
        a[...] = data
        a.attrs.update(attrs)
    g.attrs.update(group_attrs or {})
    return p


def _segment_store(tmp_path) -> Path:
    """A tiny stand-in for `data/store_v8/waveforms.zarr` -- the real one is 52,431 windows.

    Same shape contract as `src/biosqa/data/store.py::build_store`: one `[N, 1, L]` array per
    MODALITY, no attrs anywhere, window lengths differing per modality.
    """
    p = tmp_path / "waveforms.zarr"
    g = zarr.open_group(str(p), mode="w")
    for name, (n, length) in {"ecg": (4, 250), "ppg": (3, 64), "eeg": (2, 128), "eda": (2, 48)}.items():
        a = g.create_array(name, shape=(n, 1, length), dtype="float32")
        a[...] = np.zeros((n, 1, length), dtype="float32")
    return p


# --- the store_v8 trap ------------------------------------------------------------------
def test_segment_store_is_refused_not_opened_as_a_recording(tmp_path):
    """The headline defect: a research SegmentStore opened cleanly as a 4-channel recording."""
    p = _segment_store(tmp_path)
    assert guess_format(p).format == "zarr"          # it still SNIFFS as zarr; the opener refuses it
    with pytest.raises(ValueError) as exc:
        open_recording(p)
    msg = str(exc.value)
    assert "SegmentStore" in msg                     # names what it actually is
    assert "3-D" in msg and "not a recording" in msg


def test_segment_store_refusal_names_the_offending_array(tmp_path):
    """A bare 'bad shape' would leave the user guessing which of the four arrays tripped it."""
    with pytest.raises(ValueError, match=r"'(ecg|ppg|eeg|eda)' is 3-D \(\d+, 1, \d+\)"):
        open_recording(_segment_store(tmp_path))


# --- sampling rate: declared, or refused --------------------------------------------------
def test_per_channel_rates_are_read_per_channel(tmp_path):
    """One rate for the whole store cannot be right for a mixed-modality one: 250 Hz ECG next to
    8 Hz EDA used to come back as 250/250."""
    p = _flat_store(tmp_path, {
        "ECG": (np.zeros(2500), {"fs_hz": 250.0, "unit": "mV"}),
        "EDA": (np.zeros(80), {"fs_hz": 8.0, "unit": "uS"}),
    })
    h = open_recording(p)
    assert h.fs_hz == {"ECG": 250.0, "EDA": 8.0}
    assert h.n_samples == {"ECG": 2500, "EDA": 80}
    assert h.units == {"ECG": "mV", "EDA": "uS"}     # units were hardcoded to "" for every channel


def test_group_level_rate_still_applies_to_every_channel(tmp_path):
    """A single-rate store declaring `fs` (or `sfreq`) on the group keeps working -- the fix
    removes the DEFAULT, not the group-level attribute."""
    for key in ("fs", "sfreq", "fs_hz"):
        p = _flat_store(tmp_path / key, {"A": (np.zeros(64), {}), "B": (np.zeros(64), {})},
                        group_attrs={key: 64.0})
        assert open_recording(p).fs_hz == {"A": 64.0, "B": 64.0}


def test_array_attribute_wins_over_the_group(tmp_path):
    p = _flat_store(tmp_path, {"ECG": (np.zeros(100), {"fs_hz": 360.0}), "EDA": (np.zeros(32), {})},
                    group_attrs={"fs": 4.0})
    assert open_recording(p).fs_hz == {"ECG": 360.0, "EDA": 4.0}


def test_missing_rate_is_refused_instead_of_defaulting_to_250(tmp_path):
    """It used to return 250.0 for a store that declares nothing -- which is every store this repo
    writes. Same contract as the Parquet opener's 'sampling rate unknown'."""
    p = _flat_store(tmp_path, {"EDA": (np.zeros(80), {})})
    with pytest.raises(ValueError, match="sampling rate unknown"):
        open_recording(p)


def test_implausible_declared_rate_is_refused(tmp_path):
    """0 Hz used to survive the `or 250.0` fallback as a silent 250; 8e6 Hz is a unit error."""
    for bad in (0.0, 8e6):
        p = _flat_store(tmp_path / str(bad), {"X": (np.zeros(64), {"fs_hz": bad})})
        with pytest.raises(ValueError, match="plausible"):
            open_recording(p)


def test_non_numeric_rate_is_refused_in_this_module_s_words(tmp_path):
    """`float("fast")` would otherwise surface numpy/Python's raw ValueError text."""
    p = _flat_store(tmp_path, {"X": (np.zeros(64), {"fs_hz": "fast"})})
    with pytest.raises(ValueError, match="non-numeric"):
        open_recording(p)


# --- structure --------------------------------------------------------------------------
def test_nested_layout_is_refused_instead_of_a_zero_channel_recording(tmp_path):
    """`io.store.RecordingStore.create` writes `raw/<channel>`; this opener only reads the flat
    layout, and used to answer with a RecordingHandle carrying no channels at all."""
    p = tmp_path / "nested.zarr"
    g = zarr.open_group(str(p), mode="w")
    g.create_group("raw").create_array("ECG", shape=(100,), dtype="float32")
    with pytest.raises(ValueError, match="no top-level arrays"):
        open_recording(p)


def test_read_window_returns_samples_by_channels(tmp_path):
    """With rank enforced at open, the read is 2-D `[n_samples, n_channels]` like every other
    format. On a SegmentStore it returned `(2500, 1, 10, 1)`."""
    p = _flat_store(tmp_path, {
        "A": (np.arange(100), {"fs_hz": 100.0}),
        "B": (np.arange(100) * -1.0, {"fs_hz": 100.0}),
    })
    w = read_window(open_recording(p), ["A", "B"], 10, 20)
    assert w.shape == (10, 2)
    assert np.allclose(w[:, 0], np.arange(10, 20))
    assert np.allclose(w[:, 1], -np.arange(10, 20))
