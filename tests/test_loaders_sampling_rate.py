"""Regressions for how a recording's sampling rate and channel identity are derived.

Both defects here shipped: a Parquet time column was differenced with no unit check (a datetime64
column HUNG the load worker; a millisecond column mis-timed the whole recording by 1000x), and WFDB
channels were addressed by name, so a record with two leads called 'ECG' served lead 0 for both.
"""
from pathlib import Path

import numpy as np
import pytest

from biosqa.io.loaders import open_recording, read_window

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

FS = 250.0
N = 2500


def _sig(n=N):
    return np.sin(2 * np.pi * 1.2 * np.arange(n) / FS)


def _write(tmp_path, name, cols) -> Path:
    p = tmp_path / name
    pq.write_table(pa.table(cols), p)
    return p


def test_parquet_datetime_column_gives_the_true_rate(tmp_path):
    """A pandas datetime64[ns] 'timestamp' column at a true 250 Hz used to read as 2.5e-07 Hz, which
    asked resample_signal for a ~2e10-tap FIR: the load worker neither returned nor raised."""
    pd = pytest.importorskip("pandas")
    ts = pd.to_datetime(np.arange(N) / FS, unit="s", origin="2024-01-01")
    h = open_recording(_write(tmp_path, "dt.parquet", {"timestamp": ts, "ECG": _sig()}))
    assert h.fs_hz["ECG"] == pytest.approx(FS, rel=1e-3)


def test_parquet_millisecond_column_is_not_read_as_seconds(tmp_path):
    """The normal wearable export. Read as seconds it gives 0.25 Hz, which resamples CLEANLY at
    up=1000 -- so a 10 s recording was analysed and exported as a 2.8-hour one, silently."""
    h = open_recording(_write(tmp_path, "ms.parquet",
                              {"time": np.arange(N) * (1000.0 / FS), "ECG": _sig()}))
    assert h.fs_hz["ECG"] == pytest.approx(FS)


def test_parquet_seconds_column_is_never_reinterpreted(tmp_path):
    """A genuine seconds column stays seconds: 8 Hz EDA is plausible, so no unit search may fire."""
    n = 800
    h = open_recording(_write(tmp_path, "eda.parquet",
                              {"time": np.arange(n) / 8.0, "EDA": np.sin(np.arange(n) / 8.0)}))
    assert h.fs_hz["EDA"] == pytest.approx(8.0)


def test_parquet_without_a_time_column_says_so_instead_of_assuming_250(tmp_path):
    """It used to default to 250.0 for every modality -- a 4 Hz EDA file analysed as 250 Hz."""
    with pytest.raises(ValueError, match="sampling rate unknown"):
        open_recording(_write(tmp_path, "none.parquet", {"ECG": _sig()}))


def test_parquet_unresolvable_time_column_is_an_error_not_a_default(tmp_path):
    """A constant time column has no rate at all; it must fail loudly, not silently."""
    with pytest.raises(ValueError, match="sampling rate unknown"):
        open_recording(_write(tmp_path, "flat.parquet",
                              {"time": np.zeros(N), "ECG": _sig()}))


def test_parquet_iso8601_string_time_column_resolves_the_rate(tmp_path):
    """A CSV -> Parquet round trip writes the timestamps back as ISO-8601 TEXT. That fell through to
    ``np.asarray(t, dtype=float)`` and surfaced numpy's raw ``could not convert string to float:
    '1970-01-01 00:00:00.000'`` -- a message that names neither the file, the column, nor the problem.
    It is a perfectly well-defined time column: parse it."""
    pd = pytest.importorskip("pandas")
    ts = pd.to_datetime(np.arange(N) / FS, unit="s", origin="2024-01-01")
    h = open_recording(_write(tmp_path, "isotime.parquet",
                              {"time": [str(x) for x in ts], "ECG": _sig()}))
    assert h.fs_hz["ECG"] == pytest.approx(FS, rel=1e-3)
    assert read_window(h, ["ECG"], 0, 50).shape == (50, 1)


def test_parquet_unparseable_text_time_column_gets_the_modules_own_message(tmp_path):
    """...and text that is neither numeric nor a timestamp must still fail in THIS module's words,
    not numpy's. Fail-loud is only useful if the message says what to do about it."""
    with pytest.raises(ValueError, match="sampling rate unknown"):
        open_recording(_write(tmp_path, "wordtime.parquet",
                              {"time": [f"sample-{i}" for i in range(N)], "ECG": _sig()}))


def test_parquet_non_numeric_columns_are_not_channels(tmp_path):
    """A string annotation column was exposed as a channel and only failed later, deep inside
    read_parquet_window ('could not convert string to float')."""
    h = open_recording(_write(tmp_path, "str.parquet", {
        "time": np.arange(N) / FS, "ECG": _sig(), "note": ["ok"] * N,
    }))
    assert h.channel_names == ["ECG"]
    assert read_window(h, ["ECG"], 0, 50).shape == (50, 1)


def test_wfdb_duplicate_signal_names_stay_distinct_channels(tmp_path):
    """44 real records (Long Term ST DB) are literally sig_name ['ECG', 'ECG']. Keyed by name, the
    second lead aliased onto the first and the Coordinator's dedup then dropped it entirely."""
    wfdb = pytest.importorskip("wfdb")
    n = 1000
    a = _sig(n)
    b = _sig(n) + 5.0                                   # a plainly different second lead
    wfdb.wrsamp("dup", fs=FS, units=["mV", "mV"], sig_name=["ECG", "ECG2"],
                p_signal=np.stack([a, b], axis=1), write_dir=str(tmp_path))
    hea = tmp_path / "dup.hea"                          # wrsamp refuses to WRITE a duplicate name;
    hea.write_text(hea.read_text().replace("ECG2", "ECG"))   # the real records nonetheless have one
    h = open_recording(hea)
    assert h.channel_names == ["ECG", "ECG#2"] and h.channel_index == {"ECG": 0, "ECG#2": 1}
    got_a = read_window(h, ["ECG"], 0, 200).reshape(-1)
    got_b = read_window(h, ["ECG#2"], 0, 200).reshape(-1)
    assert np.allclose(got_a, a[:200], atol=1e-3)
    assert np.allclose(got_b, b[:200], atol=1e-3)       # ... not lead 0's samples again


def _write_header_only(tmp_path, name, n_sig, fs, n_samp):
    """Hand-write a WFDB header + matching .dat.

    ``wfdb.wrsamp`` REFUSES to write a one-sample record, so the fixture cannot be
    produced through the library that reads it -- which is the point: this shape is
    not something a well-formed writer emits, but it is what BUT PPG ships and what
    ``rdheader`` happily parses.
    """
    lines = [f"{name} {n_sig} {fs} {n_samp}"]
    lines += [f"{name}.dat 16 200 16 0 0 0 0 "] * n_sig
    (tmp_path / f"{name}.hea").write_text("\n".join(lines) + "\n", encoding="ascii")
    (tmp_path / f"{name}.dat").write_bytes(bytes(2 * n_sig * n_samp))
    return tmp_path / f"{name}.hea"


def test_wfdb_header_with_one_sample_per_signal_is_refused(tmp_path):
    """A syntactically valid header that describes no recording must be REFUSED, not
    turned into a channel list.

    BUT PPG stores one record per matrix row, so its headers read
    ``100001_ECG 10000 1000 1`` -- ten thousand signals of ONE sample each, unnamed.
    ``open_recording`` accepted that and returned 10 000 channels called None, None#2 ...
    None#10000, each one sample long: the same accept-and-return-garbage failure the
    zarr path had, but on WFDB, which is the app's primary advertised input and is
    reachable straight from the file dialog.
    """
    bad = _write_header_only(tmp_path, "onesample", n_sig=10, fs=1000, n_samp=1)
    with pytest.raises(ValueError, match="not a recording"):
        open_recording(str(bad))

    # A two-sample record IS structurally a signal. "Too short to analyse" is reported
    # downstream with a better message, so this layer must let it through.
    ok = _write_header_only(tmp_path, "twosample", n_sig=1, fs=1000, n_samp=2)
    assert len(open_recording(str(ok)).channel_names) == 1
