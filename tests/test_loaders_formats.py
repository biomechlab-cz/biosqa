"""Format sniffing, the modality vote, Parquet reading, and the open(modality=) override."""
from pathlib import Path

import numpy as np
import pytest

from biosqa.io.loaders import (
    RecordingHandle, detect_modality, guess_format, modality_vote,
    open_recording, read_window, sniff_format,
)

_DUMMY = Path(__file__).resolve().parents[1] / "dummy_data"


def _w(tmp_path, name, data: bytes) -> Path:
    p = tmp_path / name
    p.write_bytes(data)
    return p


def test_sniff_edf_magic(tmp_path):
    p = _w(tmp_path, "x.edf", b"0       " + b" " * 512)
    assert guess_format(p).format == "mne"


def test_sniff_bdf_routes_to_mne_not_edf(tmp_path):
    # The live bug this fixes: .bdf used to map to read_raw_edf, which REJECTS BDF.
    p = _w(tmp_path, "x.bdf", b"\xffBIOSEMI" + b" " * 512)
    g = guess_format(p)
    assert g.format == "mne" and "BIOSEMI" in g.reason
    assert sniff_format(tmp_path / "y.bdf") == "mne"          # extension fallback also routes to mne


def test_sniff_parquet_and_gdf(tmp_path):
    assert guess_format(_w(tmp_path, "a.bin", b"PAR1" + b"\0" * 64)).format == "parquet"
    assert guess_format(_w(tmp_path, "b.gdf", b"GDF" + b"\0" * 64)).format == "mne"


def _handle(units, names=("ch0",), fs=250.0):
    return RecordingHandle(path=Path("x"), format="wfdb", channel_names=list(names),
                           fs_hz={n: fs for n in names}, n_samples={n: 100 for n in names},
                           backend=None, units=units)


def test_modality_vote_units_win():
    assert modality_vote(_handle({"ch0": "uV"}))[0] == "eeg"
    assert modality_vote(_handle({"ch0": "mV"}))[0] == "ecg"
    assert modality_vote(_handle({"ch0": "uS"}))[0] == "eda"
    m, c = modality_vote(_handle({"ch0": "uV"}))
    assert c > 0.5 and detect_modality(_handle({"ch0": "uV"})) == "eeg"


def test_modality_low_confidence_on_generic_labels():
    # generic name + no unit -> only the weak fs tie-break fires => low confidence (ask the user)
    m, c = modality_vote(_handle({}, names=("signal1",)))
    assert c <= 0.35


def test_parquet_open_and_read(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    n = 500
    t = np.arange(n) / 100.0
    pq.write_table(pa.table({"time": t, "PLETH": np.sin(t)}), tmp_path / "rec.parquet")
    h = open_recording(tmp_path / "rec.parquet")
    assert h.format == "parquet" and "PLETH" in h.channel_names and "time" not in h.channel_names
    assert abs(h.fs_hz["PLETH"] - 100.0) < 1.0
    assert read_window(h, ["PLETH"], 0, 50).shape == (50, 1)


def test_parquet_window_reads_only_overlapping_row_groups(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    n = 10000
    t = np.arange(n) / 250.0
    ch = np.sin(t)
    pq.write_table(pa.table({"time": t, "II": ch}), tmp_path / "r.parquet", row_group_size=1000)
    h = open_recording(tmp_path / "r.parquet")
    assert pq.ParquetFile(str(tmp_path / "r.parquet")).metadata.num_row_groups == 10
    # a mid-file window matches the exact slice (windowed, not a full-column read)
    assert np.allclose(read_window(h, ["II"], 2500, 2600).reshape(-1), ch[2500:2600])
    # a window straddling a row-group boundary still stitches correctly
    assert np.allclose(read_window(h, ["II"], 990, 1010).reshape(-1), ch[990:1010])


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.mark.skipif(not (_DUMMY / "test_ecg_3min.hea").exists(), reason="dummy_data not generated")
def test_open_forces_modality(qapp):
    from biosqa.viewmodels.recording_controller import RecordingListModel

    m = RecordingListModel()
    ecg = str((_DUMMY / "test_ecg_3min.hea").resolve())
    m.open(ecg, "eeg")                       # force EEG on an ECG file (the "Open ▸ EEG" path)
    info = m.handle_for(ecg)
    assert info is not None and info[1] == "eeg" and m.currentModality == "eeg"
    m.setModality(ecg, "ppg")                # post-open correction
    assert m.handle_for(ecg)[1] == "ppg"
