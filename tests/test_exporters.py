"""Round-trip tests for the quality-interval exporters (csv/tsv/json/parquet/wfdb/mat)."""
import csv
import json
from pathlib import Path

import pytest

from biosqa.export import exporters as ex
from biosqa.inference.segmenter import QualityInterval


def _intervals():
    return [
        QualityInterval(0.0, 10.0, "Q3", 0.91, ()),
        QualityInterval(10.0, 20.0, "Q1", 0.55, ("motion",)),
        QualityInterval(20.0, 30.0, "Q0", 0.40, ("motion", "muscle")),
    ]


def test_registry_covers_all_formats():
    assert set(ex.EXPORTERS) == {"csv", "tsv", "json", "parquet", "wfdb", "mat"}
    for _fmt, (writer, ext) in ex.EXPORTERS.items():
        assert callable(writer) and ext.startswith(".")


def test_csv_has_artifacts_column(tmp_path):
    p = ex.export_intervals_csv(_intervals(), tmp_path / "o.csv")
    rows = list(csv.DictReader(p.open()))
    assert len(rows) == 3 and rows[1]["tier"] == "Q1" and "motion" in rows[2]["artifacts"]


def test_tsv_is_bids_events(tmp_path):
    p = ex.export_intervals_tsv(_intervals(), tmp_path / "o.tsv")
    header = p.read_text().splitlines()[0].split("\t")
    assert header[:3] == ["onset", "duration", "trial_type"]


def test_json_carries_provenance_artifacts_tier(tmp_path):
    prov = {"modality": "ecg", "fs_hz": 250.0, "model_version": "v1"}
    p = ex.export_intervals_json(_intervals(), tmp_path / "o.json", provenance=prov)
    doc = json.loads(Path(p).read_text())
    assert doc["schema_version"] == ex.SCHEMA_VERSION
    assert doc["provenance"]["modality"] == "ecg" and doc["n_intervals"] == 3
    assert doc["intervals"][2]["artifacts"] == ["motion", "muscle"]
    assert doc["intervals"][2]["tier"] == "Q0"


def test_parquet_roundtrip(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    p = ex.export_intervals_parquet(_intervals(), tmp_path / "o.parquet")
    t = pq.read_table(p)
    assert t.num_rows == 3 and "artifacts" in t.column_names


def test_parquet_and_mat_carry_all_columns(tmp_path):
    """Regression: the flat-table formats must not silently drop columns (uncertainty/rate_usable/…)."""
    pq = pytest.importorskip("pyarrow.parquet")
    t = pq.read_table(ex.export_intervals_parquet(_intervals(), tmp_path / "o.parquet"))
    assert set(t.column_names) == set(ex.INTERVAL_COLUMNS)
    loadmat = pytest.importorskip("scipy.io").loadmat
    m = loadmat(ex.export_intervals_mat(_intervals(), tmp_path / "o.mat"))["quality_intervals"]
    assert set(ex.INTERVAL_COLUMNS) <= set(m.dtype.names)


def test_reclassify_preserves_model_tier_in_export(tmp_path):
    """A reviewer reclassify mutates the displayed tier but export must still report model_tier."""
    from biosqa.viewmodels.export_controller import ExportController
    from biosqa.viewmodels.quality_segment_model import QualitySegmentModel

    m = QualitySegmentModel()
    m.load_intervals([QualityInterval(0, 10, "Q0", 0.4, ())])
    assert m.set_tier(0, "Q2") and m._all_intervals[0].model_tier == "Q0"
    ctl = ExportController()
    ctl.attach(m, None, None)
    got = {}
    ctl.exportSucceeded.connect(lambda p: got.setdefault("p", p))
    ctl.exportToPath(str(tmp_path / "r.csv"), "csv")
    row = list(csv.DictReader(Path(got["p"]).open()))[0]
    assert row["tier"] == "Q2" and row["model_tier"] == "Q0"


def test_wfdb_annotation_roundtrip(tmp_path):
    wfdb = pytest.importorskip("wfdb")
    ex.export_intervals_wfdb(_intervals(), tmp_path / "rec.qual", fs=250.0)
    ann = wfdb.rdann(str(tmp_path / "rec"), "qual")
    assert list(ann.sample) == [0, 2500, 5000]
    assert list(ann.symbol) == ["~", "~", "~"]
    assert "Q0" in ann.aux_note[2] and "muscle" in ann.aux_note[2]


def test_wfdb_needs_fs(tmp_path):
    with pytest.raises(ValueError):
        ex.export_intervals_wfdb(_intervals(), tmp_path / "rec.qual", fs=None)


def test_mat_roundtrip(tmp_path):
    loadmat = pytest.importorskip("scipy.io").loadmat
    p = ex.export_intervals_mat(_intervals(), tmp_path / "o.mat")
    m = loadmat(p)
    assert "quality_intervals" in m


def test_controller_normalizes_extension(tmp_path):
    from biosqa.viewmodels.export_controller import ExportController

    class _Seg:
        _all_intervals = _intervals()

    c = ExportController()
    c.attach(_Seg(), None, None)
    got = {}
    c.exportSucceeded.connect(lambda p: got.setdefault("p", p))
    c.exportFailed.connect(lambda m: got.setdefault("err", m))
    c.exportToPath(str(tmp_path / "noext"), "json")
    assert got.get("p", "").endswith(".json"), got


def test_corrected_tier_is_the_effective_export_tier():
    """interval_records: a reviewer correction becomes ``tier``; the model's grade is kept in
    ``model_tier`` and ``overridden`` flips true (regression for the relabel-loses-tier bug)."""
    recs = ex.interval_records(_intervals(), corrected={2: "Q2"})
    assert recs[2]["tier"] == "Q2" and recs[2]["model_tier"] == "Q0" and recs[2]["overridden"] is True
    # untouched intervals keep tier == model_tier and overridden False
    assert recs[0]["tier"] == recs[0]["model_tier"] == "Q3" and recs[0]["overridden"] is False


def test_relabel_and_note_survive_export(tmp_path):
    """End-to-end: relabel Q0->Q2 THEN add a note (the order that used to drop the note) — the CSV
    must carry the corrected tier, the model's original tier, and the note."""
    from biosqa.viewmodels.export_controller import ExportController
    from biosqa.viewmodels.selection_controller import SelectionController

    ivs = _intervals()

    class _Seg:
        _all_intervals = ivs

    sel = SelectionController()
    sel.select(ivs[2])            # the Q0 segment
    sel.relabel("Q2")
    sel.addNote("recoverable after filtering")

    ctl = ExportController()
    ctl.attach(_Seg(), sel, None)
    got = {}
    ctl.exportSucceeded.connect(lambda p: got.setdefault("p", p))
    ctl.exportToPath(str(tmp_path / "corr.csv"), "csv")
    row = list(csv.DictReader(Path(got["p"]).open()))[2]
    assert row["tier"] == "Q2" and row["model_tier"] == "Q0"
    assert row["overridden"] == "True" and "recoverable" in row["note"]


def test_note_only_review_survives_export(tmp_path):
    """A note with NO relabel must still export (addNote previously recorded nothing)."""
    from biosqa.viewmodels.export_controller import ExportController
    from biosqa.viewmodels.selection_controller import SelectionController

    ivs = _intervals()

    class _Seg:
        _all_intervals = ivs

    sel = SelectionController()
    sel.select(ivs[1])            # the Q1 segment
    sel.addNote("baseline wander")

    ctl = ExportController()
    ctl.attach(_Seg(), sel, None)
    got = {}
    ctl.exportSucceeded.connect(lambda p: got.setdefault("p", p))
    ctl.exportToPath(str(tmp_path / "note.csv"), "csv")
    row = list(csv.DictReader(Path(got["p"]).open()))[1]
    assert row["note"] == "baseline wander" and row["tier"] == "Q1"  # note-only: tier unchanged
