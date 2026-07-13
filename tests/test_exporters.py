"""Round-trip tests for the quality-interval exporters (csv/tsv/json/parquet/wfdb/mat)."""
import csv
import json
import sys
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
    """Regression: the flat-table formats must not silently drop columns (uncertainty/rate_usable/…).

    ``channel`` is named LITERALLY here as well as via INTERVAL_COLUMNS: asserting only against the
    constant is self-referential — deleting the column from the constant would keep this green."""
    pq = pytest.importorskip("pyarrow.parquet")
    assert "channel" in ex.INTERVAL_COLUMNS     # the grades describe ONE channel; the table must say which
    t = pq.read_table(ex.export_intervals_parquet(_intervals(), tmp_path / "o.parquet"))
    assert set(t.column_names) == set(ex.INTERVAL_COLUMNS)
    assert "channel" in t.column_names
    loadmat = pytest.importorskip("scipy.io").loadmat
    m = loadmat(ex.export_intervals_mat(_intervals(), tmp_path / "o.mat"))["quality_intervals"]
    assert set(ex.INTERVAL_COLUMNS) <= set(m.dtype.names)
    assert "channel" in m.dtype.names


def test_flat_writers_name_the_graded_channel(tmp_path):
    """C: csv/tsv/parquet/mat are the formats that feed downstream analysis and they carried no
    channel at all — so a grade could not be tied back to the signal it was computed on. Unknown
    channel stays an EMPTY cell (an honest blank, never a guessed "0" or the first channel)."""
    pq = pytest.importorskip("pyarrow.parquet")
    loadmat = pytest.importorskip("scipy.io").loadmat

    rows = list(csv.DictReader(ex.export_intervals_csv(_intervals(), tmp_path / "c.csv",
                                                       channel="II").open()))
    assert [r["channel"] for r in rows] == ["II", "II", "II"]

    tsv = ex.export_intervals_tsv(_intervals(), tmp_path / "c.tsv", channel="II").read_text()
    lines = tsv.splitlines()
    assert lines[0].split("\t")[:3] == ["onset", "duration", "trial_type"]   # BIDS core still leads
    assert lines[0].split("\t")[-1] == "channel" and lines[1].split("\t")[-1] == "II"

    t = pq.read_table(ex.export_intervals_parquet(_intervals(), tmp_path / "c.parquet", channel="II"))
    assert t.column("channel").to_pylist() == ["II", "II", "II"]

    m = loadmat(ex.export_intervals_mat(_intervals(), tmp_path / "c.mat", channel="II"))
    cells = m["quality_intervals"]["channel"][0][0].ravel()      # MATLAB cell array of char rows
    assert [str(v[0]) for v in cells] == ["II"] * 3

    blank = list(csv.DictReader(ex.export_intervals_csv(_intervals(), tmp_path / "b.csv").open()))
    assert {r["channel"] for r in blank} == {""}      # no analysis context -> empty, not fabricated


def test_export_controller_writes_the_analyzed_channel_from_the_analysis_context(tmp_path):
    """The channel reaches the flat table through the REAL path the Coordinator drives: it stamps the
    SelectionController with the analysis identity (recording, graded channel, index, model, revision)
    and the ExportController reads it back. Before, only the JSON provenance and the WFDB ``chan``
    field carried it — the CSV named no channel at all."""
    from biosqa.viewmodels.export_controller import ExportController
    from biosqa.viewmodels.selection_controller import SelectionController

    ivs = _intervals()

    class _Seg:
        _all_intervals = ivs

    sel = SelectionController()
    # exactly the call Coordinator._bind_selection_context makes after inference on ["RESP", "II"]
    sel.set_context(recording="/data/rec2ch.hea", channel="II", channel_index=1,
                    model_version="v1", revision=3)

    ctl = ExportController()
    ctl.attach(_Seg(), sel, None)
    got = {}
    ctl.exportSucceeded.connect(lambda p: got.setdefault("p", p))
    ctl.exportToPath(str(tmp_path / "ctx.csv"), "csv")
    rows = list(csv.DictReader(Path(got["p"]).open()))
    assert rows and {r["channel"] for r in rows} == {"II"}


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


def test_local_path_over_url_forms(tmp_path):
    """F13: the QML FileDialog hands back a URL. The hand-rolled parser DROPPED a UNC URL's host, so
    an export aimed at \\\\nas01\\quality landed on the local disk instead of the network share."""
    from biosqa.viewmodels.export_controller import _local_path

    # THE bug this guards. Qt encodes a URL host in the //server/share form on EVERY platform, so this
    # is the one assertion that must hold everywhere.
    assert _local_path("file://nas01/quality/exports/run1.csv") == "//nas01/quality/exports/run1.csv"

    assert _local_path("file:///srv/exports/my%20run.csv") == "/srv/exports/my run.csv"   # percent-decoded
    assert _local_path(r"C:\exports\run1.csv") == r"C:\exports\run1.csv"      # plain path: untouched
    assert _local_path("/srv/exports/run1.csv") == "/srv/exports/run1.csv"

    # Drive letters are a WINDOWS concept, and Qt's leading-slash strip for them is #ifdef Q_OS_WIN
    # (qurl.cpp). On POSIX "file:///C:/x" legitimately denotes a path *named* "/C:/x", so asserting the
    # Windows form unconditionally asserts the platform rather than the behaviour -- which is exactly
    # how this test broke CI on ubuntu while passing locally.
    if sys.platform == "win32":
        assert _local_path("file:///C:/exports/run1.csv") == "C:/exports/run1.csv"
        assert _local_path("file:///C:/exports/my%20run.csv") == "C:/exports/my run.csv"


def test_wfdb_annotates_the_analyzed_channel(tmp_path):
    """WFDB annotations name a signal INDEX — it must be the channel inference graded, not 0."""
    wfdb = pytest.importorskip("wfdb")
    ex.export_intervals_wfdb(_intervals(), tmp_path / "rec.qual", fs=250.0, inference_channel=1)
    ann = wfdb.rdann(str(tmp_path / "rec"), "qual")
    assert list(ann.chan) == [1, 1, 1]


def test_csv_note_cannot_execute_as_a_spreadsheet_formula(tmp_path):
    """A reviewer note starting with = + - @ is EXECUTED when the CSV/TSV is opened in a spreadsheet.
    Neutralized for the spreadsheet writers only — json keeps the note verbatim (it is training data,
    not a cell)."""
    note = {0: "=cmd|'/c calc'!A1"}
    rows = list(csv.DictReader(ex.export_intervals_csv(_intervals(), tmp_path / "f.csv",
                                                       notes=note).open()))
    assert rows[0]["note"] == "'=cmd|'/c calc'!A1"
    tsv = ex.export_intervals_tsv(_intervals(), tmp_path / "f.tsv", notes=note).read_text()
    assert "\t'=cmd" in tsv
    doc = json.loads(Path(ex.export_intervals_json(_intervals(), tmp_path / "f.json",
                                                   notes=note)).read_text())
    assert doc["intervals"][0]["note"] == "=cmd|'/c calc'!A1"       # verbatim for the training sink


def test_failed_export_does_not_truncate_the_destination(tmp_path, monkeypatch):
    """Writers stage into a temp file in the same directory and os.replace() it: a write that fails
    part-way must leave neither a truncated file at the destination nor a stray staging file."""
    out = tmp_path / "keep.csv"
    ex.export_intervals_csv(_intervals(), out)
    good = out.read_text()

    class _BoomWriter:
        def __init__(self, *a, **kw):
            pass

        def writeheader(self):
            pass

        def writerows(self, rows):
            raise RuntimeError("disk full")

    monkeypatch.setattr(ex.csv, "DictWriter", _BoomWriter)
    with pytest.raises(RuntimeError):
        ex.export_intervals_csv(_intervals(), out)
    assert out.read_text() == good                 # the previous export survived intact
    assert not list(tmp_path.glob("*.part"))       # and nothing was left half-written


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
