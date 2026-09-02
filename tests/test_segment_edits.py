"""Boundary editor: structural segment edits (move/split/merge/reclassify) on the segment model
and via the SelectionController's QML slots."""
from biosqa.inference.segmenter import QualityInterval as QI
from biosqa.viewmodels.quality_segment_model import QualitySegmentModel
from biosqa.viewmodels.selection_controller import SelectionController


def _model(intervals):
    m = QualitySegmentModel()
    m.load_intervals(intervals)
    return m


def test_move_boundary_reassigns_sliver():
    m = _model([QI(0, 10, "Q0", 0.4, ()), QI(10, 20, "Q2", 0.9, ())])
    assert m.move_boundary(0, 15) is True
    assert m._all_intervals[0].end_sec == 15 and m._all_intervals[1].start_sec == 15
    assert m._all_intervals[0].tier == "Q0" and m._all_intervals[1].tier == "Q2"


def test_move_boundary_clamps():
    m = _model([QI(0, 10, "Q0", 0.4, ()), QI(10, 20, "Q2", 0.9, ())])
    assert m.move_boundary(0, 1e6) is True
    assert abs(m._all_intervals[0].end_sec - (20 - m.MIN_SEGMENT_SEC)) < 1e-6
    assert m.move_boundary(5, 3) is False        # no boundary after the last index


def test_split_interval():
    m = _model([QI(0, 20, "Q1", 0.5, ("motion",))])
    assert m.split_interval(0, 8) == 1
    assert len(m._all_intervals) == 2
    assert m._all_intervals[0].end_sec == 8 and m._all_intervals[1].start_sec == 8
    assert m._all_intervals[0].tier == m._all_intervals[1].tier == "Q1"
    assert m.split_interval(0, 0.01) == -1        # too close to the edge


def test_merge_with_next_keeps_longer_tier():
    m = _model([QI(0, 18, "Q0", 0.4, ("motion",)), QI(18, 20, "Q2", 0.9, ("baseline",))])
    assert m.merge_with_next(0) is True
    assert len(m._all_intervals) == 1
    mm = m._all_intervals[0]
    assert mm.start_sec == 0 and mm.end_sec == 20 and mm.tier == "Q0"   # Q0 is the longer piece
    assert "motion" in mm.artifacts and "baseline" in mm.artifacts


def test_set_tier():
    m = _model([QI(0, 10, "Q0", 0.4, ())])
    assert m.set_tier(0, "Q2") is True and m._all_intervals[0].tier == "Q2"
    assert m.set_tier(0, "") is False


def test_selection_nudge_and_reclassify():
    m = _model([QI(0, 10, "Q0", 0.4, ()), QI(10, 20, "Q2", 0.9, ())])
    sel = SelectionController()
    sel.attach_segments(m)
    sel.selectByAllIndex(1)                       # select the Q2 segment
    sel.nudgeSelectedStart(-3.0)                  # boundary 10 -> 7: Q0 grows, Q2 shrinks
    assert m._all_intervals[0].end_sec == 7 and m._all_intervals[1].start_sec == 7
    assert sel.selectedSegment.startSec == 7      # selection stayed on the (edited) Q2 piece
    sel.reclassifySelected("Q1")                  # updates the band AND queues the correction
    assert m._all_intervals[1].tier == "Q1"
    assert sel.selectedSegment.tier == "Q1"
    assert len(sel.collected_overrides()) == 1


def test_selection_split_and_merge():
    m = _model([QI(0, 20, "Q0", 0.4, ())])
    sel = SelectionController()
    sel.attach_segments(m)
    sel.selectByAllIndex(0)
    sel.splitSelected()                           # midpoint -> two pieces
    assert m._total_count() == 2
    assert sel.selectedSegment.endSec == 10       # left piece kept selected
    sel.mergeSelectedNext()                       # merge them back
    assert m._total_count() == 1 and sel.selectedSegment.endSec == 20


def test_next_poor_index():
    m = _model([QI(0, 10, "Q3", 0.9, ()), QI(10, 20, "Q1", 0.5, ()),
                QI(20, 30, "Q3", 0.9, ()), QI(30, 40, "Q0", 0.3, ())])
    assert m.nextPoorIndex(0.0) == 1        # first poor after t=0 is the Q1 at index 1
    assert m.nextPoorIndex(10.0) == 3       # after t=10, next poor is the Q0 at index 3
    assert m.nextPoorIndex(30.0) == -1      # nothing poor starts after t=30


def test_flag_for_review_persists(tmp_path, monkeypatch):
    import json

    import biosqa.viewmodels.selection_controller as sc
    monkeypatch.setattr(sc, "_training_queue_path", lambda: tmp_path / "q.jsonl")
    m = _model([QI(0, 10, "Q0", 0.4, ())])
    sel = SelectionController()
    sel.attach_segments(m)
    sel.selectByAllIndex(0)
    path = sel.flagForReview("looks noisy")
    assert path and (tmp_path / "q.jsonl").exists()
    row = json.loads((tmp_path / "q.jsonl").read_text().strip())
    assert row["flagged"] is True and row["note"] == "looks noisy" and row["tier"] == "Q0"
    assert sel.selectedSegment.flagged is True


def test_first_segment_start_nudge_is_noop():
    m = _model([QI(0, 10, "Q0", 0.4, ()), QI(10, 20, "Q2", 0.9, ())])
    sel = SelectionController()
    sel.attach_segments(m)
    sel.selectByAllIndex(0)
    sel.nudgeSelectedStart(-2.0)                  # no previous segment to reassign to
    assert m._all_intervals[0].start_sec == 0
