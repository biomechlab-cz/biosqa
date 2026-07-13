"""Card selection + durable training-queue write for the reverse channel."""
import json

from PySide6.QtCore import QStandardPaths

from biosqa.inference.segmenter import QualityInterval
from biosqa.viewmodels.quality_segment_model import QualitySegmentModel
from biosqa.viewmodels.selection_controller import SelectionController


def _wired():
    m = QualitySegmentModel()
    m.load_intervals([
        QualityInterval(0.0, 10.0, "Q3", 0.9),
        QualityInterval(10.0, 20.0, "Q0", 0.4, ("motion",)),
    ])
    s = SelectionController()
    s.attach_segments(m)
    return m, s


def test_select_by_all_index():
    _, s = _wired()
    s.selectByAllIndex(1)
    assert s.selectedSegment is not None
    assert s.selectedSegment.tier == "Q0"
    assert s.selectedAllIndex == 1
    s.selectByAllIndex(99)          # out of range -> unchanged
    assert s.selectedAllIndex == 1


def test_training_queue_persists_corrected_tier():
    QStandardPaths.setTestModeEnabled(True)   # redirect AppDataLocation to a throwaway test dir
    _, s = _wired()
    s.selectByAllIndex(1)           # model graded Q0
    s.relabel("Q2")                 # reviewer corrects to Q2
    s.addNote("looks fine after all")
    path = s.saveToTrainingQueue()
    assert path
    rec = [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()][-1]
    assert rec["tier"] == "Q2"           # the CORRECTED tier, not the model's original
    assert rec["modelTier"] == "Q0" and rec["corrected"] is True
    assert rec["note"] == "looks fine after all"


def test_save_queue_no_selection_returns_empty():
    s = SelectionController()
    assert s.saveToTrainingQueue() == ""


def test_context_change_drops_reviews():
    """A review belongs to ONE (recording, channel, model, revision). Switching recording — or
    re-segmenting the same one — drops it wholesale; it is never re-anchored to whatever now happens
    to start at the same second (the override store was keyed by time alone and lived forever)."""
    _, s = _wired()
    s.set_context(recording="A.hea", channel="II", channel_index=0, model_version="v1", revision=1)
    s.selectByAllIndex(0)
    s.relabel("Q1")
    s.addNote("A's note")
    assert len(s.collected_overrides()) == 1

    dropped = s.set_context(recording="B.hea", channel="II", channel_index=0,
                            model_version="v1", revision=1)   # a DIFFERENT recording
    assert dropped == 1
    assert s.collected_overrides() == [] and s.selectedSegment is None

    s.selectByAllIndex(0)
    s.relabel("Q1")
    dropped = s.set_context(recording="B.hea", channel="II", channel_index=0,
                            model_version="v1", revision=2)   # same recording, RE-SEGMENTED
    assert dropped == 1 and s.collected_overrides() == []


def test_training_row_carries_recording_identity():
    """The active-learning sink is unattributable without it: a (start, end) pair alone matches a
    segment in every recording ever opened."""
    QStandardPaths.setTestModeEnabled(True)
    _, s = _wired()
    s.set_context(recording="/data/recA.hea", channel="II", channel_index=1,
                  model_version="ecg-v2", revision=3)
    s.selectByAllIndex(1)
    s.relabel("Q2")
    path = s.saveToTrainingQueue()
    rec = [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()][-1]
    assert rec["recording"] == "/data/recA.hea" and rec["channel"] == "II"
    assert rec["modelVersion"] == "ecg-v2" and rec["revision"] == 3
    assert rec["timestamp"].startswith("20") and rec["timestamp"].endswith("+00:00")
