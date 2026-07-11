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
