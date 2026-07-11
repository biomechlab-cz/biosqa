"""Viewport segment-card queries on the segment model."""
from biosqa.inference.segmenter import QualityInterval
from biosqa.viewmodels.quality_segment_model import QualitySegmentModel


def _model():
    m = QualitySegmentModel()
    m.load_intervals([
        QualityInterval(0.0, 10.0, "Q3", 0.9),
        QualityInterval(10.0, 20.0, "Q0", 0.4, ("motion",)),
        QualityInterval(20.0, 35.0, "Q2", 0.7),
    ])
    return m


def test_segments_in_range_overlap():
    m = _model()
    out = m.segmentsInRange(12.0, 22.0)          # overlaps intervals 1 and 2
    assert [c["index"] for c in out] == [1, 2]
    assert out[0]["tier"] == "Q0" and out[0]["artifacts"] == ["motion"]
    assert out[0]["startSec"] == 10.0 and out[0]["endSec"] == 20.0
    assert [c["index"] for c in m.segmentsInRange(1.0, 5.0)] == [0]   # fully inside interval 0
    assert m.segmentsInRange(-5.0, -1.0) == []                        # before all -> empty
    assert m.segmentsInRange(0.0, 35.0) and len(m.segmentsInRange(0.0, 35.0)) == 3


def test_all_index_helpers():
    m = _model()
    ivs = m._all_intervals
    assert m.interval_at_all(1) is ivs[1]
    assert m.interval_at_all(99) is None
    assert m.all_index_of(ivs[2]) == 2
    assert m.filtered_row_of_all(2) == 2         # under the default "all" filter, rows == full index
