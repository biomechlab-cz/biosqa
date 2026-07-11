"""RecordingListModel modality auto-detect confidence gate: a low-confidence AUTO vote (effectively a
guess) must warn the user, while a confident vote or an explicitly-forced modality must not."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

import biosqa.viewmodels.recording_controller as rc

_app = QApplication.instance() or QApplication([])


class _Handle:
    fs_hz = {"ch": 250.0}
    n_samples = {"ch": 1000}


def _spy(signal):
    calls = []
    signal.connect(lambda *a: calls.append(a))
    return calls


def _model(monkeypatch, vote):
    monkeypatch.setattr(rc, "open_recording", lambda p: _Handle())
    monkeypatch.setattr(rc, "modality_vote", lambda h: vote)
    return rc.RecordingListModel()


def test_low_confidence_auto_detect_warns(monkeypatch):
    m = _model(monkeypatch, ("eeg", 0.3))                     # below the ~0.35 fs-tiebreak floor
    uncertain, mismatch = _spy(m.modalityUncertain), _spy(m.modalityMismatch)
    m.open("/fake/a.dat")                                     # AUTO path (no forced modality)
    assert len(uncertain) == 1 and uncertain[0][0] == "eeg" and abs(uncertain[0][1] - 0.3) < 1e-6
    assert mismatch == []                                     # not a forced mismatch


def test_confident_auto_detect_is_silent(monkeypatch):
    m = _model(monkeypatch, ("ecg", 0.9))
    uncertain = _spy(m.modalityUncertain)
    m.open("/fake/b.dat")
    assert uncertain == []                                    # confident → no warning


def test_forced_modality_suppresses_uncertain_warning(monkeypatch):
    m = _model(monkeypatch, ("eeg", 0.2))                     # low-confidence vote...
    uncertain = _spy(m.modalityUncertain)
    m.open("/fake/c.dat", "ecg")                              # ...but the user forced ecg → no auto-warn
    assert uncertain == []


def test_handle_cache_is_lru_bounded_and_closes_evicted(monkeypatch):
    """The open-handle cache must be LRU-bounded and close evicted backends — MNE keeps a file descriptor
    open per handle, so an unbounded cache leaks fds over a long session."""
    closed = []

    class _Backend:
        def close(self):
            closed.append(1)

    class _H:
        def __init__(self):
            self.fs_hz = {"ch": 250.0}
            self.n_samples = {"ch": 1000}
            self.backend = _Backend()

    handles = iter([_H() for _ in range(4)])
    monkeypatch.setattr(rc, "modality_vote", lambda h: ("ecg", 0.9))
    monkeypatch.setattr(rc, "open_recording", lambda p: next(handles))
    monkeypatch.setattr(rc, "_MAX_OPEN_HANDLES", 3)           # small cap for the test
    m = rc.RecordingListModel()
    for i in range(4):
        m.open(f"/fake/rec{i}.dat")
    assert len(m._handles) == 3                               # bounded, not 4
    assert closed == [1]                                     # the evicted (oldest) backend was closed
