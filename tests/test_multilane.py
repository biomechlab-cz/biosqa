"""Multi-lane plot: SignalViewController stacks visible channels into disjoint horizontal bands
of a fixed [0,1] axis, and setLaneChannels tracks the visible set."""
import numpy as np

from biosqa.viewmodels.signal_view_controller import SignalViewController


class _FakeSeries:
    """Captures what the controller would push to a QtCharts LineSeries."""
    def __init__(self):
        self.tw = None
        self.yw = None

    def clear(self):
        pass

    def replaceNp(self, tw, yw):
        self.tw = np.asarray(tw)
        self.yw = np.asarray(yw)


def _controller_with_two_lanes():
    c = SignalViewController()
    t = np.arange(2000, dtype=np.float64) / 100.0
    c._handle = object()
    c._channels = ["A", "B"]
    c._fs = 100.0
    c._n_samples = 2000
    c._duration_sec = 20.0
    c._view_start_sec, c._view_end_sec = 0.0, 20.0
    c._caches = {"A": (t, np.sin(t), -1.0, 1.0), "B": (t, 3.0 * np.sin(t), -3.0, 3.0)}
    c._lanes = ["A", "B"]
    return c


def test_two_lanes_map_to_disjoint_bands():
    c = _controller_with_two_lanes()
    sa, sb = _FakeSeries(), _FakeSeries()
    c._series_map = {"A": sa, "B": sb}
    c._refill_lane("A")
    c._refill_lane("B")
    # top lane (A) lives in the upper band, bottom lane (B) in the lower — no overlap
    assert sa.yw.min() >= 0.5 and sa.yw.max() <= 1.0
    assert sb.yw.min() >= 0.0 and sb.yw.max() <= 0.5
    # both normalized despite different raw amplitudes (1x vs 3x)
    assert (sa.yw.max() - sa.yw.min()) > 0.2 and (sb.yw.max() - sb.yw.min()) > 0.2


def test_single_lane_is_not_normalized():
    c = _controller_with_two_lanes()
    c._lanes = ["A"]
    s = _FakeSeries()
    c._series_map = {"A": s}
    c._refill_lane("A", force_y=True)
    # single lane keeps raw amplitude (~[-1,1]) and drives the shared auto-scaled Y range
    assert s.yw.min() < -0.5 and s.yw.max() > 0.5
    assert c.viewLo < 0 and c.viewHi > 0


def test_set_lane_channels_tracks_visible_set():
    c = _controller_with_two_lanes()
    c.setLaneChannels(["A"])                 # hide B
    assert c._lanes == ["A"] and c.laneCount == 1
    assert "B" not in c._caches               # freed (not the primary)
    c.setLaneChannels(["A", "B"])
    assert c.laneChannels == ["A", "B"] and c.laneCount == 2


def test_window_xy_decimation_keeps_a_spike():
    """The SECOND decimation (visible window -> WINDOW_POINTS) must keep extrema too: striding here
    would throw away the very spike the min/max plot cache was built to preserve."""
    c = SignalViewController()
    fs, n = 100.0, 200_000                    # >> WINDOW_POINTS -> the decimation branch runs
    t = np.arange(n, dtype=np.float64) / fs
    y = np.sin(t)
    spike_i = 123_457                         # off any plausible stride grid
    y[spike_i] = 9.0
    c._channels = ["A"]
    c._fs = fs
    c._n_samples = n
    c._duration_sec = n / fs
    c._view_start_sec, c._view_end_sec = 0.0, n / fs
    c._caches = {"A": (t, y, -1.0, 9.0)}
    c._lanes = ["A"]
    s = _FakeSeries()
    c._series_map = {"A": s}
    c._refill_lane("A", force_y=True)
    assert s.yw.size <= c.WINDOW_POINTS + 2   # still inside the per-frame point budget
    assert s.yw.max() == 9.0                  # ...and the spike is on screen
    assert np.all(np.diff(s.tw) >= 0)         # x stays monotone (min/max emitted in time order)
