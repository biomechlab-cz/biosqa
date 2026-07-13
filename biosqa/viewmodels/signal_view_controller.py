"""View window + per-lane QtCharts trace binding for the signal plot (Plan 2 §3/§9).

Owns the visible window (t0, t1) and the multi-lane QtCharts render path: it binds one
``QLineSeries`` per drawn channel (``loadTraceFor``) and re-fills each from an in-memory full-res
cache on pan/zoom (``setView`` → ``_refill_lane`` → ``replaceNp``). Non-primary lane caches build
off-thread (``ChannelCacheTask``). ``curveForRange`` serves the segment-inspector / grid mini-plots
via a min/max envelope of a bounded window.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCharts import QLineSeries
from PySide6.QtCore import Property, QObject, QThreadPool, Signal, Slot

from biosqa.io.loaders import read_window
from biosqa.io.pyramid import minmax_envelope_indices

#: max samples pulled for a miniature/zoom envelope on the disk-read fallback path (the cache-slice path
#: has no such read); a preview decimates to <=800 buckets so more resolution than this is wasted.
_PREVIEW_CAP = 20000


def _minmax_envelope(x: np.ndarray, y: np.ndarray, n_buckets: int):
    """Bucketed min/max envelope -> (x, y_min, y_max), each length ~``n_buckets``.

    The correct decimation for a signal *plot* (unlike a single-series LTTB): each
    horizontal bucket keeps both extremes, so spikes/artifacts survive at any zoom.
    """
    n = y.shape[0]
    if n <= n_buckets * 2 or n_buckets < 1:
        return x.astype(np.float64), y.astype(np.float32), y.astype(np.float32)
    edges = np.linspace(0, n, n_buckets + 1).astype(int)
    xs, ymin, ymax = [], [], []
    for i in range(n_buckets):
        a, b = edges[i], edges[i + 1]
        if b <= a:
            continue
        seg = y[a:b]
        xs.append(x[a]); ymin.append(float(seg.min())); ymax.append(float(seg.max()))
    return (np.asarray(xs, dtype=np.float64),
            np.asarray(ymin, dtype=np.float32), np.asarray(ymax, dtype=np.float32))


class SignalViewController(QObject):
    """Owns the current view window (t0, t1) and LOD level for the plot canvas."""

    viewStartSecChanged = Signal()
    viewEndSecChanged = Signal()
    durationSecChanged = Signal()
    traceRangeChanged = Signal()   # the primary channel's amplitude range (QtCharts Y axis)
    laneLayoutChanged = Signal()   # the set of drawn (visible) channel lanes changed

    #: cap on simultaneously-plotted lanes (high-density EEG can have 100+ channels; the
    #: channel-list panel still shows all, but decimating every lane per pan is wasteful).
    MAX_PLOT_LANES = 16

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._view_start_sec = 0.0
        self._view_end_sec = 10.0
        # open-recording state for the plot (set by the Coordinator via set_recording)
        self._handle = None
        self._channels: list[str] = []   # lanes to draw (capped for high-density recordings)
        self._fs = 1.0
        self._n_samples = 0
        self._duration_sec = 0.0
        self._trace_lo = -1.0          # full-channel amplitude range (fallback)
        self._trace_hi = 1.0
        self._series = None            # the QtCharts LineSeries (held so pan can re-fill it)
        self._full_t = None            # full-resolution (t, y) cache of the primary channel
        self._full_y = None
        self._view_lo = -1.0           # amplitude range of the VISIBLE window (QtCharts Y axis)
        self._view_hi = 1.0
        # multi-lane: one QtCharts series + one full-res cache per DRAWN (visible) channel. When
        # >1 lane is drawn each is normalized into its own horizontal band of a fixed [0,1] Y axis
        # (stacked view); a single lane keeps the auto-scaled viewLo/viewHi behaviour.
        self._caches: dict[str, tuple] = {}          # channel -> (full_t, full_y, lo, hi)
        self._series_map: dict[str, "QLineSeries"] = {}   # channel -> bound QtCharts series
        self._lanes: list[str] = []                  # ordered visible channels drawn (<= MAX_PLOT_LANES)
        self._pool = QThreadPool.globalInstance()    # non-primary lane caches build off-thread
        self._cache_carriers: list = []              # keep ChannelCacheWorkerSignals alive
        self._load_gen = 0                           # bumped per recording; drops stale lane reads
        self._pending_channels: set = set()          # lanes whose off-thread read is in flight (dedup)

    # -- properties ---------------------------------------------------------
    def _get_view_start_sec(self) -> float:
        return self._view_start_sec

    viewStartSec = Property(float, _get_view_start_sec, notify=viewStartSecChanged)

    def _get_view_end_sec(self) -> float:
        return self._view_end_sec

    viewEndSec = Property(float, _get_view_end_sec, notify=viewEndSecChanged)

    def _get_duration_sec(self) -> float:
        return self._duration_sec

    durationSec = Property(float, _get_duration_sec, notify=durationSecChanged)

    def _get_trace_lo(self) -> float:
        return self._trace_lo

    def _get_trace_hi(self) -> float:
        return self._trace_hi

    traceLo = Property(float, _get_trace_lo, notify=traceRangeChanged)
    traceHi = Property(float, _get_trace_hi, notify=traceRangeChanged)

    def _get_view_lo(self) -> float:
        return self._view_lo

    def _get_view_hi(self) -> float:
        return self._view_hi

    viewLo = Property(float, _get_view_lo, notify=traceRangeChanged)
    viewHi = Property(float, _get_view_hi, notify=traceRangeChanged)

    def _lane_count(self) -> int:
        return len(self._lanes)

    laneCount = Property(int, _lane_count, notify=laneLayoutChanged)

    def _lane_channels(self):
        return list(self._lanes)

    laneChannels = Property("QVariant", _lane_channels, notify=laneLayoutChanged)

    #: max points fed to the QtCharts series per frame — viewport windowing keeps this small
    #: (only the visible slice) so the scene-graph re-tessellation stays cheap => high FPS.
    WINDOW_POINTS = 2200

    def _window_xy(self, full_t, full_y):
        """Decimate the current view window (+ a small margin) of a full-res cache to
        ``<= WINDOW_POINTS`` points → ``(tw, yw)`` float arrays, or ``(None, None)``.

        This is the SECOND decimation (the cache itself is the first): it too keeps a min/max bucket
        envelope rather than striding, so a spike the cache preserved is not thrown away here instead.
        Both stages must keep extrema or neither is worth anything."""
        if full_t is None or full_y is None or full_t.size < 2:
            return None, None
        a, b = self._view_start_sec, self._view_end_sec
        span = max(b - a, 1e-6)
        margin = span * 0.12
        i0 = max(0, int(np.searchsorted(full_t, a - margin, side="left")))
        i1 = max(i0 + 2, int(np.searchsorted(full_t, b + margin, side="right")))
        if i1 - i0 > self.WINDOW_POINTS:
            # Bucket width from the cache's MEAN point density (the envelope cache is not uniformly
            # spaced, so full_t[1]-full_t[0] is not its dt) and buckets absolutely aligned (multiples
            # of `spb`) so the bucket phase is stable while panning — a window-relative grid would
            # make the trace shimmer as you scroll.
            density = full_t.size / max(float(full_t[-1] - full_t[0]), 1e-12)   # cache points / sec
            approx_n = span * 1.24 * density
            spb = max(2, int(np.ceil(approx_n / max(1, self.WINDOW_POINTS // 2))))
            a0 = i0 - (i0 % spb)
            seg_t, seg_y = full_t[a0:i1], full_y[a0:i1]
            idx = minmax_envelope_indices(seg_y, spb)
            return (np.ascontiguousarray(seg_t[idx], dtype=np.float64),
                    np.ascontiguousarray(seg_y[idx], dtype=np.float64))
        return (np.ascontiguousarray(full_t[i0:i1], dtype=np.float64),
                np.ascontiguousarray(full_y[i0:i1], dtype=np.float64))

    @staticmethod
    def _robust_range(tw, yw, a, b):
        """Robust visible amplitude range (0.5/99.5 percentiles so a lone spike can't jerk it)."""
        vis = yw[(tw >= a) & (tw <= b)]
        if vis.size < 2:
            vis = yw
        lo = float(np.percentile(vis, 0.5))
        hi = float(np.percentile(vis, 99.5))
        if hi <= lo:
            lo, hi = float(vis.min()), float(vis.max())
        return lo, (hi if hi > lo else lo + 1.0)

    def _refill_lane(self, channel: str, force_y: bool = False) -> None:
        """Re-fill one channel's series from its cache over the visible window. A single lane
        auto-scales the shared Y axis (viewLo/viewHi); with >1 lane each is normalized into its
        own horizontal band of a fixed [0,1] axis so the channels stack without overlapping."""
        series = self._series_map.get(channel)
        cache = self._caches.get(channel)
        if series is None or cache is None:
            return
        tw, yw = self._window_xy(cache[0], cache[1])
        if tw is None or tw.size < 2:
            return
        a, b = self._view_start_sec, self._view_end_sec
        lo, hi = self._robust_range(tw, yw, a, b)
        n_lanes = max(1, len(self._lanes))
        if n_lanes == 1:
            ymapped = yw
            pad = (hi - lo) * 0.12
            new_lo, new_hi = lo - pad, hi + pad
            cur = self._view_hi - self._view_lo
            if (force_y or cur <= 0 or abs(new_lo - self._view_lo) > 0.12 * cur
                    or abs(new_hi - self._view_hi) > 0.12 * cur):
                self._view_lo, self._view_hi = new_lo, new_hi
                self.traceRangeChanged.emit()
        else:
            k = self._lanes.index(channel) if channel in self._lanes else 0
            center = 1.0 - (k + 0.5) / n_lanes          # top lane first
            half = (0.5 / n_lanes) * 0.82               # leave a gap between lanes
            ynorm = (yw - lo) / (hi - lo if hi > lo else 1.0)
            ymapped = np.ascontiguousarray(center - half + ynorm * (2.0 * half), dtype=np.float64)
        try:
            series.replaceNp(tw, ymapped)   # replaceNp (NOT appendNp — that hangs on GPU)
        except RuntimeError:
            # series destroyed under us (view swap / shutdown) — drop the dangling ref; a fresh
            # WaveformChart re-binds via loadTraceFor when the view returns.
            self._series_map.pop(channel, None)

    def _refill_window(self, force_y: bool = False) -> None:
        """Re-fill every bound lane over the visible window (pan/zoom hot path)."""
        for ch in list(self._series_map.keys()):
            self._refill_lane(ch, force_y=force_y)

    @Slot(QLineSeries)
    def loadTrace(self, series: QLineSeries) -> None:  # noqa: N802
        """Back-compat: bind a series to the PRIMARY channel (single-lane callers)."""
        ch = self._channels[0] if self._channels else ""
        self.loadTraceFor(series, ch)

    @Slot(QLineSeries, str)
    def loadTraceFor(self, series: QLineSeries, channel: str) -> None:  # noqa: N802
        """Bind a QtCharts LineSeries to ``channel`` and draw the current window. The primary
        channel's cache is seeded off-thread (set_recording_cached); a non-primary lane's cache is
        built OFF-THREAD (a whole-channel read must never block the GUI — P1), so the lane draws
        once its cache arrives."""
        if not channel:
            return
        series.clear()
        self._series_map[channel] = series
        if channel not in self._lanes:
            self._lanes.append(channel)
        primary = self._channels[0] if self._channels else None
        if channel in self._caches:
            self._refill_lane(channel, force_y=True)
            return
        if channel == primary and self._full_t is not None and self._full_t.size >= 2:
            self._caches[channel] = (self._full_t, self._full_y, self._trace_lo, self._trace_hi)
            self._refill_lane(channel, force_y=True)
            return
        if self._handle is not None and self._fs > 0 and channel not in self._pending_channels:
            from biosqa.workers.qt_threads import ChannelCacheTask
            from biosqa.workers.signals import ChannelCacheWorkerSignals
            self._pending_channels.add(channel)      # dedup: _rebuildLanes re-invokes this per layout change
            carrier = ChannelCacheWorkerSignals()
            carrier._gen = self._load_gen  # type: ignore[attr-defined]
            carrier.ready.connect(self._on_channel_cache)
            self._cache_carriers.append(carrier)
            self._pool.start(ChannelCacheTask(self._handle, channel, self._fs, carrier))

    @Slot(str, object, object, float, float)
    def _on_channel_cache(self, channel: str, full_t, full_y, lo: float, hi: float) -> None:
        """A non-primary lane's cache finished building off-thread — store it and draw the lane
        (unless it was toggled off, or a NEW recording opened while the read was in flight)."""
        self._pending_channels.discard(channel)
        if getattr(self.sender(), "_gen", self._load_gen) != self._load_gen:
            return                                    # stale read from a superseded recording
        if channel not in self._lanes:
            return
        self._caches[channel] = (full_t, full_y, float(lo), float(hi))
        self._refill_lane(channel, force_y=True)

    @Slot(list)
    def setLaneChannels(self, channels: list) -> None:  # noqa: N802
        """Set the ordered list of VISIBLE channels to draw (from the channel-list toggles). Drops
        series/caches for channels no longer shown, then re-lays-out; QML re-binds via loadTraceFor.

        A lane is a DRAWN TRACE, so with NO recording bound there are no lanes. That is not pedantry:
        the channel list is populated for a FAILED open too (the channels exist; nothing was graded),
        and ``SignalView.qml`` mirrors it into lanes on ``countChanged`` — which left an empty lane
        placeholder, with no handle and no cache, for a recording that was never analysed. The plot on
        a failed open must be genuinely empty, not empty-looking. The real binding paths
        (:meth:`set_recording` / :meth:`set_recording_cached`) set ``_lanes`` themselves once a handle
        exists, so this costs the normal open nothing."""
        lanes = [str(c) for c in channels if c][: self.MAX_PLOT_LANES]
        if self._handle is None:
            lanes = []
        if lanes == self._lanes:
            return
        self._lanes = lanes
        for ch in list(self._series_map.keys()):
            if ch not in lanes:
                self._series_map.pop(ch, None)
        for ch in list(self._caches.keys()):        # keep the primary cache; free the rest
            if ch not in lanes and ch != (self._channels[0] if self._channels else None):
                self._caches.pop(ch, None)
        self.laneLayoutChanged.emit()
        self._refill_window(force_y=True)

    @Slot(float, float)
    def setView(self, start_sec: float, end_sec: float) -> None:  # noqa: N802
        """Set the visible window IMMEDIATELY (no debounce / no disk read) — the QtCharts
        X axis binds to viewStartSec/viewEndSec, so pan/zoom is realtime against the loaded series."""
        if end_sec <= start_sec:
            return
        start_sec = max(0.0, start_sec)
        if self._duration_sec > 0:
            end_sec = min(end_sec, self._duration_sec)
            start_sec = min(start_sec, max(0.0, end_sec - 0.05))
        if start_sec != self._view_start_sec:
            self._view_start_sec = start_sec
            self.viewStartSecChanged.emit()
        if end_sec != self._view_end_sec:
            self._view_end_sec = end_sec
            self.viewEndSecChanged.emit()
        self._refill_window()   # re-window the visible slice (pan/zoom hot path)

    # -- recording binding (called by the Coordinator on open) ---------------
    def clear(self) -> None:
        """Drop EVERY trace of the open recording, leaving an honestly EMPTY plot.

        The Coordinator calls this on each open, BEFORE anything about the new recording can fail
        (``Coordinator._invalidate``). Without it a failed open (e.g. a modality whose model isn't in
        ``models/``) left the PREVIOUS recording's waveform drawn and its handle bound — so
        ``valueAt``/``curveForRange``/zoom kept serving recording A's samples while the title, channel
        list, modality and fs already said B. An empty plot beats a plausible wrong one.
        """
        for series in list(self._series_map.values()):
            try:
                series.clear()          # erase the drawn points now, not on the next QML relayout
            except RuntimeError:        # series already destroyed (view swap / shutdown)
                pass
        self._handle = None
        self._channels = []
        self._lanes = []
        self._fs = 1.0
        self._n_samples = 0
        self._duration_sec = 0.0
        self._full_t = self._full_y = None
        self._trace_lo, self._trace_hi = -1.0, 1.0
        self._view_lo, self._view_hi = -1.0, 1.0
        self._caches = {}
        self._series_map = {}
        self._cache_carriers = []
        self._pending_channels = set()
        self._load_gen += 1             # any in-flight lane read now belongs to a superseded recording
        self._view_start_sec = 0.0
        self._view_end_sec = 10.0
        self.durationSecChanged.emit()
        self.viewStartSecChanged.emit()
        self.viewEndSecChanged.emit()
        self.traceRangeChanged.emit()
        self.laneLayoutChanged.emit()   # QML tears the (now empty) lane pool down

    def set_recording(self, handle, channels, fs: float) -> None:
        """Bind the plot to a recording's channels and draw the head window.

        ``channels`` may be a single name or a list; up to ``MAX_PLOT_LANES`` are drawn
        (the channel-list panel still lists them all).
        """
        if isinstance(channels, str):
            channels = [channels]
        self._handle = handle
        self._channels = list(channels)[: self.MAX_PLOT_LANES]
        self._fs = float(fs) or 1.0
        ref = self._channels[0] if self._channels else None
        self._n_samples = int(handle.n_samples.get(ref, 0)) if ref else 0
        self._duration_sec = self._n_samples / self._fs if self._fs else 0.0
        # reset per-lane caches/series; the primary cache is built lazily by loadTraceFor here
        self._full_t = self._full_y = None
        self._caches = {}
        self._series_map = {}
        self._cache_carriers = []
        self._pending_channels = set()
        self._load_gen += 1
        self._lanes = [ref] if ref else []
        self.laneLayoutChanged.emit()
        # set the initial window BEFORE durationSecChanged: the WaveformChart reloads on that
        # signal, and loadTraceFor -> _refill_lane must window against the correct 0..30s view.
        self._view_start_sec = 0.0
        self._view_end_sec = min(30.0, self._duration_sec) if self._duration_sec > 0 else 10.0
        self.durationSecChanged.emit()
        self.viewStartSecChanged.emit()
        self.viewEndSecChanged.emit()

    def set_recording_cached(self, handle, channels, fs: float, full_t, full_y,
                             trace_lo: float, trace_hi: float, n_samples: int) -> None:
        """Like :meth:`set_recording` but with a PRECOMPUTED primary-channel cache (built off-thread
        by ``LoadResampleTask``), so the full-channel read no longer runs on the GUI thread. The
        per-lane VISIBLE-window decimation still runs inline (it reads only the small visible slice)."""
        if isinstance(channels, str):
            channels = [channels]
        self._handle = handle
        self._channels = list(channels)[: self.MAX_PLOT_LANES]
        self._fs = float(fs) or 1.0
        self._n_samples = int(n_samples)
        self._duration_sec = self._n_samples / self._fs if self._fs else 0.0
        self._full_t = full_t                 # precomputed primary cache — loadTraceFor draws from it
        self._full_y = full_y
        self._trace_lo, self._trace_hi = float(trace_lo), float(trace_hi)
        # reset per-lane caches/series for the new recording; seed the primary from the pushed cache
        primary = self._channels[0] if self._channels else None
        self._caches = {}
        self._series_map = {}
        self._cache_carriers = []
        self._pending_channels = set()
        self._load_gen += 1
        if primary is not None and full_t is not None:
            self._caches[primary] = (full_t, full_y, float(trace_lo), float(trace_hi))
        self._lanes = [primary] if primary else []
        self._view_start_sec = 0.0
        self._view_end_sec = min(30.0, self._duration_sec) if self._duration_sec > 0 else 10.0
        self.durationSecChanged.emit()
        self.viewStartSecChanged.emit()
        self.viewEndSecChanged.emit()
        self.traceRangeChanged.emit()
        self.laneLayoutChanged.emit()         # QML rebuilds the lane pool + re-binds via loadTraceFor
        self._refill_window(force_y=True)

    @Slot(float, float, result="QVariant")
    def curveForRange(self, start_sec: float, end_sec: float):  # noqa: N802
        """Decimated min/max envelope of the primary channel over [start, end] seconds, as
        ``{x, ymin, ymax, lo, hi}`` lists — for the Segment Inspector's zoomed waveform and the segment-grid
        minis. Prefers the in-memory full-resolution cache the trace already draws from: slicing it is a pure
        numpy op, whereas a fresh ``read_window`` here is a SYNCHRONOUS disk read on the GUI thread (a
        multi-minute span is a ~1e5–1e6-sample read *per card* — the segment grid called this once per
        delegate). Falls back to a bounded disk read only until the cache is populated."""
        if not self._channels or self._fs <= 0:
            return None
        ch = self._channels[0]
        cache = self._caches.get(ch)
        if cache is not None and cache[0] is not None and getattr(cache[0], "size", 0) >= 2:
            ft, fy = cache[0], cache[1]
            i0 = int(np.searchsorted(ft, start_sec, side="left"))
            i1 = int(np.searchsorted(ft, end_sec, side="right"))
            i0 = max(0, min(i0, fy.size - 1))
            i1 = max(i0 + 2, min(i1, fy.size))
            x = np.asarray(ft[i0:i1], dtype=np.float64)
            raw = np.asarray(fy[i0:i1], dtype=np.float32)
        else:
            if self._handle is None:
                return None
            s0 = max(0, int(round(start_sec * self._fs)))
            s1 = min(self._n_samples, int(round(end_sec * self._fs)))
            if s1 - s0 < 2:
                return None
            # bound the fallback read: stride so a huge span never pulls more than ~_PREVIEW_CAP samples
            # for a miniature envelope (the disk read is still contiguous, but the array/decimation stays
            # bounded and the cache-slice path above handles the common case with no I/O at all).
            step = max(1, (s1 - s0) // _PREVIEW_CAP)
            try:
                raw = np.asarray(read_window(self._handle, [ch], s0, s1),
                                 dtype=np.float32).reshape(-1)[::step]
            except Exception:  # noqa: BLE001
                return None
            x = (s0 + np.arange(raw.shape[0], dtype=np.float64) * step) / self._fs
        if raw.shape[0] < 2:
            return None
        n_buckets = max(2, min(raw.shape[0] // 2, 800))
        xm, ymin, ymax = _minmax_envelope(x, raw, n_buckets)
        lo, hi = float(ymin.min()), float(ymax.max())
        if hi <= lo:
            hi = lo + 1.0
        return {"x": xm.tolist(), "ymin": ymin.tolist(), "ymax": ymax.tolist(),
                "lo": lo, "hi": hi}

    @Slot(float, result=float)
    def valueAt(self, sec: float) -> float:  # noqa: N802
        """Primary-lane amplitude at ``sec`` for the hover tooltip, read from the SAME full-res
        cache the QtCharts trace draws from (was reading the legacy decimation curve, which the
        active path no longer populates — so the tooltip went stale/zero after panning). P3 fix."""
        if not self._channels:
            return 0.0
        cache = self._caches.get(self._channels[0])
        if cache is None:
            return 0.0
        ft, fy = cache[0], cache[1]
        if ft is None or getattr(ft, "size", 0) < 1:
            return 0.0
        i = int(np.searchsorted(ft, sec))
        i = max(0, min(i, fy.size - 1))
        return float(fy[i])

