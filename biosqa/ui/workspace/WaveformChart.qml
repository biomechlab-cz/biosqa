import QtQuick
import QtCharts
import "../"
import "../components"

// GPU-composited waveform (QtCharts LineSeries) for the primary channel. The whole
// channel is loaded into the series ONCE (signalView.loadTrace); pan/zoom just move the
// X axis over it via signalView.setView() — realtime, no per-frame disk read. Chart
// margins are 0 so the plot fills this Item, and overlays/bands map over the Item's own
// width/height (chart.plotArea isn't reliable across render backends).
Item {
    id: root
    objectName: "waveformChart"
    property string tool: "pan"
    property real playheadSec: NaN

    readonly property real viewStart: signalView.viewStartSec
    readonly property real viewEnd: signalView.viewEndSec
    readonly property real span: Math.max(viewEnd - viewStart, 1e-9)

    // One QtCharts LineSeries per VISIBLE channel lane, created imperatively — a Repeater's
    // dynamically-created series aren't adopted by ChartView, so we createSeries() in JS.
    // Rebuilt whenever the drawn-lane set changes (channel-visibility toggles) or a recording opens.
    property var _laneSeries: ({})
    readonly property var _lanePalette: ["#8FE3D0", "#6E8BFF", "#E0A32E", "#C08CF2",
                                         "#FF7A85", "#35D0BA", "#9AD07A", "#E56A9B"]
    function _laneColor(i) { return root._lanePalette[i % root._lanePalette.length] }

    function _rebuildLanes() {
        var chans = signalView.laneChannels || []
        for (var ch in root._laneSeries) {                 // remove lanes no longer drawn
            if (chans.indexOf(ch) < 0) {
                chart.removeSeries(root._laneSeries[ch])
                delete root._laneSeries[ch]
            }
        }
        for (var i = 0; i < chans.length; i++) {           // create + (re)bind current lanes
            var c = chans[i]
            var s = root._laneSeries[c]
            if (!s) {
                s = chart.createSeries(ChartView.SeriesTypeLine, c, axX, axY)
                s.useOpenGL = false
                s.width = 1
                root._laneSeries[c] = s
            }
            s.color = chans.length > 1 ? root._laneColor(i) : Theme.traceColor
            signalView.loadTraceFor(s, c)
        }
    }
    function reload() { root._rebuildLanes() }
    Component.onCompleted: root._rebuildLanes()
    Connections {
        target: signalView
        function onDurationSecChanged() { root._rebuildLanes() }
        function onLaneLayoutChanged() { root._rebuildLanes() }
    }

    function _fmtClock(sec) {
        sec = Math.max(0, Math.round(sec))
        var h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60
        var p = function (n) { return (n < 10 ? "0" : "") + n }
        return p(h) + ":" + p(m) + ":" + p(s)
    }
    function xForSec(sec) { return ((sec - viewStart) / span) * width }
    function secForX(px) { return viewStart + (px / Math.max(width, 1)) * span }

    ChartView {
        id: chart
        anchors.fill: parent
        antialiasing: true
        backgroundColor: "transparent"
        plotAreaColor: "transparent"
        legend.visible: false
        margins.top: 0; margins.bottom: 0; margins.left: 0; margins.right: 0

        ValueAxis {
            id: axX; min: root.viewStart; max: root.viewEnd
            visible: false; gridVisible: false; labelsVisible: false; lineVisible: false; minorGridVisible: false
        }
        ValueAxis {
            // single lane → auto-scaled amplitude; multi-lane → fixed [0,1] with each channel
            // normalized into its own horizontal band (stacked view).
            id: axY
            min: signalView.laneCount > 1 ? 0.0 : signalView.viewLo
            max: signalView.laneCount > 1 ? 1.0 : signalView.viewHi
            visible: false; gridVisible: false; labelsVisible: false; lineVisible: false; minorGridVisible: false
        }
        // lane series are created imperatively in _rebuildLanes() (see above)
    }

    // ---- lane labels + separators (multi-lane only) -------------------------
    Item {
        anchors.fill: parent
        visible: signalView.laneCount > 1
        Repeater {
            model: signalView.laneChannels
            delegate: Item {
                id: laneLbl
                required property int index
                required property string modelData
                readonly property int n: Math.max(1, signalView.laneCount)
                x: 0; width: root.width
                y: (index / n) * root.height
                height: root.height / n
                Rectangle {                               // separator above each band but the first
                    visible: laneLbl.index > 0
                    width: parent.width; height: 1
                    color: Qt.rgba(1, 1, 1, 0.06)
                }
                Text {
                    x: 6; y: 3
                    text: laneLbl.modelData
                    color: root._laneColor(laneLbl.index)
                    font.family: Theme.fontMono; font.pixelSize: 10
                }
            }
        }
    }

    // ---- quality bands (translucent, over the trace) ------------------------
    // ONE painted item, not one QQuickItem per RLE segment: a long recording carries
    // thousands of bands and the old Repeater instantiated a Rectangle for every one of
    // them (startup stall, and N x/width bindings re-evaluated on every pan frame). The
    // bands carry no mouse interaction here, so there is no hit-testing to preserve.
    Canvas {
        id: bandsLayer
        anchors.fill: parent
        readonly property var bands: segments.segmentBands
        readonly property real totalDur: segments.totalDurationSec
        readonly property real vs: root.viewStart
        readonly property real ve: root.viewEnd
        readonly property bool cbPal: Theme.useColorBlindPalette
        onBandsChanged: requestPaint()
        onTotalDurChanged: requestPaint()
        onVsChanged: requestPaint()
        onVeChanged: requestPaint()
        onCbPalChanged: requestPaint()
        onPaint: {
            var ctx = getContext("2d")
            ctx.reset()
            var bs = bandsLayer.bands
            var dur = bandsLayer.totalDur
            if (!bs || bs.length === 0 || dur <= 0)
                return
            for (var i = 0; i < bs.length; i++) {
                var s0 = bs[i].start * dur
                var s1 = (bs[i].start + bs[i].width) * dur
                if (s1 < bandsLayer.vs || s0 > bandsLayer.ve)
                    continue
                var c = Theme.tierInfo(bs[i].tier).color
                var xa = root.xForSec(s0)
                ctx.fillStyle = Qt.rgba(c.r, c.g, c.b, 0.15)
                ctx.fillRect(xa, 0, Math.max(1, root.xForSec(s1) - xa), height)
            }
        }
    }

    // selected-segment highlight (a click in the table / segment card / jump-to-poor lights it up here).
    // Geometry is CLAMPED to the chart bounds [0, root.width]: without it a segment starting before the
    // viewport gives a negative x and the rect + its border overflow LEFT into the adjacent panel (root is
    // not clipped).
    Rectangle {
        readonly property var s: selection.selectedSegment
        readonly property real x0: s ? Math.max(0, root.xForSec(s.startSec)) : 0
        readonly property real x1: s ? Math.min(root.width, root.xForSec(s.endSec)) : 0
        visible: s !== null && s.endSec > root.viewStart && s.startSec < root.viewEnd
        x: x0
        width: Math.max(2, x1 - x0)
        y: 0; height: root.height
        color: Qt.rgba(Theme.accent.r, Theme.accent.g, Theme.accent.b, 0.06)
        border.color: Theme.accent
        border.width: 2
        radius: 2
    }

    // faint zero midline (single-lane only; multi-lane draws per-band separators instead)
    Rectangle {
        visible: signalView.laneCount <= 1
        x: 0; width: root.width; y: root.height / 2; height: 1
        color: Qt.rgba(1, 1, 1, 0.05)
    }

    // playhead
    Rectangle {
        visible: !isNaN(root.playheadSec) && root.playheadSec >= root.viewStart && root.playheadSec <= root.viewEnd
        x: root.xForSec(root.playheadSec) - 0.75
        y: 0; width: 1.5; height: root.height
        color: Theme.accent
    }

    // measure/zoom drag box
    Rectangle {
        visible: ma.pressed && ma.startX >= 0 && root.tool !== "pan"
        x: Math.min(ma.startX, ma.curX); width: Math.abs(ma.curX - ma.startX)
        y: 0; height: root.height
        color: Qt.rgba(Theme.accent.r, Theme.accent.g, Theme.accent.b, 0.12)
        border.color: Theme.accent; border.width: 1
        Text {
            anchors.horizontalCenter: parent.horizontalCenter; anchors.top: parent.top; anchors.topMargin: 6
            visible: root.tool === "measure" && parent.width > 20
            text: "Δ " + Math.abs(root.secForX(ma.curX) - root.secForX(ma.startX)).toFixed(2) + " s"
            color: Theme.textPrimary; font.family: Theme.fontMono; font.pixelSize: 11
        }
    }

    // ---- interaction (scoped to the chart; does NOT cover toolbar/minimap) ---
    MouseArea {
        id: ma
        anchors.fill: parent
        hoverEnabled: true
        acceptedButtons: Qt.LeftButton
        cursorShape: root.tool === "pan" ? (pressed ? Qt.ClosedHandCursor : Qt.ArrowCursor) : Qt.CrossCursor
        property real startX: -1
        property real curX: -1
        property real startView: 0

        onPressed: (m) => { startX = m.x; curX = m.x; startView = root.viewStart }
        onReleased: (m) => {
            if (startX >= 0 && root.tool === "zoom" && Math.abs(m.x - startX) > 6) {
                var a = root.secForX(Math.min(startX, m.x)), b = root.secForX(Math.max(startX, m.x))
                if (b - a > 0.05) signalView.setView(a, b)
            }
            startX = -1; curX = -1
        }
        onPositionChanged: (m) => {
            if (pressed && startX >= 0) {
                curX = m.x
                if (root.tool === "pan") {
                    var dsec = ((m.x - startX) / Math.max(root.width, 1)) * root.span
                    var ns = Math.max(0, startView - dsec)
                    signalView.setView(ns, ns + root.span)   // realtime, immediate
                }
                tip.visible = false; root.playheadSec = NaN
                return
            }
            var sec = root.secForX(m.x)
            root.playheadSec = sec
            var seg = segments.segmentAt(sec)
            tip.timeText = root._fmtClock(sec)
            tip.windowText = sec.toFixed(1) + " s"
            tip.valueText = signalView.valueAt(sec).toFixed(2) + " a.u."
            // The else-branch is load-bearing: without it the tooltip kept the LAST segment's
            // tier/confidence and showed it over ungraded time (pre-inference, or a gap).
            if (seg) {
                tip.hasQuality = true
                tip.qualityTier = seg.tier
                tip.confidence = seg.confidence
            } else {
                tip.clearQuality()
            }
            tip.x = Math.min(m.x + 14, root.width - tip.width - 8)
            tip.y = Math.min(m.y + 14, root.height - tip.height - 8)
            tip.visible = true
        }
        onExited: { tip.visible = false; root.playheadSec = NaN }
        onWheel: (w) => {
            var factor = w.angleDelta.y > 0 ? 0.82 : 1.22
            var cur = root.secForX(w.x)
            var ns = Math.max(0, cur - (cur - root.viewStart) * factor)
            var nspan = Math.max(0.2, root.span * factor)
            signalView.setView(ns, ns + nspan)
        }
    }

    HoverTooltip { id: tip; objectName: "hoverTooltip"; visible: false; z: 60 }
}
