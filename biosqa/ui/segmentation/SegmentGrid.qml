import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../"
import "../components"

// Grid of clickable segment miniatures — the switchable alternative to SegmentTable. Each card
// shows a mini-waveform of the segment's real samples plus the same info the table denotes: tier
// chip + code, start–end, confidence, artifacts, and the ↺ recoverable marker. Bound to the same
// `segments` model + `selection` source of truth, so selection/highlight stay in sync everywhere.
//
// A VIRTUALIZING GridView (reuseItems), not a Flow+Repeater: the Repeater built a card — each with
// its own Canvas — for every RLE segment in the recording, and a stagger Timer existed only to
// spread the resulting thundering herd of envelope builds. With recycling, only the cells actually
// on screen exist, so the curve is fetched in onReused/Component.onCompleted and the Timer is gone.
GridView {
    id: root
    clip: true
    model: segments
    reuseItems: true
    boundsBehavior: Flickable.StopAtBounds
    cacheBuffer: 300

    ScrollBar.vertical: ScrollBar { policy: ScrollBar.AsNeeded }

    // Responsive columns: fit as many ~minCard-wide cards as the width allows, then stretch each to
    // fill the row so the grid spans the full width edge-to-edge (like the table) and re-flows as
    // the window resizes.
    readonly property int minCard: 190
    readonly property int gap: 12
    readonly property int cardH: 150
    readonly property int cols: Math.max(1, Math.floor((width + gap) / (minCard + gap)))

    cellWidth: Math.floor(width / cols)
    cellHeight: cardH + gap

    // bounded viewport => virtualization; short grids keep their natural height
    property int maxBodyHeight: 560
    readonly property int rowsNeeded: Math.ceil(count / Math.max(1, cols))
    implicitHeight: count === 0 ? 0 : Math.min(rowsNeeded * cellHeight, maxBodyHeight)

    function fmt(sec) {
        sec = Math.max(0, Math.round(sec))
        var m = Math.floor(sec / 60), s = sec % 60
        return m + ":" + (s < 10 ? "0" : "") + s
    }

    delegate: Item {
        id: cell
        required property int index
        required property real startSec
        required property real endSec
        required property string tier
        required property real confidence
        required property var artifacts
        required property bool recoverable

        width: root.cellWidth
        height: root.cellHeight

        // A recycled delegate keeps its old curve until the new range is fetched, so refresh on
        // both first build and every reuse.
        function refreshCurve() {
            mini.curve = signalView.curveForRange(cell.startSec, cell.endSec)
        }
        Component.onCompleted: cell.refreshCurve()
        GridView.onReused: cell.refreshCurve()

        Rectangle {
            id: card
            anchors.fill: parent
            anchors.rightMargin: root.gap
            anchors.bottomMargin: root.gap
            readonly property var pal: Theme.tierInfo(cell.tier)
            readonly property bool selected: selection.selectedIndex === cell.index

            radius: 10
            color: selected ? Qt.rgba(Theme.accent.r, Theme.accent.g, Theme.accent.b, 0.10) : Theme.bgPanelAlt
            border.width: selected ? 2 : 1
            border.color: selected ? Theme.accent : (cardHover.containsMouse ? Theme.accent : Theme.borderColor)

            Accessible.role: Accessible.Button
            Accessible.name: "Segment " + (cell.index + 1) + ", " + cell.tier + " " + card.pal.label
                             + ", " + Math.round(cell.confidence * 100) + "% confidence"

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 10
                spacing: 6

                // header: tier chip + code + recoverable + confidence
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 6
                    TierChip { tier: cell.tier; size: 16 }
                    Text {
                        text: cell.tier; color: card.pal.color
                        font.family: Theme.fontMono; font.pixelSize: 12; font.weight: Font.DemiBold
                    }
                    Text {
                        visible: cell.recoverable; text: "↺"; color: Theme.accent
                        font.pixelSize: 12; font.bold: true
                    }
                    Item { Layout.fillWidth: true }
                    Text {
                        text: (cell.confidence * 100).toFixed(0) + "%"; color: Theme.textMuted
                        font.family: Theme.fontMono; font.pixelSize: 11
                    }
                }

                // mini-waveform (the segment's decimated min/max envelope)
                Rectangle {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    radius: 6
                    color: Theme.bgPanel
                    border.color: Theme.borderRow
                    border.width: 1
                    clip: true
                    Canvas {
                        id: mini
                        anchors.fill: parent
                        anchors.margins: 4
                        property var curve: null
                        // brighter tint on dark backgrounds; a bright centre line makes the trace
                        // clearly readable regardless of theme.
                        readonly property color traceCol: Theme.dark
                            ? Qt.lighter(card.pal.color, 1.25) : card.pal.color
                        onCurveChanged: requestPaint()
                        onWidthChanged: requestPaint()      // re-render when the card re-flows/resizes
                        onHeightChanged: requestPaint()
                        onPaint: {
                            var ctx = getContext("2d"); ctx.reset()
                            var c = mini.curve
                            if (!c || !c.x || c.x.length < 2) return
                            var xs = c.x, ymn = c.ymin, ymx = c.ymax
                            var lo = c.lo, hi = c.hi, rng = (hi > lo) ? (hi - lo) : 1
                            var x0 = xs[0], x1 = xs[xs.length - 1], xr = (x1 > x0) ? (x1 - x0) : 1
                            var tc = mini.traceCol
                            function px(x) { return (x - x0) / xr * width }
                            function py(y) { return height - (y - lo) / rng * height }
                            // filled min/max envelope
                            ctx.fillStyle = Qt.rgba(tc.r, tc.g, tc.b, 0.38)
                            ctx.beginPath(); ctx.moveTo(px(xs[0]), py(ymx[0]))
                            for (var i = 1; i < xs.length; i++) ctx.lineTo(px(xs[i]), py(ymx[i]))
                            for (i = xs.length - 1; i >= 0; i--) ctx.lineTo(px(xs[i]), py(ymn[i]))
                            ctx.closePath(); ctx.fill()
                            // bright centre line (crisp + visible on dark)
                            ctx.strokeStyle = Qt.rgba(tc.r, tc.g, tc.b, 0.98)
                            ctx.lineWidth = 1.3
                            ctx.beginPath()
                            for (i = 0; i < xs.length; i++) {
                                var mid = (ymn[i] + ymx[i]) / 2
                                if (i === 0) ctx.moveTo(px(xs[i]), py(mid))
                                else ctx.lineTo(px(xs[i]), py(mid))
                            }
                            ctx.stroke()
                        }
                    }
                    Text {
                        anchors.centerIn: parent
                        visible: !mini.curve || !mini.curve.x || mini.curve.x.length < 2
                        text: "no preview"; color: Theme.textMuted
                        font.family: Theme.fontUi; font.pixelSize: 10
                    }
                }

                // footer: index + time
                Text {
                    text: "#" + (cell.index + 1) + " · " + root.fmt(cell.startSec) + "–" + root.fmt(cell.endSec)
                    color: Theme.textSecondary
                    font.family: Theme.fontMono; font.pixelSize: 10
                    Layout.fillWidth: true; elide: Text.ElideRight
                }
                Text {
                    visible: cell.artifacts && cell.artifacts.length > 0
                    text: (cell.artifacts && cell.artifacts.length > 0) ? cell.artifacts.join(", ") : ""
                    color: Theme.textMuted
                    font.family: Theme.fontUi; font.pixelSize: 9
                    Layout.fillWidth: true; elide: Text.ElideRight
                }
            }

            MouseArea {
                id: cardHover
                anchors.fill: parent
                hoverEnabled: true
                cursorShape: Qt.PointingHandCursor
                onClicked: {
                    selection.selectByIndex(cell.index)
                    if (cell.endSec > cell.startSec)
                        signalView.setView(cell.startSec, cell.endSec)
                }
                onDoubleClicked: {
                    selection.selectByIndex(cell.index)
                    AppController.go("inspector")
                }
            }
        }
    }
}
