import QtQuick
import QtQuick.Layouts
import "../"
import "../components"

// Grid of clickable segment miniatures — the switchable alternative to SegmentTable. Each card
// shows a mini-waveform of the segment's real samples plus the same info the table denotes: tier
// chip + code, start–end, confidence, artifacts, and the ↺ recoverable marker. Bound to the same
// `segments` model + `selection` source of truth, so selection/highlight stay in sync everywhere.
Flow {
    id: root
    spacing: 12
    property alias count: rep.count

    // Responsive columns: fit as many ~minCard-wide cards as the width allows, then stretch each to
    // fill the row so the grid spans the full width edge-to-edge (like the table) and re-flows as
    // the window resizes.
    readonly property int minCard: 190
    readonly property int cols: Math.max(1, Math.floor((width + spacing) / (minCard + spacing)))
    readonly property real cardW: Math.floor((width - (cols - 1) * spacing) / cols)

    function fmt(sec) {
        sec = Math.max(0, Math.round(sec))
        var m = Math.floor(sec / 60), s = sec % 60
        return m + ":" + (s < 10 ? "0" : "") + s
    }

    Repeater {
        id: rep
        model: segments
        delegate: Rectangle {
            id: card
            required property int index
            required property real startSec
            required property real endSec
            required property string tier
            required property real confidence
            required property var artifacts
            required property bool recoverable
            readonly property var pal: Theme.currentQualityPalette()[tier]
                                       || Theme.currentQualityPalette()["Q3"]
            readonly property bool selected: selection.selectedIndex === index

            width: root.cardW
            height: 150
            radius: 10
            color: selected ? Qt.rgba(Theme.accent.r, Theme.accent.g, Theme.accent.b, 0.10) : Theme.bgPanelAlt
            border.width: selected ? 2 : 1
            border.color: selected ? Theme.accent : (cardHover.containsMouse ? Theme.accent : Theme.borderColor)

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 10
                spacing: 6

                // header: tier chip + code + recoverable + confidence
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 6
                    TierChip { tier: card.tier; size: 16 }
                    Text {
                        text: card.tier; color: card.pal.color
                        font.family: Theme.fontMono; font.pixelSize: 12; font.weight: Font.DemiBold
                    }
                    Text {
                        visible: card.recoverable; text: "↺"; color: Theme.accent
                        font.pixelSize: 12; font.bold: true
                    }
                    Item { Layout.fillWidth: true }
                    Text {
                        text: (card.confidence * 100).toFixed(0) + "%"; color: Theme.textMuted
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
                        // Stagger the envelope builds so a large grid doesn't paint N Canvases in one
                        // frame. The read itself is now an in-memory cache slice (see curveForRange), so
                        // this only smooths rendering. A MODULO spread (not the old `min(2500, index*10)`,
                        // which collapsed every card at index>=250 onto the same 2500 ms tick — a thundering
                        // herd) keeps at most ~1/50th of the cards firing in any one slot.
                        Timer {
                            interval: 16 + (card.index % 50) * 12
                            running: true; repeat: false
                            onTriggered: mini.curve = signalView.curveForRange(card.startSec, card.endSec)
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
                    text: "#" + (card.index + 1) + " · " + root.fmt(card.startSec) + "–" + root.fmt(card.endSec)
                    color: Theme.textSecondary
                    font.family: Theme.fontMono; font.pixelSize: 10
                    Layout.fillWidth: true; elide: Text.ElideRight
                }
                Text {
                    visible: card.artifacts && card.artifacts.length > 0
                    text: (card.artifacts && card.artifacts.length > 0) ? card.artifacts.join(", ") : ""
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
                    selection.selectByIndex(card.index)
                    if (card.endSec > card.startSec)
                        signalView.setView(card.startSec, card.endSec)
                }
                onDoubleClicked: {
                    selection.selectByIndex(card.index)
                    AppController.go("inspector")
                }
            }
        }
    }
}
