import QtQuick
import QtQuick.Shapes
import QtQuick.Layouts
import "../"
import "../components"

// Overview donut hero (mockup lines 336-356): per-tier ring (strokeWidth 18) with a
// centered usable-% readout, above a tier legend showing each tier's share of the
// recording. Fed by `segments.tierFractions`.
//
// There is NO fallback distribution. This tile previously painted a hardcoded mockup
// ring whenever no inference had run, so an untouched app showed an invented quality
// breakdown as if it were measured. An empty tile is the only honest empty state.
ColumnLayout {
    id: root
    spacing: 0

    // e.g. { "Q3": 0.48, "Q2": 0.355, "Q1": 0.10, "Q0": 0.065 }
    property var fractions: ({})

    readonly property bool hasData: root.fractions && Object.keys(root.fractions).length > 0
    readonly property var _tiers: ["Q3", "Q2", "Q1", "Q0"]
    readonly property real _usable: (root.fractions["Q3"] || 0) + (root.fractions["Q2"] || 0)
    readonly property string usableText: root.hasData ? ((root._usable * 100).toFixed(1) + "%") : "—"

    // ---- ring ----------------------------------------------------------------
    Item {
        Layout.alignment: Qt.AlignHCenter
        Layout.topMargin: 8
        Layout.bottomMargin: 18
        implicitWidth: 176
        implicitHeight: 176

        // Canvas ring (a Repeater can't host ShapePath delegates — must be Items).
        Canvas {
            id: ring
            anchors.fill: parent
            property var watch: root.fractions
            onWatchChanged: requestPaint()
            // repaint on a color-blind palette toggle too — the ring reads currentQualityPalette() in
            // onPaint, so without this the ring keeps stale tier colors until fractions next change.
            property bool cbPal: Theme.useColorBlindPalette
            onCbPalChanged: requestPaint()
            onPaint: {
                var ctx = getContext("2d")
                ctx.reset()
                var cx = 88, cy = 88, r = 88 - 11
                ctx.lineWidth = 18
                if (!root.hasData) {                       // no inference yet: an empty track, no tiers
                    ctx.beginPath()
                    ctx.strokeStyle = Theme.bgPanelAlt
                    ctx.arc(cx, cy, r, 0, 2 * Math.PI)
                    ctx.stroke()
                    return
                }
                var start = -Math.PI / 2
                for (var i = 0; i < root._tiers.length; i++) {
                    var t = root._tiers[i]
                    var frac = root.fractions[t] || 0
                    if (frac <= 0)
                        continue
                    var end = start + 2 * Math.PI * frac
                    ctx.beginPath()
                    ctx.strokeStyle = Theme.tierInfo(t).color
                    ctx.arc(cx, cy, r, start, end)
                    ctx.stroke()
                    start = end
                }
            }
        }

        Column {
            anchors.centerIn: parent
            spacing: 1
            Text {
                anchors.horizontalCenter: parent.horizontalCenter
                text: root.usableText
                color: root.hasData ? Theme.textPrimary : Theme.textMuted
                font.family: Theme.fontMono
                font.pixelSize: 26
                font.bold: true
                font.letterSpacing: -0.5
            }
            Text {
                anchors.horizontalCenter: parent.horizontalCenter
                text: "usable"
                color: Theme.textSecondary
                font.family: Theme.fontUi
                font.pixelSize: 10
            }
        }
    }

    // ---- empty state ---------------------------------------------------------
    Text {
        visible: !root.hasData
        Layout.fillWidth: true
        horizontalAlignment: Text.AlignHCenter
        text: "No inference yet — open a recording to measure the quality distribution."
        color: Theme.textMuted
        font.family: Theme.fontUi
        font.pixelSize: 12
        wrapMode: Text.WordWrap
    }

    // ---- legend --------------------------------------------------------------
    ColumnLayout {
        visible: root.hasData
        Layout.fillWidth: true
        spacing: 9

        Repeater {
            model: root.hasData ? root._tiers : []
            delegate: RowLayout {
                id: legRow
                required property string modelData
                readonly property var _p: Theme.tierInfo(modelData)
                Layout.fillWidth: true
                spacing: 9

                TierChip { tier: legRow.modelData; size: 15 }

                Text {
                    text: legRow.modelData
                    color: legRow._p.color
                    font.family: Theme.fontMono
                    font.pixelSize: 11
                    font.weight: Font.DemiBold
                    Layout.preferredWidth: 22
                }
                Text {
                    text: legRow._p.label
                    color: Theme.textSecondary
                    font.family: Theme.fontUi
                    font.pixelSize: 12
                    Layout.fillWidth: true
                }
                Text {
                    text: ((root.fractions[legRow.modelData] || 0) * 100).toFixed(1) + "%"
                    color: Theme.textPrimary
                    font.family: Theme.fontMono
                    font.pixelSize: 12
                }
            }
        }
    }

    Item { Layout.fillHeight: true }
}
