import QtQuick
import QtQuick.Shapes
import QtQuick.Layouts
import "../"
import "../components"

// Overview donut hero (mockup lines 336-356): per-tier ring (strokeWidth 18) with a
// centered usable-% readout, above a tier legend showing each tier's share of the
// recording. Fed by `segments.tierFractions`; falls back to a static mockup
// distribution when no inference has run yet so the tile is never blank.
ColumnLayout {
    id: root
    spacing: 0

    // e.g. { "Q3": 0.48, "Q2": 0.355, "Q1": 0.10, "Q0": 0.065 }
    property var fractions: ({})

    readonly property var _static: ({ "Q3": 0.48, "Q2": 0.355, "Q1": 0.10, "Q0": 0.065 })
    readonly property var _eff: (root.fractions && Object.keys(root.fractions).length > 0)
                                ? root.fractions : root._static
    readonly property var _tiers: ["Q3", "Q2", "Q1", "Q0"]
    readonly property real _usable: (root._eff["Q3"] || 0) + (root._eff["Q2"] || 0)

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
            property var watch: root._eff
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
                var start = -Math.PI / 2
                for (var i = 0; i < root._tiers.length; i++) {
                    var t = root._tiers[i]
                    var frac = root._eff[t] || 0
                    if (frac <= 0)
                        continue
                    var end = start + 2 * Math.PI * frac
                    ctx.beginPath()
                    ctx.strokeStyle = Theme.currentQualityPalette()[t].color
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
                text: (root._usable * 100).toFixed(1) + "%"
                color: Theme.textPrimary
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

    // ---- legend --------------------------------------------------------------
    ColumnLayout {
        Layout.fillWidth: true
        spacing: 9

        Repeater {
            model: root._tiers
            delegate: RowLayout {
                id: legRow
                required property string modelData
                readonly property var _p: Theme.currentQualityPalette()[modelData]
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
                    text: ((root._eff[legRow.modelData] || 0) * 100).toFixed(1) + "%"
                    color: Theme.textPrimary
                    font.family: Theme.fontMono
                    font.pixelSize: 12
                }
            }
        }
    }

    Item { Layout.fillHeight: true }
}
