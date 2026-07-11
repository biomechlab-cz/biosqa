import QtQuick
import QtQuick.Shapes
import "../"

// design spec (c): "QualityBandDelegate must render fill + glyph + code,
// never color alone" -- the redundant, load-bearing non-color encoding for
// color-blind safety. One instance per quality interval, positioned by
// `QualityOverlay.qml`.
Item {
    id: root

    property string tier: "Q3"
    property real confidence: 1.0

    readonly property var entry: Theme.currentQualityPalette()[tier] ?? Theme.currentQualityPalette()["Q3"]

    Rectangle {
        anchors.fill: parent
        color: root.entry.color
        opacity: 0.22 + 0.15 * root.confidence
    }

    // Diagonal hatch, uniquely on Q0 (design spec (c)).
    Shape {
        anchors.fill: parent
        visible: !!root.entry.hatch
        ShapePath {
            strokeWidth: 1
            strokeColor: Qt.rgba(0, 0, 0, 0.35)
            fillColor: "transparent"
            startX: 0; startY: root.height
            PathLine { x: root.width; y: 0 }
        }
    }

    // Glyph + mono tier code, co-rendered with the fill (never color alone).
    Row {
        anchors.centerIn: parent
        spacing: 3
        visible: root.width > 24

        Text {
            text: root.entry.glyph
            color: Theme.textPrimary
            font.pixelSize: 10
        }
        Text {
            text: root.tier
            color: Theme.textPrimary
            font.family: Theme.fontMono
            font.pixelSize: 9
            font.bold: true
        }
    }
}
