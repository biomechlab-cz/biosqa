import QtQuick
import QtQuick.Shapes
import "../"

// design spec (b): "arc drawn from a bound `confidence` float via
// `Shape`/`Canvas`". Pure QML, no Python object behind it -- bound directly
// to `selection.selectedSegment.confidence`.
Item {
    id: root
    property real confidence: 0.0  // 0..1
    property color arcColor: Theme.accent
    implicitWidth: 64
    implicitHeight: 64

    readonly property real _startAngle: -90
    readonly property real _sweep: 360 * Math.max(0, Math.min(1, confidence))

    Shape {
        anchors.fill: parent
        ShapePath {
            strokeWidth: 6
            strokeColor: Theme.borderColor
            fillColor: "transparent"
            capStyle: ShapePath.RoundCap
            PathAngleArc {
                centerX: root.width / 2
                centerY: root.height / 2
                radiusX: root.width / 2 - 4
                radiusY: root.height / 2 - 4
                startAngle: 0
                sweepAngle: 359.999
            }
        }
        ShapePath {
            strokeWidth: 6
            strokeColor: root.arcColor
            fillColor: "transparent"
            capStyle: ShapePath.RoundCap
            PathAngleArc {
                centerX: root.width / 2
                centerY: root.height / 2
                radiusX: root.width / 2 - 4
                radiusY: root.height / 2 - 4
                startAngle: root._startAngle
                sweepAngle: root._sweep
            }
        }
    }

    Text {
        anchors.centerIn: parent
        text: Math.round(root.confidence * 100) + "%"
        color: Theme.textPrimary
        font.family: Theme.fontMono
        font.pixelSize: 13
        font.bold: true
    }
}
