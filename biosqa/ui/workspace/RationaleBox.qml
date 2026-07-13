import QtQuick
import QtQuick.Layouts
import "../"

// design spec (a)/(b): "this is where Plan 1's explainability surfaces to the
// user" (Plan 2 §8.2). A blockquote with a 2px tier-colored left bar and a
// RATIONALE header. Bound to `selection.selectedSegment.rationale`.
ColumnLayout {
    id: root
    spacing: 8

    property string rationale: ""
    property color tierColor: Theme.accent

    Text {
        text: "RATIONALE"
        color: Theme.textMuted
        font.family: Theme.fontUi
        font.pixelSize: 11
        font.weight: Font.DemiBold
        font.letterSpacing: 0.7
    }

    Rectangle {
        Layout.fillWidth: true
        color: Theme.bgPanelAlt
        radius: Theme.radiusControl
        border.color: Theme.borderColor
        border.width: 1
        implicitHeight: rationaleText.implicitHeight + 22

        Rectangle {  // left tier bar
            anchors.left: parent.left
            anchors.top: parent.top
            anchors.bottom: parent.bottom
            width: 2
            radius: 1
            color: root.tierColor
        }

        Text {
            id: rationaleText
            anchors.fill: parent
            anchors.leftMargin: 13
            anchors.rightMargin: 13
            anchors.topMargin: 11
            anchors.bottomMargin: 11
            text: root.rationale.length > 0 ? root.rationale : "No rationale available for this segment."
            color: Theme.textBody
            font.family: Theme.fontUi
            font.pixelSize: 12
            lineHeight: 1.45
            wrapMode: Text.WordWrap
        }
    }
}
