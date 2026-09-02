import QtQuick
import QtQuick.Layouts
import "../"

// design spec (b): glitch-type/artifact tags for the selected segment
// (Plan 1 §12.1). Bound to `selection.selectedSegment.artifacts` (a list of
// strings); header + wrapping chips with a dot + name.
ColumnLayout {
    id: root
    property var artifacts: []
    spacing: 9

    Text {
        text: "DETECTED ARTIFACTS"
        color: Theme.textMuted
        font.family: Theme.fontUi
        font.pixelSize: 11
        font.weight: Font.DemiBold
        font.letterSpacing: 0.7
    }

    Flow {
        Layout.fillWidth: true
        spacing: 7

        Repeater {
            model: root.artifacts
            delegate: Rectangle {
                id: chip
                required property string modelData
                radius: Theme.radiusChip
                color: Theme.hoverBg
                border.color: Theme.chipBorderMuted
                border.width: 1
                implicitWidth: chipRow.implicitWidth + 18
                implicitHeight: chipRow.implicitHeight + 8

                Row {
                    id: chipRow
                    anchors.centerIn: parent
                    spacing: 6

                    Rectangle {
                        anchors.verticalCenter: parent.verticalCenter
                        width: 6; height: 6; radius: 3
                        color: Theme.statusRed
                    }
                    Text {
                        text: chip.modelData
                        color: Theme.textPrimary
                        font.family: Theme.fontUi
                        font.pixelSize: 11
                    }
                }
            }
        }

        Text {
            visible: root.artifacts.length === 0
            text: "No artifacts flagged"
            color: Theme.textMuted
            font.family: Theme.fontUi
            font.pixelSize: 11
        }
    }
}
