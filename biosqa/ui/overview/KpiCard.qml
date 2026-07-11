import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../"

// design spec (a): OverviewView is a "bento KPI dashboard". One KPI tile in the
// 6-up top row (mockup lines 324-331): sentence-case label, big mono value (tinted
// per-metric via `valueColor`), and a mono sublabel caption.
Rectangle {
    id: root
    color: Theme.bgPanel
    radius: Theme.radiusNav
    border.color: Theme.borderColor
    border.width: 1
    clip: true
    // ensure the card is always tall enough for its content (label + value + sublabel)
    implicitHeight: 84

    property string label: ""
    property string value: ""
    property string sublabel: ""
    property color valueColor: Theme.textPrimary

    ColumnLayout {
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.verticalCenter: parent.verticalCenter
        anchors.leftMargin: 15
        anchors.rightMargin: 15
        spacing: 5

        Label {
            text: root.label
            color: Theme.textSecondary
            font.family: Theme.fontUi
            font.pixelSize: 11
            elide: Text.ElideRight
            Layout.fillWidth: true
        }
        Label {
            text: root.value
            color: root.valueColor
            font.family: Theme.fontMono
            font.pixelSize: 20
            font.weight: Font.DemiBold
            font.letterSpacing: -0.4
            elide: Text.ElideRight
            Layout.fillWidth: true
        }
        Label {
            visible: root.sublabel.length > 0
            text: root.sublabel
            color: Theme.textMuted
            font.family: Theme.fontMono
            font.pixelSize: 10
            elide: Text.ElideRight
            Layout.fillWidth: true
        }
    }
}
