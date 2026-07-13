import QtQuick
import QtQuick.Controls
import "../"

// Secondary outline button (design mockup): transparent fill, hairline border that
// brightens on hover, muted text that lifts to primary on hover.
Button {
    id: root
    padding: 9
    implicitHeight: 30

    contentItem: Text {
        text: root.text
        color: root.hovered ? Theme.textPrimary : Theme.textSecondary
        font.family: Theme.fontUi
        font.pixelSize: 12
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
    }
    background: Rectangle {
        radius: Theme.radiusControl
        color: root.hovered ? Theme.hoverBg : "transparent"
        border.width: 1
        border.color: root.hovered ? Theme.borderHover : Theme.borderColor
    }
}
