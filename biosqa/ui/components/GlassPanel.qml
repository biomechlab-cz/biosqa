import QtQuick
import QtQuick.Layouts
import "../"

// Shared card wrapper (design mockup bento tiles / inspector cards): panel fill,
// hairline border, panel radius, optional title + subtitle header. Child items are
// stacked in an inner ColumnLayout (`default property content`).
Rectangle {
    id: root
    property string title: ""
    property string subtitle: ""
    property int pad: 18
    property alias spacing: body.spacing
    default property alias content: body.data

    color: Theme.bgPanel
    border.color: Theme.borderColor
    border.width: 1
    radius: Theme.radiusPanel
    implicitHeight: outer.implicitHeight + pad * 2
    implicitWidth: outer.implicitWidth + pad * 2

    ColumnLayout {
        id: outer
        anchors.fill: parent
        anchors.margins: root.pad
        spacing: 12

        ColumnLayout {
            visible: root.title !== ""
            Layout.fillWidth: true
            spacing: 2
            Text {
                text: root.title
                color: Theme.textPrimary
                font.family: Theme.fontUi
                font.pixelSize: 13
                font.weight: Font.DemiBold
            }
            Text {
                visible: root.subtitle !== ""
                text: root.subtitle
                color: Theme.textMuted
                font.family: Theme.fontUi
                font.pixelSize: 11
            }
        }

        ColumnLayout {
            id: body
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 10
        }
    }
}
