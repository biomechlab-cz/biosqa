import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../"

// Accent call-to-action button (design mockup Export / Save-correction): solid accent
// fill, dark-on-accent text (#062521), optional leading glyph, hover brightness.
Button {
    id: root
    property string glyph: ""
    padding: 0
    implicitHeight: 30

    contentItem: RowLayout {
        spacing: 7
        Text {
            visible: root.glyph !== ""
            text: root.glyph
            color: Theme.chipDark
            font.family: Theme.fontUi
            font.pixelSize: 13
            font.bold: true
            Layout.leftMargin: 13
        }
        Text {
            text: root.text
            color: Theme.chipDark
            font.family: Theme.fontUi
            font.pixelSize: 13
            font.weight: Font.DemiBold
            Layout.leftMargin: root.glyph === "" ? 13 : 0
            Layout.rightMargin: 13
            Layout.fillWidth: true
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
        }
    }
    background: Rectangle {
        radius: Theme.radiusControl
        color: root.down ? Qt.darker(Theme.accent, 1.08)
                         : (root.hovered ? Qt.lighter(Theme.accent, 1.07) : Theme.accent)
    }
}
