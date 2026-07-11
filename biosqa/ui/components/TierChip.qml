import QtQuick
import "../"

// Shared quality-tier chip (design mockup): a rounded square with a translucent
// tier-tinted fill, a tier-colored border, and the tier glyph centered in the tier
// color. `filled` inverts it (solid tier fill + dark glyph) for verdict badges.
Rectangle {
    id: root
    property string tier: "Q3"
    property int size: 16
    property bool filled: false

    readonly property var _p: (Theme.currentQualityPalette()[tier]
                               || Theme.currentQualityPalette()["Q3"])
    property color tierColor: _p.color

    width: size
    height: size
    radius: Math.max(2, Math.round(size * 0.28))
    color: filled ? tierColor : Qt.rgba(tierColor.r, tierColor.g, tierColor.b, 0.16)
    border.width: filled ? 0 : 1
    border.color: tierColor

    Text {
        anchors.centerIn: parent
        text: root._p.glyph
        color: root.filled ? Theme.bgApp : root.tierColor
        font.family: Theme.fontUi
        font.pixelSize: Math.round(root.size * 0.56)
        font.bold: true
    }
}
