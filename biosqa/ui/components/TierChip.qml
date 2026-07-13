import QtQuick
import "../"

// Shared quality-tier chip (design mockup): a rounded square with a translucent
// tier-tinted fill, a tier-colored border, and the tier glyph centered in the tier
// color. `filled` inverts it (solid tier fill + dark glyph) for verdict badges.
//
// An unknown/empty tier renders as a blank muted chip, NOT as Q3: the old
// `?? palette["Q3"]` fallback painted a green "excellent" check for a segment that
// had no grade at all.
Rectangle {
    id: root
    property string tier: ""
    property int size: 16
    property bool filled: false

    readonly property bool known: Theme.isTier(tier)
    readonly property var _p: Theme.tierInfo(tier)
    property color tierColor: _p.color

    // exposed for screen readers: a band narrower than ~24px drops its glyph + code and
    // falls back to color alone, so the grade must also be carried non-visually.
    Accessible.role: Accessible.Graphic
    Accessible.name: root.known ? (root.tier + " " + root._p.label) : "No grade"

    width: size
    height: size
    radius: Math.max(2, Math.round(size * 0.28))
    color: !root.known ? "transparent"
         : (filled ? tierColor : Qt.rgba(tierColor.r, tierColor.g, tierColor.b, 0.16))
    border.width: (filled && root.known) ? 0 : 1
    border.color: root.known ? tierColor : Theme.borderColor

    Text {
        anchors.centerIn: parent
        text: root._p.glyph
        color: root.filled ? Theme.bgApp : root.tierColor
        font.family: Theme.fontUi
        font.pixelSize: Math.round(root.size * 0.56)
        font.bold: true
    }
}
