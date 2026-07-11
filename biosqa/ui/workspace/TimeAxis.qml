import QtQuick
import "../"

// design spec (a): "TimeAxis" strip below the stacked channel lanes.
Rectangle {
    id: root
    color: Theme.bgRail
    border.color: Theme.borderColor
    border.width: 1

    property real startSec: 0
    property real endSec: 10
    readonly property int tickCount: 6

    function formatSec(sec) {
        const m = Math.floor(sec / 60)
        const s = Math.floor(sec % 60)
        return m + ":" + (s < 10 ? "0" : "") + s
    }

    Row {
        anchors.fill: parent
        Repeater {
            model: root.tickCount
            delegate: Item {
                required property int index
                width: root.width / root.tickCount
                height: root.height

                Text {
                    anchors.centerIn: parent
                    text: root.formatSec(root.startSec
                        + (root.endSec - root.startSec) * (index / (root.tickCount - 1)))
                    color: Theme.textMuted
                    font.family: Theme.fontMono
                    font.pixelSize: 10
                }
            }
        }
    }
}
