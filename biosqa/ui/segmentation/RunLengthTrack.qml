import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../"
import "../components"
import "../workspace"

// design spec (a): the run-length timeline of QualitySegmentationView, scaled
// to the whole recording. Same `QualityBandDelegate` rendering as the plot
// overlay (design spec (c) consistency), wrapped in a GlassPanel card.
GlassPanel {
    id: root
    pad: 16
    spacing: 10

    // Total recording duration (falls back to the last segment end so the
    // track still scales before the plot binds a recording).
    property real totalDurationSec: signalView.durationSec > 0 ? signalView.durationSec : 1.0

    function fmt(sec) {
        sec = Math.max(0, Math.round(sec))
        var h = Math.floor(sec / 3600)
        var m = Math.floor((sec % 3600) / 60)
        var s = sec % 60
        function p(n) { return (n < 10 ? "0" : "") + n }
        return p(h) + ":" + p(m) + ":" + p(s)
    }

    // ---- header row -----------------------------------------------------
    RowLayout {
        Layout.fillWidth: true
        Text {
            text: "Full recording · " + root.fmt(root.totalDurationSec)
            color: Theme.textSecondary
            font.family: Theme.fontUi
            font.pixelSize: 11
        }
        Item { Layout.fillWidth: true }
        Text {
            text: "click to select · double-click to open"
            color: Theme.textMuted
            font.family: Theme.fontMono
            font.pixelSize: 10
        }
    }

    // ---- run-length blocks ---------------------------------------------
    Rectangle {
        id: blocks
        Layout.fillWidth: true
        Layout.preferredHeight: 40
        radius: Theme.radiusControl
        color: Theme.bgApp
        clip: true

        function xForSec(sec) {
            return (sec / Math.max(root.totalDurationSec, 1e-9)) * width
        }

        Repeater {
            model: segments
            delegate: Item {
                id: block
                required property int index
                required property string tier
                required property real confidence
                required property real startSec
                required property real endSec
                readonly property bool selected: selection.selectedIndex === block.index
                readonly property var pal: Theme.tierInfo(block.tier)

                x: blocks.xForSec(startSec)
                width: Math.max(1, blocks.xForSec(endSec) - blocks.xForSec(startSec))
                height: blocks.height

                QualityBandDelegate {
                    anchors.fill: parent
                    tier: block.tier
                    confidence: block.confidence
                }

                // hover brighten
                Rectangle {
                    anchors.fill: parent
                    color: "white"
                    opacity: hover.containsMouse ? 0.16 : 0
                }
                // selected outline (clear in-place feedback)
                Rectangle {
                    anchors.fill: parent
                    color: "transparent"
                    border.color: "white"
                    border.width: 2
                    radius: 2
                    visible: block.selected
                }

                MouseArea {
                    id: hover
                    anchors.fill: parent
                    hoverEnabled: true
                    cursorShape: Qt.PointingHandCursor
                    onClicked: {
                        selection.selectByIndex(block.index)
                        if (block.endSec > block.startSec)
                            signalView.setView(block.startSec, block.endSec)
                    }
                    onDoubleClicked: {
                        selection.selectByIndex(block.index)
                        AppController.toast("Segment #" + (block.index + 1) + " · " + block.tier
                            + " " + block.pal.label + " → inspector")
                        AppController.go("inspector")
                    }

                    ToolTip.visible: hover.containsMouse
                    ToolTip.delay: 150
                    ToolTip.text: block.tier + " " + block.pal.label + "  ·  "
                        + root.fmt(block.startSec) + "–" + root.fmt(block.endSec)
                        + "  ·  " + Math.round(block.confidence * 100) + "% conf"
                }
            }
        }
    }

    // ---- 3-tick time ruler ---------------------------------------------
    RowLayout {
        Layout.fillWidth: true
        Text {
            text: root.fmt(0)
            color: Theme.textMuted
            font.family: Theme.fontMono
            font.pixelSize: 10
        }
        Item { Layout.fillWidth: true }
        Text {
            text: root.fmt(root.totalDurationSec / 2)
            color: Theme.textMuted
            font.family: Theme.fontMono
            font.pixelSize: 10
        }
        Item { Layout.fillWidth: true }
        Text {
            text: root.fmt(root.totalDurationSec)
            color: Theme.textMuted
            font.family: Theme.fontMono
            font.pixelSize: 10
        }
    }
}
