import QtQuick
import QtQuick.Layouts
import "../"
import "../components"

// design spec (b): "SegmentTable (View delegates only -- bind to Python
// models)". Bound to the `segments` (QualitySegmentModel) context property,
// laid out as a 7-column table inside a GlassPanel card. Lives inside the
// page ScrollView, so it renders all rows (natural height) via a Repeater.
GlassPanel {
    id: root
    pad: 0
    spacing: 0
    clip: true

    // Live (filtered) row count, consumed by the view header meta line.
    property alias count: rep.count

    // Column geometry (mockup: 56 / 130 / 1fr / 96 / 96 / 90 / 90, gap 12).
    readonly property int colGap: 12
    readonly property int hPad: 18
    readonly property int wNo: 56
    readonly property int wTier: 130
    readonly property int wStart: 96
    readonly property int wEnd: 96
    readonly property int wDur: 90
    readonly property int wConf: 90

    function fmt(sec) {
        sec = Math.max(0, Math.round(sec))
        var h = Math.floor(sec / 3600)
        var m = Math.floor((sec % 3600) / 60)
        var s = sec % 60
        function p(n) { return (n < 10 ? "0" : "") + n }
        return p(h) + ":" + p(m) + ":" + p(s)
    }

    Column {
        id: tableCol
        Layout.fillWidth: true

        // ---- header row -------------------------------------------------
        Rectangle {
            width: parent.width
            height: 38
            color: Theme.bgPanelAlt

            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: root.hPad
                anchors.rightMargin: root.hPad
                spacing: root.colGap

                component HeadCell: Text {
                    color: Theme.textMuted
                    font.family: Theme.fontUi
                    font.pixelSize: 10
                    font.weight: Font.DemiBold
                    font.letterSpacing: 0.5
                    verticalAlignment: Text.AlignVCenter
                }

                HeadCell { text: "#"; Layout.preferredWidth: root.wNo }
                HeadCell { text: "TIER"; Layout.preferredWidth: root.wTier }
                HeadCell { text: "ARTIFACTS"; Layout.fillWidth: true }
                HeadCell { text: "START"; Layout.preferredWidth: root.wStart }
                HeadCell { text: "END"; Layout.preferredWidth: root.wEnd }
                HeadCell { text: "DURATION"; Layout.preferredWidth: root.wDur }
                HeadCell { text: "CONF."; Layout.preferredWidth: root.wConf }
            }
        }

        // ---- rows -------------------------------------------------------
        Repeater {
            id: rep
            model: segments
            delegate: Rectangle {
                id: row
                required property int index
                required property real startSec
                required property real endSec
                required property string tier
                required property real confidence
                required property var artifacts

                readonly property var pal: Theme.currentQualityPalette()[tier]
                    || Theme.currentQualityPalette()["Q3"]
                readonly property bool selected: selection.selectedIndex === row.index

                width: tableCol.width
                height: 42
                color: selected ? Qt.rgba(Theme.accent.r, Theme.accent.g, Theme.accent.b, 0.10)
                     : (rowHover.containsMouse ? Theme.hoverBg : "transparent")

                // selected / hover left accent bar (clear in-place feedback)
                Rectangle {
                    anchors.left: parent.left
                    anchors.top: parent.top
                    anchors.bottom: parent.bottom
                    width: 3
                    color: row.selected ? row.pal.color
                         : (rowHover.containsMouse ? Theme.accent : "transparent")
                }

                // bottom hairline
                Rectangle {
                    anchors.bottom: parent.bottom
                    width: parent.width
                    height: 1
                    color: Theme.borderRow
                }

                MouseArea {
                    id: rowHover
                    anchors.fill: parent
                    hoverEnabled: true
                    cursorShape: Qt.PointingHandCursor
                    onClicked: {
                        selection.selectByIndex(row.index)
                        if (row.endSec > row.startSec)
                            signalView.setView(row.startSec, row.endSec)
                    }
                    onDoubleClicked: {
                        selection.selectByIndex(row.index)
                        AppController.toast("Segment #" + (row.index + 1) + " · " + row.tier
                            + " " + row.pal.label + " → inspector")
                        AppController.go("inspector")
                    }
                }

                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: root.hPad
                    anchors.rightMargin: root.hPad
                    spacing: root.colGap

                    // # index
                    Text {
                        Layout.preferredWidth: root.wNo
                        text: "#" + (row.index + 1)
                        color: Theme.textMuted
                        font.family: Theme.fontMono
                        font.pixelSize: 11
                        verticalAlignment: Text.AlignVCenter
                    }

                    // Tier: chip + code + name (locked to wTier so ARTIFACTS starts at a
                    // consistent x across all rows, not shifting with the tier-name length)
                    RowLayout {
                        Layout.preferredWidth: root.wTier
                        Layout.minimumWidth: root.wTier
                        Layout.maximumWidth: root.wTier
                        clip: true
                        spacing: 7
                        TierChip {
                            tier: row.tier
                            size: 18
                            Layout.alignment: Qt.AlignVCenter
                        }
                        Text {
                            text: row.tier
                            color: row.pal.color
                            font.family: Theme.fontMono
                            font.pixelSize: 12
                            font.weight: Font.DemiBold
                            verticalAlignment: Text.AlignVCenter
                        }
                        Text {
                            text: row.pal.label
                            color: Theme.textSecondary
                            font.family: Theme.fontUi
                            font.pixelSize: 12
                            elide: Text.ElideRight
                            Layout.fillWidth: true
                            verticalAlignment: Text.AlignVCenter
                        }
                    }

                    // Artifacts
                    Text {
                        Layout.fillWidth: true
                        text: (row.artifacts && row.artifacts.length > 0)
                            ? row.artifacts.join(", ") : "—"
                        color: Theme.textSecondary
                        font.family: Theme.fontUi
                        font.pixelSize: 12
                        elide: Text.ElideRight
                        horizontalAlignment: Text.AlignLeft
                        verticalAlignment: Text.AlignVCenter
                    }

                    // Start
                    Text {
                        Layout.preferredWidth: root.wStart
                        text: root.fmt(row.startSec)
                        color: Theme.textPrimary
                        font.family: Theme.fontMono
                        font.pixelSize: 12
                        verticalAlignment: Text.AlignVCenter
                    }
                    // End
                    Text {
                        Layout.preferredWidth: root.wEnd
                        text: root.fmt(row.endSec)
                        color: Theme.textPrimary
                        font.family: Theme.fontMono
                        font.pixelSize: 12
                        verticalAlignment: Text.AlignVCenter
                    }
                    // Duration
                    Text {
                        Layout.preferredWidth: root.wDur
                        text: root.fmt(row.endSec - row.startSec)
                        color: Theme.textSecondary
                        font.family: Theme.fontMono
                        font.pixelSize: 12
                        verticalAlignment: Text.AlignVCenter
                    }
                    // Confidence
                    Text {
                        Layout.preferredWidth: root.wConf
                        text: (row.confidence * 100).toFixed(0) + "%"
                        color: Theme.textSecondary
                        font.family: Theme.fontMono
                        font.pixelSize: 12
                        verticalAlignment: Text.AlignVCenter
                    }
                }
            }
        }

        // ---- empty state ------------------------------------------------
        Item {
            width: parent.width
            height: 60
            visible: rep.count === 0
            Text {
                anchors.centerIn: parent
                text: "No quality segments computed yet."
                color: Theme.textMuted
                font.family: Theme.fontUi
                font.pixelSize: 12
            }
        }
    }
}
