import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "workspace"

// Signal canvas: toolbar (Pan/Zoom/Measure/Marker) + a GPU QtCharts waveform for the
// primary channel + time axis + minimap. The waveform (WaveformChart) owns its own
// interaction, so the toolbar buttons and the minimap are no longer swallowed by a
// full-bleed overlay MouseArea (the old bug that made the tools/minimap dead).
Item {
    id: root

    property string tool: "pan"

    function _fmtClock(sec) {
        sec = Math.max(0, Math.round(sec))
        const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60
        const p = (n) => (n < 10 ? "0" : "") + n
        return p(h) + ":" + p(m) + ":" + p(s)
    }

    Rectangle { anchors.fill: parent; color: Theme.bgApp }

    // The channel-list eye toggles drive which channels are drawn as stacked lanes. Push the
    // visible set to the plot whenever it changes (and when a recording repopulates the list).
    Connections {
        target: channels
        function onChannelVisibilityChanged(index, visible) {
            signalView.setLaneChannels(channels.visibleNames())
        }
        function onCountChanged() { signalView.setLaneChannels(channels.visibleNames()) }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        // -- toolbar ---------------------------------------------------------
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 36
            color: Theme.bgPanelAlt
            border.color: Theme.borderSubtle
            border.width: 1

            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: 16
                anchors.rightMargin: 16
                spacing: 6

                Label {
                    text: "Signal canvas"
                    color: Theme.textPrimary
                    font.family: Theme.fontUi
                    font.pixelSize: 12
                    font.weight: Font.DemiBold
                    Layout.rightMargin: 4
                }

                Repeater {
                    model: [{ k: "pan", l: "Pan" }, { k: "zoom", l: "Zoom" }, { k: "measure", l: "Measure" }]
                    delegate: Button {
                        id: toolBtn
                        required property var modelData
                        readonly property bool active: root.tool === modelData.k
                        implicitHeight: 24
                        padding: 0
                        onClicked: root.tool = modelData.k
                        contentItem: Text {
                            text: toolBtn.modelData.l
                            color: toolBtn.active ? Theme.chipDark
                                 : (toolBtn.hovered ? Theme.textPrimary : Theme.textSecondary)
                            font.family: Theme.fontUi
                            font.pixelSize: 11
                            leftPadding: 9; rightPadding: 9
                            verticalAlignment: Text.AlignVCenter
                        }
                        background: Rectangle {
                            radius: 5
                            color: toolBtn.active ? Theme.accent
                                 : (toolBtn.hovered ? Theme.bgPanel : "transparent")
                            border.width: 1
                            border.color: toolBtn.active ? Theme.accent : Theme.borderColor
                        }
                    }
                }
                Item { Layout.fillWidth: true }

                Label {
                    text: "window " + root._fmtClock(signalView.viewStartSec)
                        + " – " + root._fmtClock(signalView.viewEndSec)
                        + "  ·  GPU"
                    color: Theme.textMuted
                    font.family: Theme.fontMono
                    font.pixelSize: 11
                }
            }
        }

        // -- primary-channel lane: 88px gutter + GPU waveform ----------------
        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 0

            Rectangle {
                Layout.preferredWidth: 88
                Layout.fillHeight: true
                color: Theme.bgRail
                Rectangle { anchors.right: parent.right; width: 1; height: parent.height; color: Theme.borderSubtle }
                ColumnLayout {
                    anchors.centerIn: parent
                    spacing: 3
                    RowLayout {
                        spacing: 5
                        Rectangle {
                            width: 6; height: 6; radius: 2
                            color: Theme.modalityColors[recordings.currentModality] ?? Theme.accent
                        }
                        Text {
                            text: (recordings.currentModality || "—").toUpperCase()
                            color: Theme.textPrimary
                            font.family: Theme.fontMono; font.pixelSize: 11; font.weight: Font.Medium
                        }
                    }
                    Text {
                        text: signalView.viewLo.toFixed(1) + "…" + signalView.viewHi.toFixed(1)
                        color: Theme.textMuted; font.family: Theme.fontMono; font.pixelSize: 9
                        visible: signalView.durationSec > 0
                    }
                }
            }

            WaveformChart {
                id: waveChart
                Layout.fillWidth: true
                Layout.fillHeight: true
                tool: root.tool
            }
        }

        TimeAxis {
            Layout.fillWidth: true
            Layout.preferredHeight: 24
            startSec: signalView.viewStartSec
            endSec: signalView.viewEndSec
        }

        Minimap {
            Layout.fillWidth: true
            Layout.preferredHeight: Theme.minimapHeight
        }
    }
}
