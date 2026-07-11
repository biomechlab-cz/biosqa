import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../"
import "../components"

// design spec (a)/(b): "ChannelListPanel (left dock, same column, below
// tree) eye-toggle, mod-color chip, per-channel sparkline". Bound to the
// Python `channels` (ChannelListModel) context property.
Rectangle {
    id: root
    color: Theme.bgPanelAlt

    // Real per-recording quality bands (normalized 0..1) from the segments
    // context property. Same track for every channel row (one recording).
    readonly property var _sparkBands: segments.segmentBands

    ColumnLayout {
        anchors.fill: parent
        anchors.leftMargin: 8
        anchors.rightMargin: 8
        anchors.topMargin: 10
        anchors.bottomMargin: 10
        spacing: 4

        RowLayout {
            Layout.fillWidth: true
            Layout.leftMargin: 6
            Layout.rightMargin: 6
            Label {
                text: "CHANNELS"
                color: Theme.textMuted
                font.family: Theme.fontUi
                font.pixelSize: 11
                font.weight: Font.DemiBold
                font.letterSpacing: 0.8
            }
            Item { Layout.fillWidth: true }
            Label {
                text: channels.visibleCount + " / " + channels.count
                color: Theme.textMuted
                font.family: Theme.fontMono
                font.pixelSize: 10
            }
        }

        ListView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            model: channels
            spacing: 1

            delegate: Rectangle {
                id: chRow
                width: ListView.view.width
                height: 40
                radius: 6
                color: rowHover.hovered ? Theme.hoverBg : "transparent"
                opacity: model.channelVisible ? 1.0 : 0.5

                readonly property color modC: model.modColor

                HoverHandler { id: rowHover }
                TapHandler { onTapped: channels.toggle(index) }

                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: 8
                    anchors.rightMargin: 8
                    spacing: 8

                    // eye-toggle chip
                    Rectangle {
                        implicitWidth: 16; implicitHeight: 16; radius: 4
                        color: model.channelVisible
                            ? Qt.rgba(chRow.modC.r, chRow.modC.g, chRow.modC.b, 0.18)
                            : "transparent"
                        border.width: 1
                        border.color: model.channelVisible ? chRow.modC : Theme.chipBorderMuted
                        Rectangle {
                            anchors.centerIn: parent
                            width: 6; height: 6; radius: 3
                            visible: model.channelVisible
                            color: chRow.modC
                        }
                    }

                    // modality identity square
                    Rectangle {
                        implicitWidth: 7; implicitHeight: 7; radius: 2
                        color: chRow.modC
                    }

                    ColumnLayout {
                        Layout.fillWidth: true
                        spacing: 1
                        Label {
                            text: model.name
                            color: model.channelVisible ? Theme.textPrimary : Theme.textMuted
                            font.family: Theme.fontMono
                            font.pixelSize: 12
                            elide: Text.ElideRight
                            Layout.fillWidth: true
                        }
                        Label {
                            text: model.unit
                            color: Theme.textMuted
                            font.family: Theme.fontMono
                            font.pixelSize: 9
                            elide: Text.ElideRight
                            Layout.fillWidth: true
                        }
                    }

                    QualitySparkline {
                        Layout.preferredWidth: 46
                        Layout.preferredHeight: 16
                        radius: 2
                        opacity: model.channelVisible ? 1.0 : 0.4
                        bands: root._sparkBands
                    }
                }
            }
        }
    }
}
