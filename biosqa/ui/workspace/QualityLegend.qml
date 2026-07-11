import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../"
import "../components"

// design spec (a): "QualityLegend (left dock, bottom) Q3..Q0 chips". Reuses
// the shared TierChip so the legend and the bands it explains never drift out
// of sync (design spec (c): "keep all three... never color alone").
Rectangle {
    id: root
    color: Theme.bgPanelAlt
    border.color: Theme.borderSubtle
    border.width: 1

    readonly property var tiers: ["Q3", "Q2", "Q1", "Q0"]

    ColumnLayout {
        anchors.fill: parent
        anchors.leftMargin: 14
        anchors.rightMargin: 14
        anchors.topMargin: 10
        anchors.bottomMargin: 10
        spacing: 4

        Label {
            text: "QUALITY LEGEND"
            color: Theme.textMuted
            font.family: Theme.fontUi
            font.pixelSize: 11
            font.weight: Font.DemiBold
            font.letterSpacing: 0.8
            Layout.bottomMargin: 4
        }

        Repeater {
            model: root.tiers
            delegate: RowLayout {
                required property string modelData
                readonly property var entry: Theme.currentQualityPalette()[modelData]
                Layout.fillWidth: true
                spacing: 8

                TierChip {
                    tier: modelData
                    size: 14
                }

                Label {
                    text: modelData
                    color: entry.color
                    font.family: Theme.fontMono
                    font.pixelSize: 10
                    font.weight: Font.DemiBold
                    Layout.preferredWidth: 20
                }

                Label {
                    text: entry.label
                    color: Theme.textSecondary
                    font.family: Theme.fontUi
                    font.pixelSize: 11
                    Layout.fillWidth: true
                }
            }
        }
    }
}
