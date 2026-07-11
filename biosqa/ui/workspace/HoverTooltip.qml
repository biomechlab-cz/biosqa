import QtQuick
import QtQuick.Layouts
import "../"
import "../components"

// design spec (a)/(c): glass hover tooltip -- "time, value, and the window's
// quality + confidence" (Plan 2 §8.2). Cheap flat-translucent panel (design
// spec (c): "prefer the cheap version over the plot area").
Rectangle {
    id: root
    width: 196
    implicitHeight: col.implicitHeight + 20
    radius: Theme.radiusPanel
    color: Theme.glassBg
    border.color: Theme.glassBorder
    border.width: 1

    property string timeText: "00:45:14.320"
    property string valueText: "+0.842 mV"
    property string windowText: "#3,182"
    property string qualityTier: "Q1"
    property real confidence: 0.71

    readonly property var _p: (Theme.currentQualityPalette()[qualityTier]
                               || Theme.currentQualityPalette()["Q3"])

    ColumnLayout {
        id: col
        anchors.fill: parent
        anchors.margins: 10
        spacing: 6

        Text {
            text: root.timeText
            color: Theme.textPrimary
            font.family: Theme.fontMono
            font.pixelSize: 11
        }

        RowLayout {
            Layout.fillWidth: true
            Text {
                text: "amplitude"
                color: Theme.textSecondary
                font.family: Theme.fontUi
                font.pixelSize: 10
            }
            Item { Layout.fillWidth: true }
            Text {
                text: root.valueText
                color: Theme.textPrimary
                font.family: Theme.fontMono
                font.pixelSize: 10
            }
        }
        RowLayout {
            Layout.fillWidth: true
            Text {
                text: "window"
                color: Theme.textSecondary
                font.family: Theme.fontUi
                font.pixelSize: 10
            }
            Item { Layout.fillWidth: true }
            Text {
                text: root.windowText
                color: Theme.textPrimary
                font.family: Theme.fontMono
                font.pixelSize: 10
            }
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 1
            color: Qt.rgba(1, 1, 1, 0.08)
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 6
            TierChip {
                tier: root.qualityTier
                size: 15
            }
            Text {
                text: root.qualityTier + " " + root._p.label
                color: root._p.color
                font.family: Theme.fontMono
                font.pixelSize: 10
                font.weight: Font.DemiBold
            }
            Item { Layout.fillWidth: true }
            Text {
                text: Math.round(root.confidence * 100) + "% conf"
                color: Theme.textSecondary
                font.family: Theme.fontMono
                font.pixelSize: 10
            }
        }
    }
}
