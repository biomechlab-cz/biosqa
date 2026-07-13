import QtQuick
import QtQuick.Layouts
import "../"
import "../components"

// design spec (a)/(c): glass hover tooltip -- "time, value, and the window's
// quality + confidence" (Plan 2 §8.2). Cheap flat-translucent panel (design
// spec (c): "prefer the cheap version over the plot area").
//
// The tier/confidence row renders ONLY when `hasQuality` is set, i.e. when a real
// graded segment sits under the cursor. It used to default to "Q1 / 71% conf" and
// the caller never cleared it, so hovering ungraded time (before inference, or over
// a gap) showed a grade for a segment that does not exist.
Rectangle {
    id: root
    width: 196
    implicitHeight: col.implicitHeight + 20
    radius: Theme.radiusPanel
    color: Theme.glassBg
    border.color: Theme.glassBorder
    border.width: 1

    property string timeText: ""
    property string valueText: ""
    property string windowText: ""
    property bool hasQuality: false
    property string qualityTier: ""
    property real confidence: 0

    readonly property var _p: Theme.tierInfo(qualityTier)

    function clearQuality() {
        root.hasQuality = false
        root.qualityTier = ""
        root.confidence = 0
    }

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
            visible: root.hasQuality
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

        Text {
            visible: !root.hasQuality
            Layout.fillWidth: true
            text: "Not graded"
            color: Theme.textMuted
            font.family: Theme.fontUi
            font.pixelSize: 10
        }
    }
}
