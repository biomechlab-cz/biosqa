import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../"

// Per-modality "usable for what?" verdicts (research3 task-relative quality) — EEG per-band (δ/θ/α/β/γ,
// which carry real content above the 1/f floor) and EDA tonic (SCL) / phasic (SCR). Binds to the `guard`
// singleton's usabilityVerdicts (filled by the Coordinator on selection); each verdict = {label, usable,
// detail}. ECG/PPG surface their RATE verdict separately (the ♥ Rate-usable card), so this stays empty there.
ColumnLayout {
    id: root
    readonly property var verdicts: guard.usabilityVerdicts
    readonly property int usableCount: {
        var c = 0
        if (verdicts)
            for (var i = 0; i < verdicts.length; ++i)
                if (verdicts[i].usable) c++
        return c
    }
    visible: verdicts && verdicts.length > 0
    spacing: 6

    RowLayout {
        Layout.fillWidth: true
        spacing: 8
        Text {
            text: "TASK USABILITY"
            color: Theme.textMuted
            font.family: Theme.fontUi; font.pixelSize: 10
            font.weight: Font.DemiBold; font.letterSpacing: 0.7
        }
        Item { Layout.fillWidth: true }
        Text {
            text: root.usableCount + " / " + (root.verdicts ? root.verdicts.length : 0) + " usable"
            color: Theme.textMuted
            font.family: Theme.fontMono; font.pixelSize: 10
        }
    }

    Repeater {
        model: root.verdicts
        delegate: RowLayout {
            required property var modelData
            readonly property color tone: modelData.usable ? Theme.tierInfo("Q3").color
                                                          : Theme.tierInfo("Q1").color
            Layout.fillWidth: true
            spacing: 8
            Rectangle {
                Layout.preferredWidth: 16; Layout.preferredHeight: 16
                radius: 4
                color: Qt.rgba(tone.r, tone.g, tone.b, 0.16)
                border.width: 1; border.color: tone
                Text {
                    anchors.centerIn: parent
                    text: parent.parent.modelData.usable ? "✓" : "✕"
                    color: parent.parent.tone
                    font.pixelSize: 10; font.bold: true
                }
            }
            Text {
                text: modelData.label
                color: Theme.textSecondary
                font.family: Theme.fontUi; font.pixelSize: 11
                Layout.preferredWidth: 118
                elide: Text.ElideRight
            }
            Text {
                text: modelData.detail
                color: Theme.textMuted
                font.family: Theme.fontUi; font.pixelSize: 10
                Layout.fillWidth: true
                elide: Text.ElideRight
                ToolTip.visible: detMa.containsMouse && truncated
                ToolTip.text: modelData.detail
                MouseArea { id: detMa; anchors.fill: parent; hoverEnabled: true }
            }
        }
    }

    Text {
        Layout.fillWidth: true
        text: "Which downstream uses this segment still supports, even where the overall grade is poor."
        color: Theme.textMuted; font.family: Theme.fontUi; font.pixelSize: 9
        wrapMode: Text.WordWrap
    }
}
