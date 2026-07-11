import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../"

// Interpretable classical-SQI breakdown (research3 explainability) — reused by the right-dock
// AI Quality Inspector AND the full-page Segment Inspector. Binds to the `guard` singleton
// (sqiBreakdown / sqiBreakdownFiltered / sqiConsensus) which the Coordinator fills on selection.
//
// - Raw / Filtered toggle: the SAME bank on the raw window vs a band-pass-filtered copy, so the user
//   can see what a standard filter would (baseline wander) or would not (in-band EMG) clean up.
// - Model-vs-SQI discordance banner: when the model grade is clean (Q2/Q3) but the RAW-bank consensus
//   is low, the classical indices disagree with the score — surfaced as a warning.
ColumnLayout {
    id: root

    property string tier: ""                 // the selected segment's MODEL tier (for discordance)
    property bool showFiltered: false        // Raw (false) / Filtered (true) toggle state
    readonly property var rowsRaw: guard.sqiBreakdown
    readonly property bool filteredAvailable: guard.sqiBreakdownFiltered && guard.sqiBreakdownFiltered.length > 0
    readonly property var rowsShown: (showFiltered && filteredAvailable) ? guard.sqiBreakdownFiltered
                                                                         : guard.sqiBreakdown
    // consensus is -1 when not computable; >= 0 AND < 0.5 (incl. a real 0.0) = the model disagrees.
    readonly property bool discordant: guard.sqiConsensus >= 0.0 && guard.sqiConsensus < 0.5
                                       && (tier === "Q3" || tier === "Q2")

    visible: rowsRaw && rowsRaw.length > 0
    spacing: 6

    // -- header: title + Raw/Filtered segmented toggle -----------------------
    RowLayout {
        Layout.fillWidth: true
        spacing: 8
        Text {
            text: "SIGNAL QUALITY INDICES"
            color: Theme.textMuted
            font.family: Theme.fontUi; font.pixelSize: 10
            font.weight: Font.DemiBold; font.letterSpacing: 0.7
        }
        Item { Layout.fillWidth: true }
        Row {                                 // segmented Raw | Filtered toggle
            spacing: 0
            Repeater {
                model: [{ f: false, t: "Raw" }, { f: true, t: "Filtered" }]
                delegate: Rectangle {
                    id: segCell
                    required property var modelData
                    readonly property bool on: root.showFiltered === modelData.f
                    // the Filtered cell is disabled when no filtered view is available (filter failed),
                    // so the user can't land on a silent-empty view.
                    readonly property bool cellEnabled: !modelData.f || root.filteredAvailable
                    width: segTxt.implicitWidth + 18
                    height: 20
                    opacity: cellEnabled ? 1.0 : 0.4
                    color: on ? Theme.accent : Theme.bgPanelAlt
                    border.width: 1
                    border.color: on ? Theme.accent : Theme.borderColor
                    radius: 5
                    Text {
                        id: segTxt
                        anchors.centerIn: parent
                        text: segCell.modelData.t
                        color: segCell.on ? "#062521" : Theme.textSecondary
                        font.family: Theme.fontUi; font.pixelSize: 10
                        font.weight: segCell.on ? Font.DemiBold : Font.Normal
                    }
                    MouseArea {
                        anchors.fill: parent
                        enabled: segCell.cellEnabled
                        cursorShape: Qt.PointingHandCursor
                        onClicked: root.showFiltered = segCell.modelData.f
                    }
                }
            }
        }
    }

    // -- model-vs-SQI discordance banner ------------------------------------
    Rectangle {
        visible: root.discordant
        Layout.fillWidth: true
        color: Qt.rgba(0.878, 0.639, 0.18, 0.12)      // amber tint
        border.color: "#E0A32E"; border.width: 1
        radius: 8
        implicitHeight: discTxt.implicitHeight + 16
        Text {
            id: discTxt
            anchors.fill: parent
            anchors.margins: 8
            text: "⚠ Classical indices disagree with the model's " + root.tier
                  + " grade here (SQI consensus " + Math.round(guard.sqiConsensus * 100)
                  + "%). Inspect before trusting the clean score."
            color: "#E0A32E"
            font.family: Theme.fontUi; font.pixelSize: 10
            lineHeight: 1.3; wrapMode: Text.WordWrap
        }
    }

    // -- the bank (quality-fill bars: full green = good, empty red = poor) ---
    Repeater {
        model: root.rowsShown
        delegate: RowLayout {
            required property var modelData
            Layout.fillWidth: true
            spacing: 8
            Text {
                text: modelData.name
                color: Theme.textSecondary
                font.family: Theme.fontUi; font.pixelSize: 11
                Layout.preferredWidth: 84
                ToolTip.visible: nameMa.containsMouse
                ToolTip.text: modelData.desc + "  (" + modelData.hint + ")"
                MouseArea { id: nameMa; anchors.fill: parent; hoverEnabled: true }
            }
            Rectangle {                                // quality fill: FULL green = good, empty red = poor
                Layout.fillWidth: true
                height: 6; radius: 3; color: Theme.bgPanel
                Rectangle {
                    height: parent.height; radius: 3
                    width: Math.max(2, parent.width * Math.min(1, Math.max(0, modelData.bar)))
                    color: modelData.bar > 0.66 ? "#2FBF71"                 // green = good quality
                         : (modelData.bar > 0.33 ? "#E0A32E" : "#E5484D")   // amber → red = poor
                }
            }
            Text {
                text: "" + modelData.value
                color: Theme.textPrimary
                font.family: Theme.fontMono; font.pixelSize: 11
                Layout.preferredWidth: 44; horizontalAlignment: Text.AlignRight
            }
        }
    }

    Text {
        Layout.fillWidth: true
        text: root.showFiltered
            ? "Computed on a band-pass-filtered copy — compare with Raw to see what a filter would fix."
            : "Classical indices for this window — explanatory; they don't change the grade."
        color: Theme.textMuted; font.family: Theme.fontUi; font.pixelSize: 9
        wrapMode: Text.WordWrap
    }
}
