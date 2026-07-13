import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../"
import "../components"

// design spec (a)/(8.2): "manual override of predicted quality -> can be
// exported as new labels feeding Plan 1's active-learning loop". Calls
// straight through to `selection`'s QML-invokable slots (design spec (d)):
// acceptLabel(), relabel(tier), addNote(text).
ColumnLayout {
    id: root
    spacing: 9

    readonly property var seg: selection.selectedSegment
    readonly property color tierColor: seg ? seg.color : Theme.accent
    readonly property string tierKey: seg ? seg.tier : "Q3"
    readonly property var tiers: ["Q3", "Q2", "Q1", "Q0"]
    // The user's override (relabel records it but `seg.tier` keeps the model's
    // prediction), tracked locally so the active highlight reflects the choice.
    property string userTier: ""
    readonly property string effTier: userTier !== "" ? userTier : tierKey
    onSegChanged: userTier = ""

    Label {
        text: "YOUR DECISION"
        color: Theme.textMuted
        font.family: Theme.fontUi
        font.pixelSize: 11
        font.weight: Font.DemiBold
        font.letterSpacing: 0.7
    }

    // -- Accept AI label ----------------------------------------------------
    Button {
        id: acceptBtn
        Layout.fillWidth: true
        implicitHeight: 38
        padding: 0
        enabled: root.seg !== null
        onClicked: {
            selection.acceptLabel()
            root.userTier = root.tierKey
            AppController.toast("AI label accepted (" + root.tierKey + ")")
        }

        contentItem: RowLayout {
            spacing: 8
            Item { Layout.fillWidth: true }
            Text {
                text: "✓"
                color: root.tierColor
                font.pixelSize: 13
                font.bold: true
            }
            Text {
                text: "Accept AI label (" + root.tierKey + ")"
                color: root.tierColor
                font.family: Theme.fontUi
                font.pixelSize: 12
                font.weight: Font.DemiBold
            }
            Item { Layout.fillWidth: true }
        }
        background: Rectangle {
            radius: 8
            color: Qt.rgba(root.tierColor.r, root.tierColor.g, root.tierColor.b,
                           acceptBtn.hovered ? 0.20 : 0.14)
            border.color: root.tierColor
            border.width: 1
        }
    }

    Label {
        text: "Or re-label to a different tier"
        color: Theme.textMuted
        font.family: Theme.fontUi
        font.pixelSize: 11
    }

    // -- relabel 2x2 grid ---------------------------------------------------
    GridLayout {
        Layout.fillWidth: true
        columns: 2
        rowSpacing: 7
        columnSpacing: 7

        Repeater {
            model: root.tiers
            delegate: Button {
                id: tierBtn
                required property string modelData
                readonly property bool active: root.effTier === modelData
                readonly property color tc: Theme.tierInfo(modelData).color
                Layout.fillWidth: true
                implicitHeight: 40
                padding: 0
                enabled: root.seg !== null
                onClicked: {
                    selection.relabel(modelData)
                    root.userTier = modelData
                    AppController.toast("Relabeled to " + modelData + " · queued for retraining")
                }

                contentItem: RowLayout {
                    spacing: 6
                    Item { Layout.fillWidth: true }
                    TierChip { tier: tierBtn.modelData; size: 16 }
                    Text {
                        text: tierBtn.modelData
                        color: Theme.textPrimary
                        font.family: Theme.fontMono
                        font.pixelSize: 11
                        font.weight: Font.DemiBold
                    }
                    Text {
                        text: Theme.tierInfo(tierBtn.modelData).label
                        color: Theme.textSecondary
                        font.family: Theme.fontUi
                        font.pixelSize: 10
                    }
                    Item { Layout.fillWidth: true }
                }
                background: Rectangle {
                    radius: 7
                    color: tierBtn.active
                        ? Qt.rgba(tierBtn.tc.r, tierBtn.tc.g, tierBtn.tc.b, 0.14)
                        : Theme.bgPanelAlt
                    border.width: 1
                    border.color: tierBtn.active
                        ? tierBtn.tc
                        : (tierBtn.hovered ? tierBtn.tc : Theme.borderColor)
                }
            }
        }
    }

    // Segment RESHAPING (trim / split / merge / reclassify) now lives in the full Segment Inspector
    // view (inspector/SegmentInspectorView.qml → components/SegmentReshape), not this workspace dock.

    // -- note ---------------------------------------------------------------
    TextArea {
        id: noteField
        Layout.fillWidth: true
        Layout.preferredHeight: 56
        placeholderText: "Add a note..."
        wrapMode: TextArea.Wrap
        color: Theme.textPrimary
        font.family: Theme.fontUi
        font.pixelSize: 12
        background: Rectangle {
            radius: 7
            color: Theme.bgPanelAlt
            border.width: 1
            border.color: noteField.activeFocus ? Theme.accent : Theme.chipBorderMuted
        }
    }

    OutlineButton {
        Layout.fillWidth: true
        text: "Save note"
        enabled: root.seg !== null && noteField.text.length > 0
        onClicked: {
            selection.addNote(noteField.text)
            AppController.toast("Note saved")
            noteField.text = ""
        }
    }

    // -- override banner ----------------------------------------------------
    Rectangle {
        Layout.fillWidth: true
        visible: root.seg && root.seg.overridden
        radius: 7
        color: Qt.rgba(Theme.accent.r, Theme.accent.g, Theme.accent.b, 0.06)
        border.color: Qt.rgba(Theme.accent.r, Theme.accent.g, Theme.accent.b, 0.25)
        border.width: 1
        implicitHeight: bannerRow.implicitHeight + 20

        RowLayout {
            id: bannerRow
            anchors.fill: parent
            anchors.leftMargin: 12
            anchors.rightMargin: 12
            anchors.topMargin: 10
            anchors.bottomMargin: 10
            spacing: 9
            Text {
                text: "✓"
                color: Theme.accent
                font.pixelSize: 12
                font.bold: true
            }
            Text {
                text: "Correction queued for retraining · relabeled to " + root.effTier
                color: Theme.tealText
                font.family: Theme.fontUi
                font.pixelSize: 11
                wrapMode: Text.WordWrap
                Layout.fillWidth: true
            }
        }
    }
}
