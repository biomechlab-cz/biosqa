import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../"

// Structural edits to the segmentation itself (distinct from a soft relabel): trim a boundary,
// split, merge, or reclassify a piece. Self-contained + bound to `selection.selectedSegment`, so
// it drops into any inspector. Mutates the overlay live (plot bands, table, exports); a
// re-inference discards edits, exactly like a relabel.
ColumnLayout {
    id: root
    spacing: 8

    readonly property var seg: selection.selectedSegment
    readonly property var tiers: ["Q3", "Q2", "Q1", "Q0"]
    readonly property real editStep: 1.0   // seconds per nudge

    Label {
        text: "RESHAPE SEGMENT"
        color: Theme.textMuted
        font.family: Theme.fontUi; font.pixelSize: 11; font.weight: Font.DemiBold; font.letterSpacing: 0.7
    }
    Label {
        Layout.fillWidth: true
        text: "Trim an edge, split, or absorb a neighbour — e.g. drag a Q0/Q2 boundary or merge to keep only the Q0."
        wrapMode: Text.WordWrap
        color: Theme.textMuted; font.family: Theme.fontUi; font.pixelSize: 10
    }

    // start / end nudge steppers
    GridLayout {
        Layout.fillWidth: true
        columns: 4
        columnSpacing: 6
        rowSpacing: 6

        component EdgeStepper: Button {
            id: sb
            required property string glyph
            required property var act
            enabled: root.seg !== null
            implicitWidth: 30; implicitHeight: 28; padding: 0
            onClicked: sb.act()
            contentItem: Text {
                text: sb.glyph; color: Theme.textPrimary
                font.family: Theme.fontMono; font.pixelSize: 14
                horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter
            }
            background: Rectangle {
                radius: 6; color: sb.hovered ? Theme.hoverBg : Theme.bgPanelAlt
                border.width: 1; border.color: Theme.borderColor
            }
        }

        Label {
            text: "Start"; color: Theme.textSecondary
            font.family: Theme.fontUi; font.pixelSize: 11; Layout.preferredWidth: 34
        }
        EdgeStepper { glyph: "−"; act: () => selection.nudgeSelectedStart(-root.editStep) }
        Label {
            text: root.seg ? root.seg.startSec.toFixed(1) + "s" : "—"
            color: Theme.textPrimary; horizontalAlignment: Text.AlignHCenter
            font.family: Theme.fontMono; font.pixelSize: 11; Layout.fillWidth: true
        }
        EdgeStepper { glyph: "+"; act: () => selection.nudgeSelectedStart(root.editStep) }

        Label {
            text: "End"; color: Theme.textSecondary
            font.family: Theme.fontUi; font.pixelSize: 11; Layout.preferredWidth: 34
        }
        EdgeStepper { glyph: "−"; act: () => selection.nudgeSelectedEnd(-root.editStep) }
        Label {
            text: root.seg ? root.seg.endSec.toFixed(1) + "s" : "—"
            color: Theme.textPrimary; horizontalAlignment: Text.AlignHCenter
            font.family: Theme.fontMono; font.pixelSize: 11; Layout.fillWidth: true
        }
        EdgeStepper { glyph: "+"; act: () => selection.nudgeSelectedEnd(root.editStep) }
    }

    // split / merge
    RowLayout {
        Layout.fillWidth: true
        spacing: 7
        OutlineButton {
            Layout.fillWidth: true
            text: "⑃ Split in half"
            enabled: root.seg !== null
            onClicked: { selection.splitSelected(); AppController.toast("Segment split") }
        }
        OutlineButton {
            Layout.fillWidth: true
            text: "⇥ Merge next"
            enabled: root.seg !== null
            onClicked: { selection.mergeSelectedNext(); AppController.toast("Merged with next segment") }
        }
    }

    // reclassify this piece (changes the band, unlike a soft relabel)
    Label {
        text: "Reclassify this piece"
        color: Theme.textMuted; font.family: Theme.fontUi; font.pixelSize: 11
    }
    RowLayout {
        Layout.fillWidth: true
        spacing: 6
        Repeater {
            model: root.tiers
            delegate: Button {
                id: rcBtn
                required property string modelData
                readonly property color tc: Theme.tierInfo(modelData).color
                Layout.fillWidth: true
                implicitHeight: 30; padding: 0
                enabled: root.seg !== null
                onClicked: {
                    selection.reclassifySelected(modelData)
                    AppController.toast("Reclassified to " + modelData)
                }
                contentItem: RowLayout {
                    spacing: 5
                    Item { Layout.fillWidth: true }
                    TierChip { tier: rcBtn.modelData; size: 14 }
                    Text {
                        text: rcBtn.modelData
                        color: Theme.textPrimary
                        font.family: Theme.fontMono; font.pixelSize: 11; font.weight: Font.DemiBold
                    }
                    Item { Layout.fillWidth: true }
                }
                background: Rectangle {
                    radius: 6; color: rcBtn.hovered ? Qt.rgba(rcBtn.tc.r, rcBtn.tc.g, rcBtn.tc.b, 0.14)
                                                    : Theme.bgPanelAlt
                    border.width: 1; border.color: rcBtn.hovered ? rcBtn.tc : Theme.borderColor
                }
            }
        }
    }
}
