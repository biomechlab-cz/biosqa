import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../"
import "../components"

// design spec (a)/(d): filter pill group + "jump to next poor" + "export
// selection". Drives `segments.setFilter(id)`, `segments.jumpToNextPoor(sec)`
// and `exporter.exportSelection(...)`.
RowLayout {
    id: root
    spacing: 10

    readonly property var filterDefs: [
        { id: "all", label: "All" },
        { id: "q0", label: "Only Q0" },
        { id: "poor", label: "Poor (Q0–Q1)" },
        { id: "usable", label: "Usable (Q2–Q3)" },
        { id: "recoverable", label: "↺ Recoverable" }
    ]

    // ---- pill group -----------------------------------------------------
    Rectangle {
        Layout.alignment: Qt.AlignVCenter
        implicitWidth: pillRow.implicitWidth + 8
        implicitHeight: pillRow.implicitHeight + 8
        color: Theme.bgPanelAlt
        border.color: Theme.borderColor
        border.width: 1
        radius: 8

        RowLayout {
            id: pillRow
            anchors.centerIn: parent
            spacing: 6

            Repeater {
                model: root.filterDefs
                delegate: Button {
                    id: pill
                    required property var modelData
                    readonly property bool active: segments.filterId === modelData.id
                    padding: 0
                    implicitHeight: 26
                    implicitWidth: pillLabel.implicitWidth + 24
                    onClicked: segments.setFilter(pill.modelData.id)
                    background: Rectangle {
                        radius: 6
                        color: pill.active ? Theme.accent : "transparent"
                    }
                    contentItem: Text {
                        id: pillLabel
                        text: pill.modelData.label
                        color: pill.active ? Theme.chipDark : Theme.textSecondary
                        font.family: Theme.fontUi
                        font.pixelSize: 12
                        font.weight: Font.Medium
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                    }
                }
            }
        }
    }

    // ---- jump to next poor region --------------------------------------
    Button {
        id: jumpBtn
        Layout.alignment: Qt.AlignVCenter
        padding: 0
        implicitHeight: 32
        implicitWidth: jumpRow.implicitWidth + 24
        onClicked: {
            // advance from the currently-selected segment (so repeated clicks step through)
            var from = selection.selectedSegment ? selection.selectedSegment.startSec
                                                 : signalView.viewStartSec
            var idx = segments.nextPoorIndex(from)
            if (idx >= 0) {
                selection.selectByAllIndex(idx)     // highlights the row, track block, and plot band
                var s = selection.selectedSegment
                if (s && s.endSec > s.startSec) {
                    var w = signalView.viewEndSec - signalView.viewStartSec
                    signalView.setView(s.startSec, s.startSec + (w > 0 ? w : 10))
                    AppController.toast("Next poor region · " + s.tier + " at " + Math.round(s.startSec) + "s")
                }
            } else {
                AppController.toast("No poor region after the current selection")
            }
        }
        background: Rectangle {
            radius: 8
            color: jumpBtn.hovered ? Theme.hoverBg : Theme.bgPanel
            border.color: Theme.borderColor
            border.width: 1
        }
        contentItem: RowLayout {
            id: jumpRow
            spacing: 7
            Item { Layout.fillWidth: true }
            Rectangle {
                Layout.alignment: Qt.AlignVCenter
                Layout.preferredWidth: 8
                Layout.preferredHeight: 8
                radius: 4
                color: Theme.currentQualityPalette()["Q0"].color
            }
            Text {
                text: "Jump to next poor region"
                color: Theme.textPrimary
                font.family: Theme.fontUi
                font.pixelSize: 12
                verticalAlignment: Text.AlignVCenter
            }
            Item { Layout.fillWidth: true }
        }
    }

    Item { Layout.fillWidth: true }

    // ---- export selected-quality segments (format menu) ----------------
    AccentButton {
        id: exportSelBtn
        Layout.alignment: Qt.AlignVCenter
        text: "Export selection"
        glyph: "⤓"
        onClicked: exportSelMenu.popup(exportSelBtn, 0, exportSelBtn.height + 2)
    }
    Menu {
        id: exportSelMenu
        Repeater {
            model: [
                { label: "CSV table", fmt: "csv" },
                { label: "TSV (BIDS events)", fmt: "tsv" },
                { label: "JSON quality report", fmt: "json" },
                { label: "Parquet", fmt: "parquet" },
                { label: "WFDB annotation", fmt: "wfdb" },
                { label: "MATLAB .mat", fmt: "mat" }
            ]
            delegate: MenuItem {
                required property var modelData
                text: modelData.label
                onTriggered: exporter.exportSelection(modelData.fmt)
            }
        }
    }
}
