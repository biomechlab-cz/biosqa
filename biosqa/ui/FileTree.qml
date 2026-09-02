import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import "components"

// design spec (a): FileTreePanel (left dock, 248px) "Recordings" list.
// Pure-QML delegate bound to the Python `recordings` (RecordingListModel)
// context property (design spec (b)) -- no behavior lives here.
Rectangle {
    id: root
    color: Theme.bgPanelAlt
    border.color: Theme.borderColor
    border.width: 1

    function _fmtDur(sec) {
        sec = Math.max(0, Math.round(sec))
        const m = Math.floor(sec / 60)
        const s = sec % 60
        return m + ":" + (s < 10 ? "0" : "") + s
    }

    // Modality chosen from the "Open ▾" menu ("" = auto-detect), passed to recordings.open().
    property string _pendingModality: ""
    function _openWith(mod) { root._pendingModality = mod; openDialog.open() }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 12
        spacing: 8

        Label {
            text: "RECORDINGS"
            color: Theme.textMuted
            font.family: Theme.fontUi
            font.pixelSize: 11
            font.weight: Font.DemiBold
            font.letterSpacing: 0.8
        }

        ListView {
            id: recList
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            model: recordings
            spacing: 1
            delegate: Rectangle {
                id: recRow
                required property int index
                required property string name
                required property string path
                required property real durationSec
                width: recList.width
                height: 34
                radius: 6
                // Selected = the app's current recording (was ListView.isCurrentItem,
                // which was never set → the highlight never showed).
                color: (rowMA.containsMouse || recordings.currentPath === path)
                       ? Theme.hoverBg : "transparent"

                MouseArea {
                    id: rowMA
                    anchors.fill: parent
                    hoverEnabled: true
                    cursorShape: Qt.PointingHandCursor
                    onClicked: recordings.open(recRow.path)
                }

                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: 8
                    anchors.rightMargin: 8
                    spacing: 8
                    Text {
                        text: "▸"
                        color: Theme.textMuted
                        font.pixelSize: 10
                        Layout.alignment: Qt.AlignVCenter
                    }
                    Label {
                        text: recRow.name
                        color: Theme.textPrimary
                        font.family: Theme.fontMono
                        font.pixelSize: 12
                        elide: Text.ElideRight
                        Layout.fillWidth: true
                        Layout.alignment: Qt.AlignVCenter
                    }
                    Label {
                        text: root._fmtDur(recRow.durationSec)
                        color: Theme.textMuted
                        font.family: Theme.fontMono
                        font.pixelSize: 10
                        Layout.alignment: Qt.AlignVCenter
                    }
                }
            }

            Label {
                anchors.centerIn: parent
                visible: recordings.count === 0
                text: "No recordings open.\nDrop a file here or use File > Open."
                color: Theme.textMuted
                horizontalAlignment: Text.AlignHCenter
                font.family: Theme.fontUi
                font.pixelSize: 12
            }
        }

        OutlineButton {
            id: openBtn
            Layout.fillWidth: true
            text: "Open recording  ▾"
            onClicked: openMenu.popup(openBtn, 0, openBtn.height + 2)
        }

        // Signal-type picker. Auto-detection is NOT a manual choice — it always runs in the
        // background and warns (via recordings.modalityMismatch) if the header disagrees.
        Menu {
            id: openMenu
            padding: 4
            background: Rectangle {
                implicitWidth: 168
                color: Theme.bgPanel
                border.color: Theme.borderColor
                border.width: 1
                radius: Theme.radiusControl
            }
            Repeater {
                model: [
                    { label: "ECG",  mod: "ecg" },
                    { label: "EEG",  mod: "eeg" },
                    { label: "PPG",  mod: "ppg" },
                    { label: "EDA",  mod: "eda" }
                ]
                delegate: MenuItem {
                    id: mi
                    required property var modelData
                    implicitHeight: 32
                    onTriggered: root._openWith(modelData.mod)
                    background: Rectangle {
                        radius: 6
                        color: mi.highlighted ? Theme.hoverBg : "transparent"
                    }
                    contentItem: Text {
                        text: mi.modelData.label
                        color: Theme.textPrimary
                        font.family: Theme.fontUi
                        font.pixelSize: 13
                        leftPadding: 10
                        verticalAlignment: Text.AlignVCenter
                    }
                }
            }
        }

        Label {
            Layout.fillWidth: true
            text: "Sample ECG/PPG/EEG/EDA recordings live in the app's dummy_data/ folder — open one above to try it."
            wrapMode: Text.WordWrap
            color: Theme.textMuted
            font.family: Theme.fontUi
            font.pixelSize: 11
            Layout.topMargin: 2
        }
    }

    FileDialog {
        id: openDialog
        objectName: "openDialog"
        title: root._pendingModality ? ("Open " + root._pendingModality.toUpperCase() + " recording") : "Open recording"
        nameFilters: [
            "Recordings (*.hea *.dat *.edf *.bdf *.gdf *.vhdr *.set *.fif *.parquet)",
            "WFDB (*.hea *.dat)", "EDF/BDF (*.edf *.bdf)", "EEG (*.gdf *.vhdr *.set *.fif)",
            "All files (*)"
        ]
        onAccepted: recordings.open(
            decodeURIComponent(("" + selectedFile).replace("file:///", "")), root._pendingModality)
    }

    Shortcut {
        sequences: [StandardKey.Open]        // Ctrl+O opens a recording (auto-detect modality)
        context: Qt.ApplicationShortcut
        // an application-context shortcut fires straight THROUGH a modal popup, so it would open a
        // file dialog on top of the settings dialog
        enabled: !AppController.settingsOpen
        onActivated: root._openWith("")
    }
}
