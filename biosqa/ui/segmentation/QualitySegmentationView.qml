import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../"

// design spec (a): "QualitySegmentationView -- run-length timeline +
// filterable table". One of the four full-bleed views: a scrollable page of
// header + filter toolbar + run-length track + segment table.
ScrollView {
    id: root
    clip: true
    contentWidth: availableWidth
    // "table" | "grid" — bound to the persisted setting, so the last-used view is remembered.
    readonly property string segView: settings.segmentationView

    background: Rectangle { color: Theme.bgApp }

    Item {
        width: root.availableWidth
        implicitHeight: page.implicitHeight + 44   // 22 top + 22 bottom

        ColumnLayout {
            id: page
            x: 26
            y: 22
            width: parent.width - 52
            spacing: 0

            // ---- header ------------------------------------------------
            RowLayout {
                Layout.fillWidth: true
                spacing: 12
                Text {
                    text: "Quality Segmentation"
                    color: Theme.textPrimary
                    font.family: Theme.fontUi
                    font.pixelSize: 19
                    font.weight: Font.DemiBold
                    Layout.alignment: Qt.AlignBottom
                }
                Text {
                    text: segTable.count + " run-length segments · table of contents by quality"
                    color: Theme.textMuted
                    font.family: Theme.fontMono
                    font.pixelSize: 12
                    Layout.alignment: Qt.AlignBottom
                }
                Item { Layout.fillWidth: true }
                // Table / Grid view switch
                Row {
                    Layout.alignment: Qt.AlignBottom
                    spacing: 0
                    Repeater {
                        model: [{ k: "table", l: "Table" }, { k: "grid", l: "Grid" }]
                        delegate: Rectangle {
                            required property var modelData
                            readonly property bool active: root.segView === modelData.k
                            width: 58; height: 30
                            color: active ? Theme.accent : Theme.bgPanel
                            border.color: Theme.borderColor; border.width: 1
                            Text {
                                anchors.centerIn: parent; text: parent.modelData.l
                                color: parent.active ? Theme.chipDark : Theme.textSecondary
                                font.family: Theme.fontUi; font.pixelSize: 12
                            }
                            MouseArea {
                                anchors.fill: parent; cursorShape: Qt.PointingHandCursor
                                onClicked: settings.setSegmentationView(parent.modelData.k)
                            }
                        }
                    }
                }
            }

            // ---- intro paragraph ---------------------------------------
            Text {
                Layout.topMargin: 4
                Layout.maximumWidth: 640
                text: "Contiguous quality blocks across the full recording. Click a block or row "
                    + "to select it (double-click to open it in the inspector); filter, jump to "
                    + "poor regions, and export selected-quality segments."
                color: Theme.textSecondary
                font.family: Theme.fontUi
                font.pixelSize: 12
                wrapMode: Text.WordWrap
            }

            // ---- filter toolbar ----------------------------------------
            FilterToolbar {
                Layout.topMargin: 18
                Layout.fillWidth: true
            }

            // ---- run-length track --------------------------------------
            RunLengthTrack {
                Layout.topMargin: 16
                Layout.fillWidth: true
            }

            // ---- segment table / grid (switchable) ---------------------
            // The table is always instantiated (its `count` feeds the header meta) but hidden in
            // grid mode; the grid is lazy-loaded so it only builds thumbnails when actually shown.
            SegmentTable {
                id: segTable
                visible: root.segView === "table"
                Layout.topMargin: 20
                Layout.fillWidth: true
            }
            Loader {
                id: gridLoader
                active: root.segView === "grid"
                visible: active
                Layout.topMargin: 20
                Layout.fillWidth: true
                Layout.preferredHeight: item ? item.implicitHeight : 0
                sourceComponent: Component {
                    SegmentGrid { width: gridLoader.width }
                }
            }
        }
    }
}
