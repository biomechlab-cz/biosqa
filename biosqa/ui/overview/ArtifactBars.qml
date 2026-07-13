import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../"

// Overview artifact breakdown (mockup lines 382-389): a labelled horizontal bar per
// artifact type, filled in that artifact's own color over a dark track. Fed by
// `segments.artifactBars` ([{label, value}]); since the viewmodel does not attach a
// color, one is derived.
//
// There is NO fallback list: this tile used to show a hardcoded mockup breakdown
// ("Baseline wander 412, Motion 288, ...") whenever inference had not run, i.e. it
// invented artifact counts. No data now renders as an explicit empty state.
//
// The list is a CLIPPED, scrollable ListView so that an arbitrary number of artifact
// rows (7+) can never spill below the tile border -- it always fits within the tile.
ListView {
    id: root
    clip: true
    spacing: 12
    interactive: true
    boundsBehavior: Flickable.StopAtBounds
    model: root._rows

    ScrollBar.vertical: ScrollBar { policy: ScrollBar.AsNeeded }

    // [{ label, value }] from the viewmodel (no color); mapped to {name, count, color}.
    property var bars: []

    readonly property bool hasData: root.bars && root.bars.length > 0

    // palette used to tint real artifact rows that arrive without a color.
    readonly property var _tint: ["#E0A32E", "#E5484D", "#86C440", "#6E8BFF", "#C08CF2"]

    readonly property var _rows: {
        if (!root.hasData)
            return []
        var out = []
        for (var i = 0; i < root.bars.length; i++) {
            var b = root.bars[i]
            out.push({
                name: b.name !== undefined ? b.name : b.label,
                count: b.count !== undefined ? b.count : b.value,
                color: b.color !== undefined ? b.color : root._tint[i % root._tint.length]
            })
        }
        return out
    }

    readonly property real _max: {
        var m = 1
        for (var i = 0; i < root._rows.length; i++)
            m = Math.max(m, root._rows[i].count)
        return m
    }

    // ---- empty state ---------------------------------------------------------
    Text {
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.top: parent.top
        anchors.topMargin: 6
        visible: !root.hasData
        text: "No artifacts detected — run inference."
        color: Theme.textMuted
        font.family: Theme.fontUi
        font.pixelSize: 12
        wrapMode: Text.WordWrap
    }

    delegate: ColumnLayout {
        id: barRow
        required property var modelData
        width: ListView.view.width
        spacing: 5

        RowLayout {
            Layout.fillWidth: true
            spacing: 8
            Text {
                text: barRow.modelData.name
                color: Theme.textBody
                font.family: Theme.fontUi
                font.pixelSize: 12
                elide: Text.ElideRight
                Layout.fillWidth: true
            }
            Text {
                text: Number(barRow.modelData.count).toLocaleString(Qt.locale(), "f", 0)
                color: Theme.textSecondary
                font.family: Theme.fontMono
                font.pixelSize: 11
            }
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 6
            radius: 3
            color: Theme.bgPanelAlt
            clip: true

            Rectangle {
                anchors.left: parent.left
                anchors.top: parent.top
                anchors.bottom: parent.bottom
                width: parent.width * (barRow.modelData.count / root._max)
                radius: 3
                color: barRow.modelData.color
            }
        }
    }
}
