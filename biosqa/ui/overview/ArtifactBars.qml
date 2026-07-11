import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../"

// Overview artifact breakdown (mockup lines 382-389): a labelled horizontal bar per
// artifact type, filled in that artifact's own color over a dark track. Fed by
// `segments.artifactBars` ([{label, value}]); since the viewmodel does not attach a
// color, one is derived, and a static mockup list is shown until inference runs.
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

    readonly property var _static: [
        { name: "Baseline wander",   count: 412, color: "#E0A32E" },
        { name: "Motion",            count: 288, color: "#E5484D" },
        { name: "Muscle noise",      count: 196, color: "#E0A32E" },
        { name: "Sensor saturation", count: 124, color: "#E5484D" },
        { name: "Powerline 50 Hz",   count: 92,  color: "#86C440" }
    ]

    // palette used to tint real artifact rows that arrive without a color.
    readonly property var _tint: ["#E0A32E", "#E5484D", "#86C440", "#6E8BFF", "#C08CF2"]

    readonly property var _rows: {
        if (!root.bars || root.bars.length === 0)
            return root._static
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
                color: "#c5cdda"
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
