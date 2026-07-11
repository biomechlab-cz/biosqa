import QtQuick
import QtQuick.Layouts
import "../"
import "../components"

// Overview per-modality quality timeline (mockup lines 364-375): the REAL loaded
// modality as a prominent quality ribbon over the whole recording, with a time axis
// and a per-tier duration breakdown so the tile reads as a full dashboard panel.
ColumnLayout {
    id: root
    spacing: 14

    readonly property string _mod: recordings.currentModality || ""
    readonly property real _dur: recordings.currentDurationSec

    function _fmtHMS(sec) {
        sec = Math.max(0, Math.round(sec || 0))
        var h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60
        function p(n) { return (n < 10 ? "0" : "") + n }
        return p(h) + ":" + p(m) + ":" + p(s)
    }
    function _pct(k) {
        var f = segments.tierFractions
        return (f && f[k] > 0) ? (f[k] * 100).toFixed(1) + "%" : "0%"
    }
    function _usablePct() {
        var f = segments.tierFractions
        if (f && Object.keys(f).length > 0)
            return (((f["Q3"] || 0) + (f["Q2"] || 0)) * 100).toFixed(1) + "%"
        return "—"
    }

    // ---- empty hint (no modality loaded) -------------------------------------
    Text {
        visible: root._mod.length === 0
        Layout.fillWidth: true
        Layout.topMargin: 8
        text: "No recording loaded — run inference to see the quality timeline."
        color: Theme.textMuted
        font.family: Theme.fontUi
        font.pixelSize: 12
        wrapMode: Text.WordWrap
    }

    // ---- modality + usable% --------------------------------------------------
    RowLayout {
        visible: root._mod.length > 0
        Layout.fillWidth: true
        spacing: 8
        Rectangle {
            Layout.alignment: Qt.AlignVCenter
            width: 9; height: 9; radius: 2
            color: Theme.modalityColors[root._mod] ?? Theme.borderColor
        }
        Text {
            text: root._mod.toUpperCase()
            color: Theme.textPrimary
            font.family: Theme.fontMono
            font.pixelSize: 13
            font.weight: Font.DemiBold
        }
        Item { Layout.fillWidth: true }
        Text {
            text: root._usablePct() + " usable"
            color: Theme.textSecondary
            font.family: Theme.fontMono
            font.pixelSize: 12
        }
    }

    // ---- tall quality ribbon -------------------------------------------------
    QualitySparkline {
        visible: root._mod.length > 0
        Layout.fillWidth: true
        Layout.preferredHeight: 42
        radius: 4
        bands: segments.segmentBands
    }

    // ---- time axis -----------------------------------------------------------
    RowLayout {
        visible: root._mod.length > 0
        Layout.fillWidth: true
        Text {
            text: "00:00:00"; color: Theme.textMuted
            font.family: Theme.fontMono; font.pixelSize: 9
        }
        Item { Layout.fillWidth: true }
        Text {
            text: root._fmtHMS(root._dur / 2); color: Theme.textMuted
            font.family: Theme.fontMono; font.pixelSize: 9
        }
        Item { Layout.fillWidth: true }
        Text {
            text: root._fmtHMS(root._dur); color: Theme.textMuted
            font.family: Theme.fontMono; font.pixelSize: 9
        }
    }

    // ---- per-tier duration breakdown ----------------------------------------
    GridLayout {
        visible: root._mod.length > 0
        Layout.fillWidth: true
        Layout.topMargin: 4
        columns: 4
        columnSpacing: 14
        rowSpacing: 8
        Repeater {
            model: ["Q3", "Q2", "Q1", "Q0"]
            delegate: RowLayout {
                required property string modelData
                readonly property var p: Theme.currentQualityPalette()[modelData]
                Layout.fillWidth: true
                spacing: 6
                TierChip { tier: modelData; size: 14 }
                Text {
                    text: modelData
                    color: p.color
                    font.family: Theme.fontMono
                    font.pixelSize: 11
                    font.weight: Font.DemiBold
                }
                Text {
                    text: root._pct(modelData)
                    color: Theme.textSecondary
                    font.family: Theme.fontMono
                    font.pixelSize: 11
                    Layout.fillWidth: true
                }
            }
        }
    }

    Item { Layout.fillHeight: true }
}
