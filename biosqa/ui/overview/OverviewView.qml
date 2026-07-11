import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../"
import "../components"

// design spec (a): "OverviewView -- bento KPI dashboard" (mockup lines 315-405).
// A scrollable page: title + mono meta, a 6-up KPI row, then the bento grid
// (donut hero / per-modality timeline / artifact types / model card).
Rectangle {
    id: root
    color: Theme.bgApp

    // HH:MM:SS formatter for real durations from the recordings viewmodel.
    function fmtHMS(sec) {
        sec = Math.max(0, Math.round(sec || 0))
        var h = Math.floor(sec / 3600)
        var m = Math.floor((sec % 3600) / 60)
        var s = sec % 60
        function p(n) { return (n < 10 ? "0" : "") + n }
        return p(h) + ":" + p(m) + ":" + p(s)
    }

    // usable share (Q3 + Q2) from the real tier fractions, as a "NN.N%" string.
    function _usablePct() {
        var f = segments.tierFractions
        if (f && Object.keys(f).length > 0)
            return (((f["Q3"] || 0) + (f["Q2"] || 0)) * 100).toFixed(1) + "%"
        return "—"
    }

    ScrollView {
        id: sv
        anchors.fill: parent
        contentWidth: availableWidth
        clip: true
        topPadding: 22
        bottomPadding: 22
        leftPadding: 26
        rightPadding: 26
        ScrollBar.horizontal.policy: ScrollBar.AlwaysOff

        ColumnLayout {
            width: sv.availableWidth
            spacing: 16

            // ---- page header -----------------------------------------------
            RowLayout {
                Layout.fillWidth: true
                Layout.bottomMargin: 4
                spacing: 12
                Label {
                    text: "Recording Overview"
                    color: Theme.textPrimary
                    font.family: Theme.fontUi
                    font.pixelSize: 19
                    font.weight: Font.DemiBold
                    font.letterSpacing: -0.2
                }
                Label {
                    text: recordings.currentName && recordings.currentName.length > 0
                          ? recordings.currentName : "No recording loaded"
                    color: Theme.textMuted
                    font.family: Theme.fontMono
                    font.pixelSize: 12
                    elide: Text.ElideMiddle
                    Layout.maximumWidth: 420
                }
                Item { Layout.fillWidth: true }
            }

            // ---- KPI row (6-up) --------------------------------------------
            GridLayout {
                Layout.fillWidth: true
                columns: 6
                columnSpacing: 14
                rowSpacing: 14

                KpiCard {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 84
                    label: "Duration"
                    value: root.fmtHMS(recordings.currentDurationSec)
                    sublabel: "hh:mm:ss"
                }
                KpiCard {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 84
                    label: "% Usable"
                    valueColor: Theme.currentQualityPalette()["Q3"].color
                    value: root._usablePct()
                    sublabel: "Q3 + Q2"
                }
                KpiCard {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 84
                    label: "Segments"
                    value: Number(segments.totalCount).toLocaleString(Qt.locale(), "f", 0)
                    sublabel: "run-length"
                }
                KpiCard {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 84
                    label: "Channels"
                    value: channels.count > 0 ? channels.count.toString() : "6"
                    sublabel: "4 modalities"
                }
                KpiCard {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 84
                    label: "Windows"
                    value: modelCard.windowSec > 0
                           ? Number(Math.round((recordings.currentDurationSec || 0) / modelCard.windowSec))
                             .toLocaleString(Qt.locale(), "f", 0)
                           : "—"
                    sublabel: (modelCard.windowSec > 0
                               ? (modelCard.windowSec < 10 ? modelCard.windowSec.toFixed(1)
                                                           : modelCard.windowSec.toFixed(0))
                               : "—") + " s / win"
                }
                KpiCard {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 84
                    label: "Flagged Q0"
                    valueColor: Theme.currentQualityPalette()["Q0"].color
                    value: {
                        var f = segments.tierFractions
                        if (f && Object.keys(f).length > 0)
                            return ((f["Q0"] || 0) * 100).toFixed(1) + "%"
                        return "—"
                    }
                    // when the filtered pass finds recoverable poor segments, tease the count here
                    sublabel: segments.recoverableCount > 0
                              ? ("↺ " + segments.recoverableCount + " recoverable")
                              : "discard"
                }
            }

            // ---- bento grid ------------------------------------------------
            GridLayout {
                id: bento
                Layout.fillWidth: true
                columns: 3
                columnSpacing: 14
                rowSpacing: 14
                readonly property real gap: 14
                // derive column width from the STABLE ScrollView width, never from the
                // grid's own `width` (that binds the layout to its own output -> recursive).
                readonly property real colW: (sv.availableWidth - 2 * gap) / 3.15

                // donut hero (col 0, spans both rows)
                GlassPanel {
                    Layout.row: 0
                    Layout.column: 0
                    Layout.rowSpan: 2
                    Layout.preferredWidth: bento.colW * 1.15
                    Layout.fillHeight: true
                    pad: 20
                    title: "Quality distribution"
                    subtitle: "Share of recording per tier"

                    DonutChart {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        fractions: segments.tierFractions
                    }
                }

                // per-modality timeline (row 0, spans cols 1-2)
                GlassPanel {
                    Layout.row: 0
                    Layout.column: 1
                    Layout.columnSpan: 2
                    Layout.preferredWidth: bento.colW * 2 + bento.gap
                    Layout.preferredHeight: 235
                    pad: 20
                    title: "Per-modality quality timeline"
                    subtitle: "Continuous tier ribbon across " + root.fmtHMS(recordings.currentDurationSec)

                    ModalityRibbon {
                        Layout.fillWidth: true
                        Layout.topMargin: 4
                    }
                }

                // artifact types (row 1, col 1)
                GlassPanel {
                    Layout.row: 1
                    Layout.column: 1
                    Layout.preferredWidth: bento.colW
                    Layout.preferredHeight: 250
                    pad: 20
                    title: "Artifact types"
                    subtitle: "Across flagged windows"

                    ArtifactBars {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        Layout.topMargin: 4
                        bars: segments.artifactBars
                    }
                }

                // model card (row 1, col 2)
                ModelCardPanel {
                    Layout.row: 1
                    Layout.column: 2
                    Layout.preferredWidth: bento.colW
                    Layout.preferredHeight: 250
                }
            }
        }
    }
}
