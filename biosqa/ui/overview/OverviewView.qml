import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../"
import "../components"

// design spec (a): "OverviewView -- bento KPI dashboard" (mockup lines 315-405).
// A scrollable page: title + mono meta, a 5-up KPI row, then the bento grid
// (donut hero / per-modality timeline / artifact types / model card).
//
// The mockup's 6th KPI ("Windows") is deliberately absent -- see the KPI row below.
Rectangle {
    id: root
    objectName: "overviewView"
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

    // Has the segmenter actually produced intervals? `segments.tierFractions` is {} until it has,
    // and every KPI below that summarises the grading must read "—" (not a defaulted 0) until then.
    readonly property bool hasStats: {
        var f = segments.tierFractions
        return !!f && Object.keys(f).length > 0
    }
    // is a recording open at all? (duration/name are 0/"" before the first open)
    readonly property bool hasRecording: recordings.currentName && recordings.currentName.length > 0

    // usable share (Q3 + Q2) from the real tier fractions, as a "NN.N%" string.
    function _usablePct() {
        if (!root.hasStats)
            return "—"
        var f = segments.tierFractions
        return (((f["Q3"] || 0) + (f["Q2"] || 0)) * 100).toFixed(1) + "%"
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
                    text: root.hasRecording ? recordings.currentName : "No recording loaded"
                    color: Theme.textMuted
                    font.family: Theme.fontMono
                    font.pixelSize: 12
                    elide: Text.ElideMiddle
                    Layout.maximumWidth: 420
                }
                Item { Layout.fillWidth: true }
            }

            // ---- KPI row (5-up) --------------------------------------------
            //
            // There is NO "Windows" KPI. It used to read `duration / modelCard.windowSec`, which
            // ignores the window OVERLAP entirely -- and the app ships 50% overlap by default
            // (settings_controller.py: "analysis/windowOverlap": 0.5), so the model actually runs
            // ~2x the windows that tile claimed. The true count is
            // `(n_samples - L_m) // stride + 1` over the RESAMPLED inference channel, computed in
            // Coordinator._run_inference and kept in its private `_pending` map; no context
            // property (recordings/segments/modelCard/inference/settings) exposes it, and the
            // streaming path derives its own. Rather than re-derive a formula in QML from the
            // record duration -- a different quantity, and a guess -- the stat is gone. Restore it
            // as a real KPI only once a controller publishes the n_windows inference really used.
            GridLayout {
                Layout.fillWidth: true
                columns: 5
                columnSpacing: 14
                rowSpacing: 14

                KpiCard {
                    objectName: "kpiDuration"
                    Layout.fillWidth: true
                    Layout.preferredHeight: 84
                    label: "Duration"
                    // no recording open => no duration; "00:00:00" is a claim about a signal
                    // that was never read.
                    value: root.hasRecording ? root.fmtHMS(recordings.currentDurationSec) : "—"
                    sublabel: "hh:mm:ss"
                }
                KpiCard {
                    objectName: "kpiUsable"
                    Layout.fillWidth: true
                    Layout.preferredHeight: 84
                    label: "% Usable"
                    valueColor: Theme.currentQualityPalette()["Q3"].color
                    value: root._usablePct()
                    sublabel: "Q3 + Q2"
                }
                KpiCard {
                    objectName: "kpiSegments"
                    Layout.fillWidth: true
                    Layout.preferredHeight: 84
                    label: "Segments"
                    // nothing segmented yet => "—", not a measured "0" run-length count
                    value: root.hasStats
                           ? Number(segments.totalCount).toLocaleString(Qt.locale(), "f", 0)
                           : "—"
                    sublabel: "run-length"
                }
                KpiCard {
                    objectName: "kpiChannels"
                    Layout.fillWidth: true
                    Layout.preferredHeight: 84
                    label: "Channels"
                    // no recording open => no channels; "6 / 4 modalities" was a mockup number
                    value: channels.count > 0 ? channels.count.toString() : "—"
                    sublabel: (recordings.currentModality && recordings.currentModality.length > 0)
                              ? recordings.currentModality.toUpperCase() : "—"
                }
                KpiCard {
                    objectName: "kpiFlaggedQ0"
                    Layout.fillWidth: true
                    Layout.preferredHeight: 84
                    label: "Flagged Q0"
                    valueColor: Theme.currentQualityPalette()["Q0"].color
                    value: root.hasStats
                           ? (((segments.tierFractions["Q0"] || 0) * 100).toFixed(1) + "%")
                           : "—"
                    // when the filtered pass finds recoverable poor segments, tease the count here
                    sublabel: !root.hasStats
                              ? "not measured"
                              : (segments.recoverableCount > 0
                                 ? ("↺ " + segments.recoverableCount + " recoverable")
                                 : "discard")
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
                        objectName: "qualityDonut"
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
                    // don't caption an unopened recording with a "00:00:00" span
                    subtitle: root.hasRecording
                              ? ("Continuous tier ribbon across "
                                 + root.fmtHMS(recordings.currentDurationSec))
                              : "Continuous tier ribbon across the recording"

                    ModalityRibbon {
                        objectName: "modalityRibbon"
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
                        objectName: "artifactBars"
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
