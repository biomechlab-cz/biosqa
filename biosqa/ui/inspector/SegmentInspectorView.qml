import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../"
import "../components"
import "../workspace"

// design spec (a): "SegmentInspectorView -- single-segment deep-dive +
// human-in-the-loop". Rebuilt (per design mockup lines 408-512) as a centered,
// single-column document rather than the workspace SignalView+dock reuse. All
// reads/writes still flow through the single `selection.selectedSegment` source
// of truth (design spec (b)); the tier-override rows and Save button call the
// same `selection.relabel()/acceptLabel()/addNote()` slots the dock used.
Rectangle {
    id: root
    color: Theme.bgApp

    readonly property var seg: selection.selectedSegment

    // The user's manually-set tier is not part of the viewmodel's segment
    // object (relabel() records an override but `tier` keeps reading the model's
    // prediction), so the "YOU SET" verdict is tracked locally here.
    property string userTier: ""

    readonly property string tierCode: seg ? seg.tier : "Q3"
    readonly property var tierInfo: Theme.currentQualityPalette()[tierCode]
                                    || Theme.currentQualityPalette()["Q3"]
    readonly property color tierColor: tierInfo.color
    readonly property string effTier: userTier !== "" ? userTier : tierCode
    readonly property var effInfo: Theme.currentQualityPalette()[effTier]
                                   || Theme.currentQualityPalette()["Q3"]

    function paletteFor(code) {
        return Theme.currentQualityPalette()[code] || Theme.currentQualityPalette()["Q3"]
    }
    function shortDesc(code) {
        return ({ "Q3": "All analysis", "Q2": "Rate / coarse",
                  "Q1": "With caution", "Q0": "Discard" })[code] || ""
    }
    function fmtTime(sec) {
        if (sec === undefined || sec === null || isNaN(sec)) return "--:--:--"
        var s = Math.max(0, sec)
        var h = Math.floor(s / 3600)
        var m = Math.floor((s % 3600) / 60)
        var ss = s % 60
        function p2(n) { return (n < 10 ? "0" : "") + Math.floor(n) }
        return p2(h) + ":" + p2(m) + ":" + (ss < 10 ? "0" : "") + ss.toFixed(2)
    }

    // real decimated samples for the selected segment's time range (null until fetched)
    property var segCurve: null
    onSegChanged: {
        userTier = ""
        segCurve = (seg && signalView.durationSec > 0)
                   ? signalView.curveForRange(seg.startSec, seg.endSec) : null
        tracePreview.requestPaint()
        if (seg)                                    // fill the classical-SQI breakdown for this segment
            guard.requestSqi(seg.startSec, seg.endSec)
    }

    // Plain-language description of each artifact TYPE (the model flags types per segment,
    // not per-sample positions) — makes the "detected artifacts" list informative.
    function artifactDesc(name) {
        var d = {
            "muscle_emg": "High-frequency EMG contamination from muscle activity.",
            "powerline": "Narrowband 50/60 Hz mains interference.",
            "baseline_wander": "Slow low-frequency baseline drift.",
            "motion": "Movement / motion artifact displacing the trace.",
            "clipping_flatline": "Amplifier saturation or a flat dropout region.",
            "burst_transient": "Short transient spikes or bursts.",
            "dropout": "Missing or zeroed samples (signal loss).",
            "electrode": "Electrode contact noise or pops.",
            "spike": "Isolated high-amplitude spikes.",
            "noise": "Broadband noise raising the signal floor."
        }
        return d[name] || (name + " detected across this segment.")
    }

    ScrollView {
        anchors.fill: parent
        clip: true
        contentWidth: availableWidth

        Item {
            width: root.width
            implicitHeight: doc.implicitHeight + 48

            ColumnLayout {
                id: doc
                width: Math.min(1320, root.width - 48)
                anchors.horizontalCenter: parent.horizontalCenter
                y: 24
                spacing: 18

                // -- breadcrumb ------------------------------------------------
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 10

                    Text {
                        id: crumb
                        text: "← Workspace"
                        color: crumbHover.hovered ? Theme.textPrimary : Theme.textSecondary
                        font.family: Theme.fontUi
                        font.pixelSize: 12
                        HoverHandler { id: crumbHover }
                        TapHandler { onTapped: AppController.go("workspace") }
                    }
                    Text {
                        text: "/"
                        color: Theme.chipBorderMuted
                        font.family: Theme.fontUi
                        font.pixelSize: 12
                    }
                    Text {
                        text: "Segment Inspector"
                        color: Theme.textSecondary
                        font.family: Theme.fontUi
                        font.pixelSize: 12
                    }
                }

                // -- empty state (no segment selected) -------------------------
                // BUG FIX: previously the whole detail block always rendered with a
                // fake "Q3 Excellent" waveform + badge even with nothing selected.
                // The detail block below is now gated on a real selection; this card
                // is shown in its place when `selection.selectedSegment` is null.
                Item {
                    visible: root.seg === null || root.seg === undefined
                    Layout.fillWidth: true
                    Layout.topMargin: 40
                    implicitHeight: 300

                    Rectangle {
                        anchors.horizontalCenter: parent.horizontalCenter
                        width: Math.min(520, parent.width)
                        implicitHeight: emptyCol.implicitHeight + 56
                        color: Theme.bgPanel
                        border.color: Theme.borderColor
                        border.width: 1
                        radius: Theme.radiusPanel

                        ColumnLayout {
                            id: emptyCol
                            anchors.centerIn: parent
                            width: parent.width - 56
                            spacing: 12

                            Text {
                                Layout.alignment: Qt.AlignHCenter
                                text: "◎"
                                color: Theme.textMuted
                                font.family: Theme.fontUi
                                font.pixelSize: 34
                            }
                            Text {
                                Layout.alignment: Qt.AlignHCenter
                                text: "No segment selected"
                                color: Theme.textPrimary
                                font.family: Theme.fontUi
                                font.pixelSize: 16
                                font.weight: Font.DemiBold
                            }
                            Text {
                                Layout.fillWidth: true
                                horizontalAlignment: Text.AlignHCenter
                                text: "Open the Segmentation view and click a row (or click a block in the workspace) to inspect it."
                                color: Theme.textSecondary
                                font.family: Theme.fontUi
                                font.pixelSize: 13
                                lineHeight: 1.45
                                wrapMode: Text.WordWrap
                            }
                        }
                    }
                }

                // -- detail block (only rendered when a segment is selected) ----
                ColumnLayout {
                    id: detailBlock
                    visible: root.seg !== null && root.seg !== undefined
                    Layout.fillWidth: true
                    spacing: 18

                // -- title -----------------------------------------------------
                RowLayout {
                    Layout.fillWidth: true
                    Layout.bottomMargin: 0
                    spacing: 12

                    Text {
                        // The viewmodel exposes no run-length segment index, so the
                        // selected segment's quality tier stands in for the "#N".
                        text: root.seg ? ("Segment · " + root.tierCode) : ""
                        color: Theme.textPrimary
                        font.family: Theme.fontUi
                        font.pixelSize: 19
                        font.weight: Font.DemiBold
                    }
                    Text {
                        text: root.seg
                              ? (root.fmtTime(root.seg.startSec) + " – " + root.fmtTime(root.seg.endSec)
                                 + " · " + (recordings.currentModality || "").toUpperCase()
                                 + " · " + Math.max(0, root.seg.endSec - root.seg.startSec).toFixed(2) + " s window")
                              : ""
                        color: Theme.textMuted
                        font.family: Theme.fontMono
                        font.pixelSize: 12
                        Layout.alignment: Qt.AlignBaseline
                    }
                    Item { Layout.fillWidth: true }
                    // XAI trigger — in line with the header; its results (heatmap + attribution + narrative)
                    // render below the waveform.
                    OutlineButton {
                        Layout.alignment: Qt.AlignVCenter
                        visible: root.segCurve !== null
                        text: guard.saliencyPending ? "Analyzing…"
                              : ((guard.saliencyMap && guard.saliencyMap.length > 0)
                                 ? "↻ Re-explain" : "🔍  Explain this grade")
                        enabled: !guard.saliencyPending && root.seg !== null && root.seg !== undefined
                        onClicked: guard.requestSaliency(root.seg.startSec, root.seg.endSec,
                                                         root.seg.tier, root.tierInfo.label, root.seg.artifacts)
                    }
                }

                // -- zoomed waveform card -------------------------------------
                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 260
                    color: Theme.bgRail
                    border.color: Theme.borderColor
                    border.width: 1
                    radius: Theme.radiusPanel
                    clip: true

                    // plot area (leaves room for the footer strip)
                    Item {
                        id: plotArea
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.top: parent.top
                        anchors.margins: 6
                        height: parent.height - 34

                        // tier-tinted band fill behind the trace
                        Rectangle {
                            anchors.fill: parent
                            radius: 8
                            color: Qt.rgba(root.tierColor.r, root.tierColor.g,
                                           root.tierColor.b, 0.10)
                        }
                        // zero baseline
                        Rectangle {
                            anchors.left: parent.left
                            anchors.right: parent.right
                            anchors.verticalCenter: parent.verticalCenter
                            height: 1
                            color: Qt.rgba(1, 1, 1, 0.06)
                        }

                        // Real per-segment waveform: the decimated min/max envelope of
                        // the selected segment's actual samples (signalView.curveForRange).
                        Canvas {
                            id: tracePreview
                            anchors.fill: parent
                            anchors.margins: 8
                            onWidthChanged: requestPaint()
                            onHeightChanged: requestPaint()
                            // repaint when the occlusion-saliency map arrives
                            Connections {
                                target: guard
                                function onSaliencyChanged() { tracePreview.requestPaint() }
                            }
                            onPaint: {
                                var ctx = getContext("2d")
                                ctx.reset()
                                // -- occlusion-saliency heatmap band (XAI: where the model looks) --
                                var sal = guard.saliencyMap
                                if (sal && sal.length > 1) {
                                    for (var j = 0; j < sal.length; ++j) {
                                        var v = sal[j]
                                        if (v <= 0.06) continue
                                        // amber→red intensity ∝ importance, painted full-height behind the trace
                                        ctx.fillStyle = Qt.rgba(0.90, 0.40 - 0.22 * v, 0.16, 0.10 + 0.5 * v)
                                        var xa = (j / sal.length) * width
                                        var xb = ((j + 1) / sal.length) * width
                                        ctx.fillRect(xa, 0, (xb - xa) + 1, height)
                                    }
                                }
                                var c = root.segCurve
                                if (!c || !c.x || c.x.length < 2)
                                    return
                                ctx.strokeStyle = Theme.traceColor
                                ctx.lineWidth = 1.3
                                var n = c.x.length
                                var x0 = c.x[0]
                                var tspan = (c.x[n - 1] - x0) || 1
                                var vspan = (c.hi - c.lo) || 1
                                ctx.beginPath()
                                for (var i = 0; i < n; ++i) {
                                    var px = ((c.x[i] - x0) / tspan) * width
                                    var pmax = height - ((c.ymax[i] - c.lo) / vspan) * height
                                    var pmin = height - ((c.ymin[i] - c.lo) / vspan) * height
                                    if (i === 0) ctx.moveTo(px, pmax)
                                    else ctx.lineTo(px, pmax)
                                    ctx.lineTo(px, pmin)
                                }
                                ctx.stroke()
                            }
                        }
                    }

                    // floating tier badge (top-left)
                    RowLayout {
                        anchors.top: parent.top
                        anchors.left: parent.left
                        anchors.topMargin: 14
                        anchors.leftMargin: 16
                        z: 2
                        spacing: 8

                        TierChip { tier: root.tierCode; size: 26; filled: true }
                        Text {
                            text: root.tierCode + " " + root.tierInfo.label
                            color: root.tierColor
                            font.family: Theme.fontMono
                            font.pixelSize: 13
                            font.bold: true
                        }
                    }

                    // footer: start / amplitude range / end
                    RowLayout {
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.bottom: parent.bottom
                        anchors.leftMargin: 14
                        anchors.rightMargin: 14
                        anchors.bottomMargin: 8
                        Text {
                            text: root.seg ? root.fmtTime(root.seg.startSec) : "--:--:--"
                            color: Theme.textMuted
                            font.family: Theme.fontMono
                            font.pixelSize: 10
                        }
                        Item { Layout.fillWidth: true }
                        Text {
                            text: root.segCurve
                                  ? (root.segCurve.lo.toFixed(2) + " … " + root.segCurve.hi.toFixed(2) + " a.u.")
                                  : "—"
                            color: Theme.textMuted
                            font.family: Theme.fontMono
                            font.pixelSize: 10
                        }
                        Item { Layout.fillWidth: true }
                        Text {
                            text: root.seg ? root.fmtTime(root.seg.endSec) : "--:--:--"
                            color: Theme.textMuted
                            font.family: Theme.fontMono
                            font.pixelSize: 10
                        }
                    }
                }

                // indicative-preview caption: the zoomed trace above is a
                // synthetic placeholder, not this segment's real samples.
                Text {
                    Layout.fillWidth: true
                    visible: root.segCurve === null
                    text: "Waveform preview unavailable for this segment."
                    color: Theme.textMuted
                    font.family: Theme.fontUi
                    font.pixelSize: 11
                    font.italic: true
                    wrapMode: Text.WordWrap
                }

                // -- explainability: occlusion-saliency caveat (the button lives in the header row above) --
                Text {
                    Layout.fillWidth: true
                    visible: guard.saliencyMap && guard.saliencyMap.length > 0
                    text: "Amber = the time regions that most change the grade when removed (occlusion "
                          + "saliency — a perturbation-based estimate of what the model attends to, not ground truth)."
                    color: Theme.textMuted
                    font.family: Theme.fontUi; font.pixelSize: 10
                    wrapMode: Text.WordWrap
                }

                // -- explainability: unified plain-language summary (where + why + what) --
                Rectangle {
                    visible: guard.gradeNarrative && guard.gradeNarrative.length > 0
                    Layout.fillWidth: true
                    Layout.topMargin: 2
                    implicitHeight: narrativeText.implicitHeight + 16
                    radius: 6
                    color: Qt.rgba(Theme.accentDim.r, Theme.accentDim.g, Theme.accentDim.b, 0.10)
                    border.color: Qt.rgba(Theme.accentDim.r, Theme.accentDim.g, Theme.accentDim.b, 0.28)
                    border.width: 1
                    Text {
                        id: narrativeText
                        anchors.fill: parent
                        anchors.margins: 8
                        text: "🧭  " + guard.gradeNarrative
                        color: Theme.textPrimary
                        font.family: Theme.fontUi; font.pixelSize: 11
                        wrapMode: Text.WordWrap
                        verticalAlignment: Text.AlignVCenter
                    }
                }

                // -- explainability: feature attribution ("WHY this grade — which quality property?") --
                ColumnLayout {
                    visible: guard.gradeAttribution && guard.gradeAttribution.length > 0
                    Layout.fillWidth: true
                    Layout.topMargin: 2
                    spacing: 3
                    Text {
                        text: "Why this grade — quality properties driving it"
                        color: Theme.textPrimary
                        font.family: Theme.fontUi; font.pixelSize: 11; font.bold: true
                    }
                    Repeater {
                        model: guard.gradeAttribution
                        delegate: RowLayout {
                            Layout.fillWidth: true
                            spacing: 8
                            Text {
                                text: modelData.group
                                color: Theme.textPrimary
                                font.family: Theme.fontUi; font.pixelSize: 10
                                Layout.preferredWidth: 132
                                elide: Text.ElideRight
                            }
                            Rectangle {                              // bar: length ∝ share, color = direction
                                Layout.fillWidth: true
                                Layout.preferredHeight: 9
                                radius: 2
                                color: Qt.rgba(Theme.textMuted.r, Theme.textMuted.g, Theme.textMuted.b, 0.12)
                                Rectangle {
                                    // bar length ∝ ABSOLUTE φ magnitude (not relative share), so a clean
                                    // window whose groups barely move the grade renders faint, not full-width
                                    width: parent.width * Math.max(0.015, modelData.scaled)
                                    height: parent.height; radius: 2
                                    // φ>0 pushes toward UNUSABLE (Q0 color), φ<0 toward USABLE (Q3 color)
                                    color: (modelData.phi >= 0 ? Theme.currentQualityPalette()["Q0"].color
                                                               : Theme.currentQualityPalette()["Q3"].color)
                                }
                            }
                            Text {
                                text: Math.round(modelData.share * 100) + "%"
                                color: Theme.textMuted
                                font.family: Theme.fontMono; font.pixelSize: 10
                                Layout.preferredWidth: 30
                                horizontalAlignment: Text.AlignRight
                            }
                        }
                    }
                    Text {
                        Layout.fillWidth: true
                        text: "▸ Group-Shapley over the model's fused quality features. Red pushes toward unusable, "
                              + "green toward usable. A perturbation estimate of the model's reliance, not ground truth; "
                              + "silent about what the grade reads from the raw waveform directly."
                        color: Theme.textMuted
                        font.family: Theme.fontUi; font.pixelSize: 9
                        wrapMode: Text.WordWrap
                    }
                }

                // -- reshape segment (kept directly under the signal preview) --
                GlassPanel {
                    visible: root.seg !== null && root.seg !== undefined
                    Layout.fillWidth: true
                    Layout.topMargin: 8
                    SegmentReshape { Layout.fillWidth: true }
                }

                // -- two-column body ------------------------------------------
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 18

                    // ---- reasoning column (stretch 1.4) ----
                    ColumnLayout {
                        Layout.fillWidth: true
                        Layout.preferredWidth: doc.width * 0.56
                        Layout.alignment: Qt.AlignTop
                        spacing: 16

                        // AI rationale
                        GlassPanel {
                            Layout.fillWidth: true
                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 14

                                ConfidenceGauge {
                                    Layout.preferredWidth: 88
                                    Layout.preferredHeight: 88
                                    confidence: root.seg ? root.seg.confidence : 0
                                    arcColor: root.tierColor
                                }
                                ColumnLayout {
                                    Layout.fillWidth: true
                                    spacing: 8
                                    Text {
                                        text: "AI RATIONALE"
                                        color: Theme.textMuted
                                        font.family: Theme.fontUi
                                        font.pixelSize: 11
                                        font.weight: Font.DemiBold
                                        font.letterSpacing: 1
                                    }
                                    Text {
                                        Layout.fillWidth: true
                                        text: (root.seg && root.seg.rationale.length > 0)
                                              ? root.seg.rationale
                                              : "No rationale available for this segment."
                                        color: "#c5cdda"
                                        font.family: Theme.fontUi
                                        font.pixelSize: 13
                                        lineHeight: 1.5
                                        wrapMode: Text.WordWrap
                                    }
                                }
                            }
                        }

                        // detected artifacts
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 6
                            Text {
                                text: "DETECTED ARTIFACTS"
                                color: Theme.textMuted
                                font.family: Theme.fontUi
                                font.pixelSize: 11
                                font.weight: Font.DemiBold
                                font.letterSpacing: 1
                            }
                            Text {
                                text: "· flagged across this segment"
                                color: Theme.chipBorderMuted
                                font.family: Theme.fontUi
                                font.pixelSize: 11
                            }
                        }

                        Text {
                            visible: !root.seg || root.seg.artifacts.length === 0
                            text: "No artifacts flagged for this segment."
                            color: Theme.textMuted
                            font.family: Theme.fontUi
                            font.pixelSize: 12
                        }

                        Repeater {
                            model: root.seg ? root.seg.artifacts : []
                            delegate: Rectangle {
                                id: artCard
                                required property int index
                                required property string modelData
                                Layout.fillWidth: true
                                implicitHeight: artCol.implicitHeight + 24
                                color: Theme.bgPanel
                                border.color: Theme.borderColor
                                border.width: 1
                                radius: Theme.radiusControl

                                ColumnLayout {
                                    id: artCol
                                    anchors.left: parent.left
                                    anchors.right: parent.right
                                    anchors.verticalCenter: parent.verticalCenter
                                    anchors.leftMargin: 14
                                    anchors.rightMargin: 14
                                    spacing: 6

                                    RowLayout {
                                        Layout.fillWidth: true
                                        spacing: 9
                                        Rectangle {
                                            Layout.preferredWidth: 8
                                            Layout.preferredHeight: 8
                                            radius: 4
                                            color: root.tierColor
                                        }
                                        Text {
                                            text: artCard.modelData
                                            color: Theme.textPrimary
                                            font.family: Theme.fontMono
                                            font.pixelSize: 13
                                            font.weight: Font.DemiBold
                                        }
                                        Item { Layout.fillWidth: true }
                                    }
                                    Text {
                                        Layout.fillWidth: true
                                        text: root.artifactDesc(artCard.modelData)
                                        color: Theme.textSecondary
                                        font.family: Theme.fontUi
                                        font.pixelSize: 11
                                        wrapMode: Text.WordWrap
                                    }
                                }
                            }
                        }

                        // interpretable classical-SQI breakdown (Raw/Filtered toggle + model-vs-SQI banner)
                        GlassPanel {
                            Layout.fillWidth: true
                            visible: guard.sqiBreakdown && guard.sqiBreakdown.length > 0
                            SqiBreakdownPanel { Layout.fillWidth: true; tier: root.tierCode }
                        }

                        // per-modality task usability (EEG per-band, EDA tonic/phasic)
                        GlassPanel {
                            Layout.fillWidth: true
                            visible: guard.usabilityVerdicts && guard.usabilityVerdicts.length > 0
                            UsabilityPanel { Layout.fillWidth: true }
                        }
                    }

                    // ---- override column (stretch 1.0) ----
                    ColumnLayout {
                        Layout.fillWidth: true
                        Layout.preferredWidth: doc.width * 0.42
                        Layout.alignment: Qt.AlignTop
                        spacing: 0

                        GlassPanel {
                            Layout.fillWidth: true
                            title: "Human-in-the-loop"
                            subtitle: "The researcher decides. Corrections flow to the training queue."

                            // verdict trail
                            Rectangle {
                                Layout.fillWidth: true
                                Layout.topMargin: 6
                                implicitHeight: 58
                                color: Theme.bgPanelAlt
                                border.color: Theme.borderColor
                                border.width: 1
                                radius: Theme.radiusControl

                                RowLayout {
                                    anchors.fill: parent
                                    anchors.margins: 12
                                    spacing: 10
                                    ColumnLayout {
                                        Layout.fillWidth: true
                                        spacing: 5
                                        Text {
                                            text: "MODEL SAID"
                                            Layout.alignment: Qt.AlignHCenter
                                            color: Theme.textMuted
                                            font.family: Theme.fontUi
                                            font.pixelSize: 10
                                        }
                                        Text {
                                            text: root.tierCode
                                            Layout.alignment: Qt.AlignHCenter
                                            color: root.tierColor
                                            font.family: Theme.fontMono
                                            font.pixelSize: 14
                                            font.bold: true
                                        }
                                    }
                                    Text {
                                        text: "→"
                                        color: Theme.textMuted
                                        font.family: Theme.fontUi
                                        font.pixelSize: 16
                                    }
                                    ColumnLayout {
                                        Layout.fillWidth: true
                                        spacing: 5
                                        Text {
                                            text: "YOU SET"
                                            Layout.alignment: Qt.AlignHCenter
                                            color: Theme.textMuted
                                            font.family: Theme.fontUi
                                            font.pixelSize: 10
                                        }
                                        Text {
                                            text: root.effTier
                                            Layout.alignment: Qt.AlignHCenter
                                            color: root.effInfo.color
                                            font.family: Theme.fontMono
                                            font.pixelSize: 14
                                            font.bold: true
                                        }
                                    }
                                }
                            }

                            Text {
                                text: "Set quality tier"
                                Layout.topMargin: 4
                                color: Theme.textMuted
                                font.family: Theme.fontUi
                                font.pixelSize: 11
                            }

                            // tier-override rows
                            Repeater {
                                model: ["Q3", "Q2", "Q1", "Q0"]
                                delegate: Rectangle {
                                    required property string modelData
                                    readonly property var p: root.paletteFor(modelData)
                                    readonly property bool active: root.userTier === modelData
                                    Layout.fillWidth: true
                                    implicitHeight: 40
                                    radius: Theme.radiusControl
                                    color: active
                                           ? Qt.rgba(p.color.r, p.color.g, p.color.b, 0.10)
                                           : Theme.bgPanelAlt
                                    border.width: 1
                                    border.color: active ? p.color
                                                 : (rowHover.hovered ? "#33415a" : Theme.borderColor)

                                    HoverHandler { id: rowHover }
                                    TapHandler {
                                        onTapped: {
                                            root.userTier = modelData
                                            selection.relabel(modelData)
                                            AppController.toast("Relabeled to " + modelData)
                                        }
                                    }

                                    RowLayout {
                                        anchors.fill: parent
                                        anchors.leftMargin: 12
                                        anchors.rightMargin: 12
                                        spacing: 10
                                        TierChip { tier: modelData; size: 20 }
                                        Text {
                                            text: modelData
                                            color: p.color
                                            font.family: Theme.fontMono
                                            font.pixelSize: 12
                                            font.weight: Font.DemiBold
                                            Layout.preferredWidth: 24
                                        }
                                        Text {
                                            text: p.label
                                            color: Theme.textPrimary
                                            font.family: Theme.fontUi
                                            font.pixelSize: 12
                                            Layout.fillWidth: true
                                        }
                                        Text {
                                            text: root.shortDesc(modelData)
                                            color: Theme.textSecondary
                                            font.family: Theme.fontUi
                                            font.pixelSize: 11
                                        }
                                    }
                                }
                            }

                            // Free-text reviewer note (persisted with the correction).
                            TextField {
                                id: noteField
                                Layout.fillWidth: true
                                Layout.topMargin: 4
                                placeholderText: "Add a review note (optional)…"
                                text: (selection.selectedSegment && selection.selectedSegment.note)
                                      ? selection.selectedSegment.note : ""
                                color: Theme.textPrimary
                                font.family: Theme.fontUi; font.pixelSize: 12
                                selectByMouse: true
                                onEditingFinished: selection.addNote(text)
                                background: Rectangle {
                                    radius: Theme.radiusControl; color: Theme.bgPanelAlt
                                    border.color: parent.activeFocus ? Theme.accent : Theme.borderColor
                                }
                            }

                            AccentButton {
                                Layout.fillWidth: true
                                text: "Save correction → training queue"
                                glyph: "↓"
                                onClicked: {
                                    selection.addNote(noteField.text)
                                    var p = selection.saveToTrainingQueue()
                                    AppController.toast(p ? ("Saved → " + p.split(/[\\\/]/).pop())
                                                          : "No segment selected")
                                }
                            }

                            OutlineButton {
                                Layout.fillWidth: true
                                text: (selection.selectedSegment && selection.selectedSegment.flagged)
                                      ? "⚑ Flagged for review" : "Add note & flag for review"
                                onClicked: {
                                    var t = (noteField.text && noteField.text.length > 0)
                                            ? noteField.text : "Flagged for review"
                                    var p = selection.flagForReview(t)
                                    AppController.toast(p ? ("⚑ Flagged → " + p.split(/[\\\/]/).pop())
                                                          : "No segment selected")
                                }
                            }
                        }
                    }
                }
                } // detailBlock
            }
        }
    }
}
