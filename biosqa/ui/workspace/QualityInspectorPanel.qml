import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../"
import "../components"

// design spec (a): "QualityInspectorPanel (right dock, 322px) 'AI Quality
// Inspector'". Composes the verdict card + ConfidenceGauge + ArtifactChips +
// RationaleBox + HumanInLoopControls, all bound to the single
// `selection.selectedSegment` source of truth.
Rectangle {
    id: root
    color: Theme.bgPanel
    border.color: Theme.borderColor
    border.width: 1

    readonly property var segment: selection.selectedSegment

    // Per-tier verdict description (not exposed by the viewmodel; matches the
    // mockup's TIER().desc copy).
    readonly property var _tierDesc: ({
        "Q3": "Usable for all analysis",
        "Q2": "Usable for rate / coarse features",
        "Q1": "Use with caution",
        "Q0": "Discard — signal unrecoverable"
    })

    function _fmt(sec) {
        sec = Math.max(0, Math.round(sec))
        var m = Math.floor(sec / 60), s = sec % 60
        return (m < 10 ? "0" : "") + m + ":" + (s < 10 ? "0" : "") + s
    }

    // Compute the interpretable SQI breakdown for the newly-selected segment (cheap, on demand).
    Connections {
        target: selection
        function onSelectedSegmentChanged() {
            if (selection.selectedSegment)
                guard.requestSqi(selection.selectedSegment.startSec, selection.selectedSegment.endSec)
        }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        // -- header ----------------------------------------------------------
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 46
            color: "transparent"
            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: 16
                anchors.rightMargin: 16
                spacing: 8
                Text {
                    text: "⛨"  // shield glyph
                    color: Theme.accent
                    font.pixelSize: 15
                }
                Label {
                    text: "AI Quality Inspector"
                    color: Theme.textPrimary
                    font.family: Theme.fontUi
                    font.pixelSize: 12
                    font.weight: Font.DemiBold
                }
                Item { Layout.fillWidth: true }
                Label {
                    visible: root.segment !== null && root.segment !== undefined
                    text: root.segment ? root.segment.tier : ""
                    color: Theme.textMuted
                    font.family: Theme.fontMono
                    font.pixelSize: 10
                }
            }
            Rectangle {
                anchors.bottom: parent.bottom
                width: parent.width
                height: 1
                color: Theme.borderColor
            }
        }

        // -- body ------------------------------------------------------------
        Flickable {
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            contentWidth: width
            contentHeight: body.implicitHeight

            ColumnLayout {
                id: body
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.top: parent.top
                anchors.leftMargin: 16
                anchors.rightMargin: 16
                anchors.topMargin: 14
                spacing: 14

                // -- false-clean / input warning banner (record-level) -----
                Rectangle {
                    visible: guard.hasWarning
                    Layout.fillWidth: true
                    color: Qt.rgba(0.878, 0.639, 0.18, 0.12)   // amber tint
                    border.color: Theme.warnColor
                    border.width: 1
                    radius: 9
                    implicitHeight: bannerCol.implicitHeight + 20
                    ColumnLayout {
                        id: bannerCol
                        anchors.left: parent.left; anchors.right: parent.right
                        anchors.top: parent.top
                        anchors.margins: 10
                        spacing: 4
                        RowLayout {
                            spacing: 6
                            Text { text: "⚠"; color: Theme.warnColor; font.pixelSize: 13 }
                            Text {
                                text: "Input warning"
                                color: Theme.textPrimary
                                font.family: Theme.fontUi; font.pixelSize: 11; font.weight: Font.DemiBold
                            }
                        }
                        Text {
                            text: guard.bannerText
                            color: Theme.textSecondary
                            font.family: Theme.fontUi; font.pixelSize: 11
                            wrapMode: Text.WordWrap
                            Layout.fillWidth: true
                        }
                        // per-window reasons the integrity guard used to override the clean score (explains
                        // WHY it re-flagged — the detail behind the summary line above)
                        Repeater {
                            model: guard.guardReasons
                            delegate: Text {
                                required property var modelData
                                text: "• " + modelData
                                color: Theme.textMuted
                                font.family: Theme.fontUi; font.pixelSize: 10
                                wrapMode: Text.WordWrap
                                Layout.fillWidth: true
                                Layout.leftMargin: 4
                            }
                        }
                    }
                }

                // -- record data-quality strip ------------------------------
                Rectangle {
                    visible: guard.dataQualityFlags.length > 0 || guard.completeness < 0.999
                    Layout.fillWidth: true
                    color: Theme.bgPanelAlt
                    border.color: Theme.borderColor
                    border.width: 1
                    radius: 9
                    implicitHeight: dqCol.implicitHeight + 20
                    ColumnLayout {
                        id: dqCol
                        anchors.left: parent.left; anchors.right: parent.right
                        anchors.top: parent.top
                        anchors.margins: 10
                        spacing: 3
                        Text {
                            text: "DATA QUALITY"
                            color: Theme.textMuted
                            font.family: Theme.fontUi; font.pixelSize: 10
                            font.weight: Font.DemiBold; font.letterSpacing: 0.7
                        }
                        Text {
                            text: "Completeness " + (guard.completeness * 100).toFixed(0) + "%"
                                  + (guard.dataQualityUsable ? "" : " · not usable")
                            color: guard.dataQualityUsable ? Theme.textSecondary : Theme.dangerColor
                            font.family: Theme.fontMono; font.pixelSize: 11
                        }
                        Repeater {
                            model: guard.dataQualityFlags
                            Text {
                                text: "• " + modelData
                                color: Theme.textMuted
                                font.family: Theme.fontUi; font.pixelSize: 10
                                wrapMode: Text.WordWrap
                                Layout.fillWidth: true
                            }
                        }
                    }
                }

                // -- acquisition-regime / domain-shift banner (research3 cross-dataset weakness) --
                Rectangle {
                    visible: guard.domainShiftIndex > 0.3 || (guard.regimeFlags && guard.regimeFlags.length > 0)
                    Layout.fillWidth: true
                    color: Qt.rgba(0.878, 0.639, 0.18, 0.10)
                    border.color: Theme.warnColor; border.width: 1
                    radius: 9
                    implicitHeight: regCol.implicitHeight + 20
                    ColumnLayout {
                        id: regCol
                        anchors.left: parent.left; anchors.right: parent.right
                        anchors.top: parent.top; anchors.margins: 10
                        spacing: 3
                        RowLayout {
                            spacing: 6
                            Text { text: "⚠"; color: Theme.warnColor; font.pixelSize: 13 }
                            Text {
                                text: "Out-of-regime input"
                                color: Theme.textPrimary
                                font.family: Theme.fontUi; font.pixelSize: 11; font.weight: Font.DemiBold
                            }
                            Item { Layout.fillWidth: true }
                            Text {
                                // novelty = fraction of windows whose SQI signature is unlike anything in
                                // training (Mahalanobis) — shown alongside the spectral domain-shift index
                                visible: guard.noveltyFraction > 0.0
                                text: "novelty " + Math.round(guard.noveltyFraction * 100) + "%"
                                color: Theme.textMuted; font.family: Theme.fontMono; font.pixelSize: 10
                            }
                            Text {
                                text: "shift " + Math.round(guard.domainShiftIndex * 100) + "%"
                                color: Theme.warnColor; font.family: Theme.fontMono; font.pixelSize: 10
                            }
                        }
                        Repeater {
                            model: guard.regimeFlags
                            Text {
                                text: "• " + modelData
                                color: Theme.textSecondary
                                font.family: Theme.fontUi; font.pixelSize: 10
                                wrapMode: Text.WordWrap; Layout.fillWidth: true
                            }
                        }
                        Text {
                            text: "The recording's spectrum differs from the model's training regime — trust the "
                                  + "scores less; feed a raw recording at the model's native rate for the most reliable read."
                            color: Theme.textMuted
                            font.family: Theme.fontUi; font.pixelSize: 9
                            wrapMode: Text.WordWrap; Layout.fillWidth: true
                        }
                    }
                }

                // -- SEGMENTS IN VIEW (viewport-scoped, selectable cards) -----
                // Reflects the segments overlapping the plot's current visible range,
                // refreshed as you pan/zoom; click a card to inspect that segment below.
                property var _cards: []
                function _refreshCards() {
                    body._cards = (signalView.durationSec > 0)
                        ? segments.segmentsInRange(signalView.viewStartSec, signalView.viewEndSec) : []
                }
                Connections {
                    target: signalView
                    function onViewStartSecChanged() { body._refreshCards() }
                    function onViewEndSecChanged() { body._refreshCards() }
                    function onDurationSecChanged() { body._refreshCards() }
                }
                Connections {
                    target: segments
                    function onStatsChanged() { body._refreshCards() }
                    function onFilterChanged() { body._refreshCards() }
                }
                Component.onCompleted: body._refreshCards()

                ColumnLayout {
                    visible: body._cards.length > 0
                    Layout.fillWidth: true
                    spacing: 6
                    Text {
                        text: "SEGMENTS IN VIEW · " + body._cards.length
                        color: Theme.textMuted
                        font.family: Theme.fontUi; font.pixelSize: 10
                        font.weight: Font.DemiBold; font.letterSpacing: 0.7
                    }
                    Repeater {
                        model: body._cards
                        delegate: Rectangle {
                            id: card
                            required property var modelData
                            readonly property bool sel: selection.selectedAllIndex === modelData.index
                            Layout.fillWidth: true
                            radius: 8
                            color: card.sel ? Theme.hoverBg : Theme.bgPanelAlt
                            border.width: 1
                            border.color: card.sel ? Theme.accent : Theme.borderColor
                            implicitHeight: cardCol.implicitHeight + 16
                            ColumnLayout {
                                id: cardCol
                                anchors.left: parent.left; anchors.right: parent.right
                                anchors.top: parent.top; anchors.margins: 8
                                spacing: 4
                                RowLayout {
                                    Layout.fillWidth: true
                                    spacing: 8
                                    TierChip { tier: card.modelData.tier; size: 16 }
                                    Text {
                                        text: root._fmt(card.modelData.startSec) + " – " + root._fmt(card.modelData.endSec)
                                        color: Theme.textPrimary
                                        font.family: Theme.fontMono; font.pixelSize: 11
                                    }
                                    Text {
                                        visible: card.modelData.recoverable === true
                                        text: "↺"
                                        color: Theme.accent
                                        font.pixelSize: 12; font.bold: true
                                        ToolTip.visible: recMa.containsMouse
                                        ToolTip.text: "Recoverable → " + card.modelData.recoveredTier + " with a standard filter"
                                        MouseArea { id: recMa; anchors.fill: parent; hoverEnabled: true }
                                    }
                                    Item { Layout.fillWidth: true }
                                    Text {
                                        text: (card.modelData.confidence * 100).toFixed(0) + "%"
                                        color: Theme.textMuted
                                        font.family: Theme.fontMono; font.pixelSize: 11
                                    }
                                }
                                Flow {
                                    Layout.fillWidth: true
                                    visible: card.modelData.artifacts.length > 0
                                    spacing: 4
                                    Repeater {
                                        model: card.modelData.artifacts
                                        delegate: Rectangle {
                                            required property string modelData
                                            radius: 4; color: Theme.bgPanel
                                            border.width: 1; border.color: Theme.chipBorderMuted
                                            implicitWidth: atxt.implicitWidth + 10; implicitHeight: 16
                                            Text {
                                                id: atxt; anchors.centerIn: parent; text: parent.modelData
                                                color: Theme.textMuted; font.family: Theme.fontUi; font.pixelSize: 9
                                            }
                                        }
                                    }
                                }
                            }
                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.PointingHandCursor
                                onClicked: selection.selectByAllIndex(card.modelData.index)
                            }
                        }
                    }
                }

                Label {
                    visible: (root.segment === null || root.segment === undefined) && body._cards.length === 0
                    text: "Open a recording, then click a segment card above (or a block in the\nplot / table) to inspect its predicted quality."
                    color: Theme.textMuted
                    font.family: Theme.fontUi
                    font.pixelSize: 12
                    wrapMode: Text.WordWrap
                    Layout.fillWidth: true
                }

                // Selected-window caption
                RowLayout {
                    visible: root.segment !== null && root.segment !== undefined
                    Layout.fillWidth: true
                    spacing: 4
                    Text {
                        text: "Selected window ·"
                        color: Theme.textMuted
                        font.family: Theme.fontUi
                        font.pixelSize: 10
                    }
                    Text {
                        text: root.segment
                            ? root.segment.startSec.toFixed(1) + "s – " + root.segment.endSec.toFixed(1) + "s"
                            : ""
                        color: Theme.textMuted
                        font.family: Theme.fontMono
                        font.pixelSize: 10
                    }
                }

                // -- verdict card ------------------------------------------
                Rectangle {
                    id: verdictCard
                    visible: root.segment !== null && root.segment !== undefined
                    Layout.fillWidth: true
                    readonly property color tierColor: root.segment ? root.segment.color : Theme.accent
                    readonly property string tierKey: root.segment ? root.segment.tier : "Q3"
                    color: Qt.rgba(tierColor.r, tierColor.g, tierColor.b, 0.10)
                    border.color: tierColor
                    border.width: 1
                    radius: 9
                    implicitHeight: verdictRow.implicitHeight + 24

                    RowLayout {
                        id: verdictRow
                        anchors.fill: parent
                        anchors.margins: 12
                        spacing: 12

                        TierChip {
                            tier: verdictCard.tierKey
                            size: 38
                            filled: true
                        }
                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 2
                            RowLayout {
                                spacing: 7
                                Text {
                                    text: verdictCard.tierKey
                                    color: verdictCard.tierColor
                                    font.family: Theme.fontMono
                                    font.pixelSize: 15
                                    font.bold: true
                                }
                                Text {
                                    text: root.segment ? Theme.tierInfo(root.segment.tier).label : ""
                                    color: Theme.textPrimary
                                    font.family: Theme.fontUi
                                    font.pixelSize: 13
                                    font.weight: Font.DemiBold
                                }
                            }
                            Text {
                                text: root.segment ? (root._tierDesc[root.segment.tier] || "") : ""
                                color: Theme.textSecondary
                                font.family: Theme.fontUi
                                font.pixelSize: 11
                                wrapMode: Text.WordWrap
                                Layout.fillWidth: true
                            }
                        }
                    }
                }

                // -- recoverability advisory (filtered-vs-raw second pass) --
                Rectangle {
                    visible: root.segment && root.segment.recoverable
                    Layout.fillWidth: true
                    color: Qt.rgba(Theme.accent.r, Theme.accent.g, Theme.accent.b, 0.10)
                    border.color: Theme.accent
                    border.width: 1
                    radius: 9
                    implicitHeight: recCol.implicitHeight + 20
                    ColumnLayout {
                        id: recCol
                        anchors.left: parent.left; anchors.right: parent.right
                        anchors.top: parent.top; anchors.margins: 10
                        spacing: 3
                        RowLayout {
                            spacing: 7
                            Text { text: "↺"; color: Theme.accent; font.pixelSize: 14; font.bold: true }
                            Text {
                                text: "Recoverable"
                                color: Theme.tealText
                                font.family: Theme.fontUi; font.pixelSize: 12; font.weight: Font.DemiBold
                            }
                            Item { Layout.fillWidth: true }
                            Text {
                                text: root.segment ? (root.segment.tier + " → " + root.segment.recoveredTier) : ""
                                color: Theme.textSecondary
                                font.family: Theme.fontMono; font.pixelSize: 11
                            }
                        }
                        Text {
                            text: "A standard filter would likely lift this window to "
                                  + (root.segment ? root.segment.recoveredTier : "") +
                                  ". Advisory only — verify; over-filtering can mask true corruption."
                            color: Theme.textMuted
                            font.family: Theme.fontUi; font.pixelSize: 10
                            wrapMode: Text.WordWrap
                            Layout.fillWidth: true
                        }
                    }
                }

                // -- task-relative rate-usability (poor morphology, rate still OK) --
                Rectangle {
                    visible: root.segment && root.segment.rateUsable
                    Layout.fillWidth: true
                    color: Qt.rgba(0.18, 0.82, 0.71, 0.10)
                    border.color: Theme.tealText
                    border.width: 1
                    radius: 9
                    implicitHeight: rateCol.implicitHeight + 20
                    ColumnLayout {
                        id: rateCol
                        anchors.left: parent.left; anchors.right: parent.right
                        anchors.top: parent.top; anchors.margins: 10
                        spacing: 3
                        RowLayout {
                            spacing: 7
                            Text { text: "♥"; color: Theme.tealText; font.pixelSize: 14 }
                            Text {
                                text: "Rate-usable"
                                color: Theme.tealText
                                font.family: Theme.fontUi; font.pixelSize: 12; font.weight: Font.DemiBold
                            }
                            Item { Layout.fillWidth: true }
                            Text {
                                text: root.segment && root.segment.hrBpm > 0
                                      ? (Math.round(root.segment.hrBpm) + " bpm") : ""
                                color: Theme.textSecondary
                                font.family: Theme.fontMono; font.pixelSize: 11
                            }
                        }
                        Text {
                            text: "Poor morphology, but beats are reliably detected — heart/pulse rate here is "
                                  + "trustworthy even though the waveform isn't. (Distinct from filtering.)"
                            color: Theme.textMuted
                            font.family: Theme.fontUi; font.pixelSize: 10
                            wrapMode: Text.WordWrap
                            Layout.fillWidth: true
                        }
                    }
                }

                // -- confidence gauge card ---------------------------------
                Rectangle {
                    visible: root.segment !== null && root.segment !== undefined
                    Layout.fillWidth: true
                    color: Theme.bgPanelAlt
                    border.color: Theme.borderColor
                    border.width: 1
                    radius: 9
                    implicitHeight: gaugeRow.implicitHeight + 24

                    RowLayout {
                        id: gaugeRow
                        anchors.fill: parent
                        anchors.margins: 12
                        spacing: 16

                        ConfidenceGauge {
                            implicitWidth: 72
                            implicitHeight: 72
                            confidence: root.segment ? root.segment.confidence : 0
                            arcColor: root.segment ? root.segment.color : Theme.accent
                        }
                        ColumnLayout {
                            Layout.fillWidth: true
                            spacing: 6
                            Text {
                                text: root.segment && root.segment.overridden
                                    ? "Manually overridden" : "Model confidence"
                                color: Theme.textSecondary
                                font.family: Theme.fontUi
                                font.pixelSize: 11
                            }
                            Text {
                                text: (modelCard.windowSec > 0
                                       ? (modelCard.windowSec < 10 ? modelCard.windowSec.toFixed(1)
                                                                    : modelCard.windowSec.toFixed(0))
                                         + " s analysis window"
                                       : "model analysis window")
                                color: Theme.textMuted
                                font.family: Theme.fontMono
                                font.pixelSize: 10
                                lineHeight: 1.4
                            }
                            Text {
                                // predictive uncertainty (softmax entropy) — amber when the grade is shaky
                                visible: root.segment && root.segment.uncertainty > 0.02
                                text: "uncertainty " + (root.segment ? Math.round(root.segment.uncertainty * 100) : 0) + "%"
                                color: (root.segment && root.segment.uncertainty > 0.5) ? Theme.warnColor : Theme.textMuted
                                font.family: Theme.fontMono
                                font.pixelSize: 10
                            }
                        }
                    }
                }

                // -- conformal prediction set (research3 UQ) — an ambiguity signal from the card's
                // calibrated APS threshold: set size 1 = the model commits to one grade here; ≥2 = it
                // can't cleanly separate those grades. (Surfaced as ambiguity, not a hard coverage %:
                // the threshold's coverage is only as good as the reference set it was calibrated on.)
                Rectangle {
                    id: confSetCard
                    visible: root.segment && root.segment.conformalSet && root.segment.conformalSet.length > 0
                    Layout.fillWidth: true
                    readonly property bool amb: root.segment ? root.segment.ambiguous : false
                    color: confSetCard.amb ? Qt.rgba(0.878, 0.639, 0.18, 0.10)
                                           : Qt.rgba(0.18, 0.82, 0.71, 0.08)
                    border.color: confSetCard.amb ? Theme.warnColor : Theme.tealText
                    border.width: 1
                    radius: 9
                    implicitHeight: confSetCol.implicitHeight + 20
                    ColumnLayout {
                        id: confSetCol
                        anchors.left: parent.left; anchors.right: parent.right
                        anchors.top: parent.top; anchors.margins: 10
                        spacing: 3
                        RowLayout {
                            spacing: 7
                            Text {
                                text: confSetCard.amb ? "⚠" : "✓"
                                color: confSetCard.amb ? Theme.warnColor : Theme.tealText
                                font.pixelSize: 13; font.bold: true
                            }
                            Text {
                                text: confSetCard.amb ? "Ambiguous grade" : "Confident grade"
                                color: confSetCard.amb ? Theme.warnColor : Theme.tealText
                                font.family: Theme.fontUi; font.pixelSize: 12; font.weight: Font.DemiBold
                            }
                            Item { Layout.fillWidth: true }
                            Text {
                                text: "{ " + (root.segment ? root.segment.conformalSet.join(", ") : "") + " }"
                                color: Theme.textSecondary
                                font.family: Theme.fontMono; font.pixelSize: 11
                            }
                        }
                        Text {
                            text: confSetCard.amb
                                ? "The calibrated conformal set holds more than one grade — the model can't cleanly separate these here. Treat the segment with the caution of the worst tier in the set."
                                : "The conformal prediction set is a single grade — the model commits cleanly here."
                            color: Theme.textMuted
                            font.family: Theme.fontUi; font.pixelSize: 10
                            wrapMode: Text.WordWrap; Layout.fillWidth: true
                        }
                    }
                }

                // -- interpretable classical-SQI breakdown (research3 explainability) --
                SqiBreakdownPanel {
                    visible: root.segment !== null && root.segment !== undefined
                             && guard.sqiBreakdown && guard.sqiBreakdown.length > 0
                    Layout.fillWidth: true
                    tier: root.segment ? root.segment.tier : ""
                }

                // -- per-modality task usability (EEG per-band, EDA tonic/phasic) --
                UsabilityPanel {
                    visible: root.segment !== null && root.segment !== undefined
                             && guard.usabilityVerdicts && guard.usabilityVerdicts.length > 0
                    Layout.fillWidth: true
                }

                ArtifactChips {
                    visible: root.segment !== null && root.segment !== undefined
                    Layout.fillWidth: true
                    artifacts: root.segment ? root.segment.artifacts : []
                }

                RationaleBox {
                    visible: root.segment !== null && root.segment !== undefined
                    Layout.fillWidth: true
                    rationale: root.segment ? root.segment.rationale : ""
                    tierColor: root.segment ? root.segment.color : Theme.accent
                }

                // -- on-demand LLM audit (off the decision path) ------------
                ColumnLayout {
                    visible: root.segment !== null && root.segment !== undefined
                    Layout.fillWidth: true
                    spacing: 8

                    OutlineButton {
                        Layout.fillWidth: true
                        text: guard.auditPending ? "Auditing…" : "🔎  Audit this segment (LLM)"
                        enabled: !guard.auditPending
                        onClicked: guard.requestAudit(
                            root.segment.startSec, root.segment.endSec,
                            root.segment.tier, root.segment.confidence)
                    }

                    Rectangle {
                        visible: guard.hasAudit
                        Layout.fillWidth: true
                        color: Theme.bgPanelAlt
                        radius: Theme.radiusControl
                        border.color: guard.auditError ? Theme.warnColor : Theme.borderColor
                        border.width: 1
                        implicitHeight: auditText.implicitHeight + 22

                        Rectangle {  // left accent bar
                            anchors.left: parent.left; anchors.top: parent.top; anchors.bottom: parent.bottom
                            width: 2; radius: 1
                            color: guard.auditError ? Theme.warnColor : Theme.accent
                        }
                        Text {
                            id: auditText
                            anchors.fill: parent
                            anchors.leftMargin: 13; anchors.rightMargin: 13
                            anchors.topMargin: 11; anchors.bottomMargin: 11
                            text: guard.auditText
                            color: guard.auditError ? Theme.warnColor : Theme.textBody
                            font.family: Theme.fontUi; font.pixelSize: 12
                            lineHeight: 1.45; wrapMode: Text.WordWrap
                        }
                    }
                }

                HumanInLoopControls {
                    visible: root.segment !== null && root.segment !== undefined
                    Layout.fillWidth: true
                }

                Item { Layout.preferredHeight: 8 }
            }
        }
    }
}
