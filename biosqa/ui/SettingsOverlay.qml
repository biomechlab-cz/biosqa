import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "components"

// Modal settings panel (activity-rail gear + top-bar theme toggle open it).
// Four sections, all persisted via the `settings` context property (QSettings-backed):
//   Appearance (theme / accent / color-blind), Analysis (window overlap),
//   Integrity guard (enable + bSQI threshold), LLM audit (enable / host / model / samples).
// Controls READ `settings.<prop>` and WRITE through `settings.set<Prop>(...)` so choices persist.
//
// A real Controls Popup, not an Item with a hand-rolled scrim: that gave no Escape key, no focus
// trap and no keyboard containment -- the "modal" panel was modal only visually. `modal` dims and
// blocks the app behind it, `closePolicy` restores Escape / click-outside, and open state stays in
// sync with AppController.settingsOpen in both directions.
//
// LAYOUT: every setting is a `FormRow` — a left column (title + optional muted description) that
// fills, and a right-aligned control area — so all controls line up in one clean vertical column.
Popup {
    id: root

    parent: Overlay.overlay
    anchors.centerIn: Overlay.overlay
    width: Math.min(720, (parent ? parent.width : 800) - 80)
    height: Math.min(col.implicitHeight + 80, (parent ? parent.height : 600) - 80)
    padding: 0
    z: 300

    modal: true
    focus: true
    closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside

    Accessible.role: Accessible.Dialog
    Accessible.name: "Settings"

    // Two-way sync with the controller, guarded against re-entry: close() -> onClosed ->
    // closeSettings() -> settingsOpen=false -> this handler would call close() again.
    property bool _syncing: false

    Connections {
        target: AppController
        function onSettingsOpenChanged() {
            if (root._syncing)
                return
            root._syncing = true
            if (AppController.settingsOpen)
                root.open()
            else
                root.close()
            root._syncing = false
        }
    }
    onClosed: {
        if (root._syncing)
            return
        root._syncing = true
        AppController.closeSettings()
        root._syncing = false
    }
    Component.onCompleted: if (AppController.settingsOpen) root.open()

    // A single, consistent 2-column form row: title/description on the left (fills), control(s)
    // right-aligned. Declaring child controls inside a FormRow places them in the right column.
    component FormRow: RowLayout {
        id: fr
        property string title: ""
        property string desc: ""
        default property alias controls: ctl.data
        Layout.fillWidth: true
        spacing: 16

        ColumnLayout {
            Layout.fillWidth: true
            Layout.alignment: Qt.AlignVCenter
            spacing: 2
            Label {
                text: fr.title; color: Theme.textSecondary
                font.family: Theme.fontUi; font.pixelSize: 13
                Layout.fillWidth: true; wrapMode: Text.WordWrap
            }
            Label {
                text: fr.desc; visible: fr.desc.length > 0; color: Theme.textMuted
                font.family: Theme.fontUi; font.pixelSize: 11
                Layout.fillWidth: true; wrapMode: Text.WordWrap
            }
        }
        RowLayout {
            id: ctl
            Layout.alignment: Qt.AlignRight | Qt.AlignVCenter
            spacing: 8
        }
    }

    // Uppercase muted section header, consistent across sections.
    component SectionHeader: Label {
        color: Theme.textMuted; font.family: Theme.fontUi
        font.pixelSize: 10; font.letterSpacing: 1.2
        Layout.fillWidth: true; Layout.topMargin: 2
    }

    // the dim behind the panel — the Popup's own modal veil replaces the hand-rolled scrim
    // (which had a bare MouseArea and therefore swallowed keys instead of handling them).
    Overlay.modal: Rectangle { color: Qt.rgba(0, 0, 0, 0.5) }

    background: Rectangle {
        color: Theme.bgPanel
        border.color: Theme.borderColor
        border.width: 1
        radius: Theme.radiusPanel
    }

    // ---- header (fixed) ------------------------------------------------
    RowLayout {
        id: header
        anchors { top: parent.top; left: parent.left; right: parent.right; margins: 20 }
        Label {
            text: "Settings"
            color: Theme.textPrimary
            font.family: Theme.fontUi; font.pixelSize: 16; font.weight: Font.DemiBold
            Layout.fillWidth: true
        }
        Button {
            implicitWidth: 26; implicitHeight: 26; padding: 0
            onClicked: root.close()
            Accessible.role: Accessible.Button
            Accessible.name: "Close settings"
            background: Rectangle {
                radius: Theme.radiusChip
                color: parent.hovered ? Theme.hoverBg : "transparent"
            }
            contentItem: Text {
                text: "✕"; color: Theme.textSecondary
                horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter
            }
        }
    }

    // ---- scrollable body ----------------------------------------------
    ScrollView {
        id: scroll
        anchors { top: header.bottom; left: parent.left; right: parent.right; bottom: parent.bottom
                  topMargin: 12; leftMargin: 20; rightMargin: 20; bottomMargin: 16 }
        clip: true
        contentWidth: availableWidth
        ScrollBar.horizontal.policy: ScrollBar.AlwaysOff

        ColumnLayout {
            id: col
            width: scroll.availableWidth
            spacing: 14

            // ================= APPEARANCE =================
            SectionHeader { text: "APPEARANCE" }

            FormRow {
                title: "Theme"
                Rectangle {
                    implicitWidth: themeSeg.implicitWidth + 6; implicitHeight: 30
                    radius: Theme.radiusControl; color: Theme.bgPanelAlt; border.color: Theme.borderColor
                    RowLayout {
                        id: themeSeg; anchors.centerIn: parent; spacing: 3
                        Repeater {
                            model: [{ k: "Dark", d: true }, { k: "Light", d: false }]
                            delegate: Button {
                                required property var modelData
                                readonly property bool active: settings.themeDark === modelData.d
                                implicitHeight: 24; implicitWidth: 58; padding: 0
                                onClicked: settings.setThemeDark(modelData.d)
                                background: Rectangle { radius: 5; color: parent.active ? Theme.accent : "transparent" }
                                contentItem: Text {
                                    text: modelData.k
                                    color: parent.active ? Theme.chipDark : Theme.textSecondary
                                    font.family: Theme.fontUi; font.pixelSize: 12
                                    horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter
                                }
                            }
                        }
                    }
                }
            }

            FormRow {
                title: "Accent"
                Repeater {
                    model: Theme.accentOptions
                    // Buttons, not bare Rectangle+MouseArea: the swatches are the only way to set the
                    // accent, and as plain Rectangles they were unreachable by keyboard and unnamed to
                    // a screen reader (a color IS the label, so it has to be spoken).
                    delegate: Button {
                        id: swatch
                        required property var modelData
                        readonly property bool active: settings.accent === modelData
                        Layout.alignment: Qt.AlignVCenter
                        implicitWidth: 22; implicitHeight: 22; padding: 0
                        onClicked: settings.setAccent(modelData)

                        ToolTip.visible: hovered
                        ToolTip.text: "Accent " + modelData
                        ToolTip.delay: 400

                        Accessible.role: Accessible.RadioButton
                        Accessible.name: "Accent color " + modelData
                        Accessible.checked: swatch.active

                        background: Rectangle {
                            radius: 11
                            color: swatch.modelData
                            border.width: swatch.active ? 2 : (swatch.visualFocus || swatch.hovered ? 1 : 0)
                            border.color: Theme.textPrimary
                        }
                    }
                }
            }

            FormRow {
                title: "Color-blind-safe tiers"
                desc: "Blue↔orange quality palette (no red/green)"
                Switch { checked: settings.colorBlindTiers; onToggled: settings.setColorBlindTiers(checked) }
            }

            Rectangle { Layout.fillWidth: true; height: 1; color: Theme.borderSubtle }

            // ================= ANALYSIS =================
            SectionHeader { text: "ANALYSIS" }

            FormRow {
                title: "Window overlap"
                desc: "Denser sampling → finer segment boundaries (slower)"
                Rectangle {
                    implicitWidth: ovSeg.implicitWidth + 6; implicitHeight: 30
                    radius: Theme.radiusControl; color: Theme.bgPanelAlt; border.color: Theme.borderColor
                    RowLayout {
                        id: ovSeg; anchors.centerIn: parent; spacing: 3
                        Repeater {
                            model: [{ k: "0%", v: 0.0 }, { k: "25%", v: 0.25 }, { k: "50%", v: 0.5 },
                                    { k: "75%", v: 0.75 }, { k: "90%", v: 0.9 }]
                            delegate: Button {
                                required property var modelData
                                readonly property bool active: Math.abs(settings.windowOverlap - modelData.v) < 0.01
                                implicitHeight: 24; implicitWidth: 40; padding: 0
                                onClicked: settings.setWindowOverlap(modelData.v)
                                background: Rectangle { radius: 5; color: parent.active ? Theme.accent : "transparent" }
                                contentItem: Text {
                                    text: modelData.k
                                    color: parent.active ? Theme.chipDark : Theme.textSecondary
                                    font.family: Theme.fontUi; font.pixelSize: 12
                                    horizontalAlignment: Text.AlignHCenter; verticalAlignment: Text.AlignVCenter
                                }
                            }
                        }
                    }
                }
            }

            Label {
                Layout.fillWidth: true; wrapMode: Text.WordWrap
                text: "Window length is fixed by each model (ECG 10 s @ 250 Hz) and can't be changed here — "
                      + "overlap sets the stride, so 50% steps a half-window at a time for finer segment "
                      + "boundaries. Adjacent windows of the same grade are merged into one segment, so a "
                      + "uniformly graded region shows as a single long band."
                color: Theme.textMuted; font.family: Theme.fontUi; font.pixelSize: 11
            }

            FormRow {
                title: "Recoverability check"
                desc: "Re-score a filtered copy → flag poor windows a filter would make usable"
                Switch { checked: settings.recoveryEnabled; onToggled: settings.setRecoveryEnabled(checked) }
            }

            FormRow {
                title: "Refine segment boundaries"
                desc: "Trim a poor segment to the actual artifact (a short burst won't flag the whole window)"
                Switch { checked: settings.refineBoundaries; onToggled: settings.setRefineBoundaries(checked) }
            }

            Label {
                Layout.fillWidth: true; wrapMode: Text.WordWrap
                text: "Runs a second inference pass on a band-pass-filtered copy. A poor window that "
                      + "becomes usable after filtering is tagged ↺ Recoverable (for ECG/PPG the call is "
                      + "corroborated by a filter-robust bSQI, so the model merely being fooled by filtering "
                      + "isn't mistaken for recovery). Advisory only — the raw grade stays authoritative. "
                      + "Doubles inference time."
                color: Theme.textMuted; font.family: Theme.fontUi; font.pixelSize: 11
            }

            Rectangle { Layout.fillWidth: true; height: 1; color: Theme.borderSubtle }

            // ================= INTEGRITY GUARD =================
            SectionHeader { text: "INTEGRITY GUARD" }

            FormRow {
                title: "False-clean guard"
                desc: "Re-flag pre-filtered windows the bSQI voter marks corrupt"
                Switch { checked: settings.guardEnabled; onToggled: settings.setGuardEnabled(checked) }
            }

            FormRow {
                title: "bSQI threshold"
                opacity: settings.guardEnabled ? 1.0 : 0.4
                Slider {
                    id: bsqiSlider; from: 0.5; to: 0.9; stepSize: 0.01
                    value: settings.bsqiThreshold; enabled: settings.guardEnabled
                    Layout.preferredWidth: 160; onMoved: settings.setBsqiThreshold(value)
                }
                Label {
                    text: settings.bsqiThreshold.toFixed(2); color: Theme.textPrimary
                    font.family: Theme.fontMono; font.pixelSize: 12
                    Layout.preferredWidth: 34; horizontalAlignment: Text.AlignRight
                }
            }

            // Live status: the guard ONLY fires on inputs the app detects as pre-filtered, so on a raw
            // recording moving the slider has no visible effect — say so, and show the live re-flag count
            // when it IS active (so the slider's effect is observable).
            Label {
                Layout.fillWidth: true
                Layout.leftMargin: 2
                visible: settings.guardEnabled
                wrapMode: Text.WordWrap
                font.family: Theme.fontUi; font.pixelSize: 10
                lineHeight: 1.3
                color: (signalView.durationSec > 0 && guard.prefiltered) ? Theme.warnColor : Theme.textMuted
                text: signalView.durationSec <= 0
                    ? "Only affects recordings the app detects as pre-filtered (a “false-clean” input). Lower = stricter (re-flags more deceptively-clean windows)."
                    : (guard.prefiltered
                        ? ("Active on this recording — " + guard.nOverridden + " window(s) re-flagged as deceptively clean. Move the slider to watch the count change; lower = stricter.")
                        : "This recording doesn’t look pre-filtered, so the guard is inactive here — it only re-flags deceptively-clean, pre-filtered inputs. Feed a raw recording to see it act.")
            }

            Rectangle { Layout.fillWidth: true; height: 1; color: Theme.borderSubtle }

            // ================= LLM AUDIT =================
            SectionHeader { text: "LLM AUDIT (local)" }

            FormRow {
                title: "Audit this segment"
                desc: "On-demand second opinion via a local ollama model"
                Switch { checked: settings.auditEnabled; onToggled: settings.setAuditEnabled(checked) }
            }

            FormRow {
                title: "Ollama host"
                opacity: settings.auditEnabled ? 1.0 : 0.4
                TextField {
                    Layout.preferredWidth: 210; enabled: settings.auditEnabled
                    text: settings.ollamaHost; color: Theme.textPrimary
                    font.family: Theme.fontMono; font.pixelSize: 12
                    selectByMouse: true; onEditingFinished: settings.setOllamaHost(text)
                    background: Rectangle { radius: Theme.radiusControl; color: Theme.bgPanelAlt
                                           border.color: parent.activeFocus ? Theme.accent : Theme.borderColor }
                }
            }

            FormRow {
                title: "Model"
                desc: "Larger models (e.g. qwen3:32b) are more accurate but much slower; gemma4 is a faster default"
                opacity: settings.auditEnabled ? 1.0 : 0.4
                TextField {
                    Layout.preferredWidth: 210; enabled: settings.auditEnabled
                    text: settings.auditModel; color: Theme.textPrimary
                    font.family: Theme.fontMono; font.pixelSize: 12
                    selectByMouse: true; onEditingFinished: settings.setAuditModel(text)
                    background: Rectangle { radius: Theme.radiusControl; color: Theme.bgPanelAlt
                                           border.color: parent.activeFocus ? Theme.accent : Theme.borderColor }
                }
            }

            FormRow {
                title: "Self-consistency"
                desc: "Extra passes majority-voted for reliability — each pass is another (slow) model call"
                opacity: settings.auditEnabled ? 1.0 : 0.4
                Slider {
                    id: sampleSlider; from: 1; to: 5; stepSize: 1
                    value: settings.auditSamples; enabled: settings.auditEnabled
                    Layout.preferredWidth: 130; onMoved: settings.setAuditSamples(Math.round(value))
                }
                Label {
                    text: settings.auditSamples + (settings.auditSamples === 1 ? " pass" : " passes")
                    color: Theme.textPrimary; font.family: Theme.fontMono; font.pixelSize: 12
                    Layout.preferredWidth: 62; horizontalAlignment: Text.AlignRight
                }
            }

            // The whole audit section (host/model/self-consistency) only affects the on-demand
            // "Audit this segment" button in the AI Quality Inspector, and only when a local Ollama is
            // running — nothing here changes the automatic Q0..Q3 grades. Set expectations explicitly.
            Label {
                Layout.fillWidth: true
                Layout.leftMargin: 2
                visible: settings.auditEnabled
                wrapMode: Text.WordWrap
                font.family: Theme.fontUi; font.pixelSize: 10
                lineHeight: 1.3
                color: Theme.textMuted
                text: "These settings only affect the on-demand “Audit this segment” button in the inspector "
                    + "(select a segment, then click it) and need a local Ollama running at the host above. "
                    + "They don’t change the automatic quality grades, so you won’t see an effect until you run an audit."
            }

            Rectangle { Layout.fillWidth: true; height: 1; color: Theme.borderSubtle }

            // ================= RESET =================
            RowLayout {
                Layout.fillWidth: true; Layout.bottomMargin: 4
                Item { Layout.fillWidth: true }
                OutlineButton { text: "Reset to defaults"; onClicked: settings.reset() }
            }
        }
    }
}
