import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import "workspace"
import "overview"
import "inspector"
import "segmentation"

// Root application shell (design spec (a)/§0): a single-page shell whose
// icon rail swaps one of four full-bleed views -- it does NOT dock all four
// simultaneously. `WorkspaceView` (the default/current view) is itself the
// classic docked layout (file tree | plot | AI inspector | minimap), so on
// first launch this window *is* showing the docked layout described in the
// task brief; `OverviewView`/`SegmentInspectorView`/`QualitySegmentationView`
// replace it entirely when the user clicks another rail icon.
ApplicationWindow {
    id: window
    visible: true
    width: 1440
    height: 900
    minimumWidth: 1024
    minimumHeight: 640
    title: "BioSQA Studio"
    color: Theme.bgApp
    // Explicit background so the app base renders in offscreen grabs too (the window
    // `color` alone isn't captured by grabWindow()).
    background: Rectangle { color: Theme.bgApp }

    // Persisted appearance lives in `settings` (QSettings-backed). Theme is a QML singleton
    // and can't reliably read context properties, so the root shell PUSHES settings -> Theme
    // on startup and whenever a setting changes. UI controls write to `settings.setXxx(...)`.
    Component.onCompleted: {
        Theme.dark = settings.themeDark
        Theme.accent = settings.accent
        Theme.useColorBlindPalette = settings.colorBlindTiers
    }
    Connections {
        target: settings
        function onThemeDarkChanged() { Theme.dark = settings.themeDark }
        function onAccentChanged() { Theme.accent = settings.accent }
        function onColorBlindTiersChanged() { Theme.useColorBlindPalette = settings.colorBlindTiers }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        TopBar {
            Layout.fillWidth: true
            Layout.preferredHeight: Theme.topBarHeight
        }

        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 0

            ActivityRail {
                Layout.preferredWidth: Theme.activityRailWidth
                Layout.fillHeight: true
                currentView: AppController.currentView
                onNavigate: (view) => AppController.go(view)
                onSettingsRequested: AppController.openSettings()
            }

            // Design spec (a): `Loader { sourceComponent: viewFor(AppController.currentView) }`.
            Loader {
                id: viewLoader
                Layout.fillWidth: true
                Layout.fillHeight: true
                sourceComponent: viewFor(AppController.currentView)

                function viewFor(view) {
                    switch (view) {
                    case "overview":
                        return overviewComponent
                    case "inspector":
                        return inspectorComponent
                    case "segmentation":
                        return segmentationComponent
                    case "workspace":
                    default:
                        return workspaceComponent
                    }
                }
            }
        }
    }

    // --- shared export save dialog + feedback toast -------------------------
    property string _exportFmt: "csv"
    FileDialog {
        id: exportDialog
        title: "Export quality intervals"
        fileMode: FileDialog.SaveFile
        nameFilters: {
            switch (window._exportFmt) {
            case "parquet": return ["Parquet (*.parquet)"]
            case "json":    return ["JSON quality report (*.json)"]
            case "tsv":     return ["BIDS events TSV (*.tsv)"]
            case "wfdb":    return ["WFDB annotation (*.qual)"]
            case "mat":     return ["MATLAB (*.mat)"]
            default:        return ["CSV (*.csv)"]
            }
        }
        onAccepted: exporter.exportToPath(selectedFile.toString(), window._exportFmt)
    }
    Connections {
        target: exporter
        function onSaveRequested(fmt) { window._exportFmt = fmt; exportDialog.open() }
        function onExportSucceeded(path) { toast.show("Exported → " + path) }
        function onExportFailed(msg) { toast.show(msg) }
    }
    // global toast channel: any component can `AppController.toast("…")`
    Connections {
        target: AppController
        function onNotify(msg) { toast.show(msg) }
    }
    // background auto-detect verification: warn when a forced modality disagrees with the header
    Connections {
        target: recordings
        function onModalityMismatch(used, detected, conf) {
            toast.show("⚠ Opened as " + used.toUpperCase() + " but this looks like " +
                       detected.toUpperCase() + " (" + Math.round(conf * 100) + "% conf) — verify the signal type.")
        }
        function onModalityUncertain(used, conf) {
            toast.show("⚠ Auto-detected " + used.toUpperCase() + " with low confidence (" +
                       Math.round(conf * 100) + "%) — verify the signal type before trusting the grades.")
        }
    }

    // modal settings panel (rail gear / top-bar toggle open it)
    SettingsOverlay {}

    Rectangle {
        id: toast
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 24
        radius: Theme.radiusPanel
        color: Theme.bgPanel
        border.color: Theme.borderColor
        opacity: 0
        width: toastLabel.implicitWidth + 28
        height: 36
        z: 100
        function show(text) { toastLabel.text = text; opacity = 1; toastTimer.restart() }
        Label {
            id: toastLabel
            anchors.centerIn: parent
            color: Theme.textPrimary
            font.family: Theme.fontMono
            font.pixelSize: 12
        }
        Timer { id: toastTimer; interval: 3200; onTriggered: toast.opacity = 0 }
        Behavior on opacity { NumberAnimation { duration: 180 } }
    }

    Component {
        id: workspaceComponent
        WorkspaceView {}
    }
    Component {
        id: overviewComponent
        OverviewView {}
    }
    Component {
        id: inspectorComponent
        SegmentInspectorView {}
    }
    Component {
        id: segmentationComponent
        QualitySegmentationView {}
    }
}
