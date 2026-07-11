import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../"

// design spec (a): the primary docked layout -- "file/channel tree | plot |
// AI inspector | minimap" (the minimap itself is docked *within*
// SignalView's own column, not a separate top-level dock). Uses `SplitView`
// (design spec (a) docking note) rather than fixed `RowLayout` columns so
// the plot region can be widened on small laptops -- runtime-draggable dock
// resize instead of QDockWidget, which QML has no native equivalent of.
SplitView {
    id: root
    orientation: Qt.Horizontal

    // -- left dock: file tree (above) + channel list/legend (below), same
    // column (design spec (a)) -----------------------------------------------
    ColumnLayout {
        SplitView.preferredWidth: Theme.fileTreeWidth
        SplitView.minimumWidth: 180
        spacing: 0

        FileTree {
            Layout.fillWidth: true
            Layout.preferredHeight: parent.height * 0.4
        }

        Controls {
            Layout.fillWidth: true
            Layout.fillHeight: true
        }
    }

    // -- center: plot canvas (SignalCanvas equivalent), minimap docked
    // within its own column (design spec (a)) --------------------------------
    SignalView {
        SplitView.fillWidth: true
        SplitView.minimumWidth: 360
    }

    // -- right dock: AI Quality Inspector -------------------------------------
    QualityInspectorPanel {
        SplitView.preferredWidth: Theme.inspectorWidth
        SplitView.minimumWidth: 240
    }
}
