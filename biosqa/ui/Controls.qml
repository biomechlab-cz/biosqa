import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "workspace"

// Plan 2 §4 `ui/controls.py` equivalent: "channel toggles, quality filter,
// time jump, export buttons". In the QML adaptation this composes the
// channel list + quality legend delegates (workspace/) with a small
// filter/time-jump/export toolbar, and is used as the lower half of
// WorkspaceView's left dock column (design spec (a): FileTreePanel above,
// ChannelListPanel + QualityLegend below, in the same column).
ColumnLayout {
    id: root
    spacing: 0

    ChannelListPanel {
        Layout.fillWidth: true
        Layout.fillHeight: true
    }

    QualityLegend {
        Layout.fillWidth: true
        // Fits the "QUALITY LEGEND" header + all 4 tier rows (Q3/Q2/Q1/Q0)
        // without clipping Q0. Not fillHeight, so ChannelListPanel takes the
        // vertical slack instead of squeezing the legend.
        Layout.fillHeight: false
        Layout.preferredHeight: 140
        Layout.minimumHeight: 140
    }
}
