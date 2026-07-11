import QtQuick
import QtQuick.Layouts
import QtQuick.Shapes
import "components"

// Plan 2 §4 `ui/minimap.py` equivalent (design spec (a)): the hours-scale
// navigator -- a coarse waveform, a quality-segment ribbon, and a draggable
// viewport rect over the whole recording. Glass overlay card docked at the
// bottom of SignalView's own column.
Rectangle {
    id: root
    color: Theme.glassBg
    border.color: Theme.glassBorder
    border.width: 1
    radius: Theme.radiusPanel

    // Real full-recording quality-segment bands (normalized 0..1) from the
    // segments context property.
    readonly property var _bands: segments.segmentBands

    ColumnLayout {
        anchors.fill: parent
        anchors.leftMargin: 12
        anchors.rightMargin: 12
        anchors.topMargin: 8
        anchors.bottomMargin: 8
        spacing: 5

        RowLayout {
            Layout.fillWidth: true
            Text {
                text: "RECORDING OVERVIEW"
                color: Theme.textSecondary
                font.family: Theme.fontUi
                font.pixelSize: 10
                font.weight: Font.DemiBold
                font.letterSpacing: 0.6
            }
            Item { Layout.fillWidth: true }
            Text {
                text: "drag viewport to navigate"
                color: Theme.textMuted
                font.family: Theme.fontMono
                font.pixelSize: 9
            }
        }

        Item {
            id: track
            Layout.fillWidth: true
            Layout.fillHeight: true

            // coarse waveform baseline placeholder
            Shape {
                anchors.fill: parent
                anchors.bottomMargin: 10
                ShapePath {
                    strokeWidth: 1
                    strokeColor: Qt.rgba(0.42, 0.46, 0.53, 0.8)
                    fillColor: "transparent"
                    startX: 0
                    startY: track.height * 0.4
                    PathLine { x: track.width; y: track.height * 0.4 }
                }
            }

            // quality-segment ribbon
            QualitySparkline {
                id: ribbon
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.bottom: parent.bottom
                height: 8
                radius: 2
                bands: root._bands
            }

            // selected-segment marker over the whole recording (so a selection made anywhere — table,
            // segment card, jump-to-poor — is visible in the recording overview). A distinct high-contrast
            // TICK just ABOVE the quality ribbon: an accent box over the tier-coloured bands read as a
            // confusing "box within a box" (accent is teal, Q3 is green), and clashed with the accent
            // viewport indicator. This is unambiguously the selection.
            Rectangle {
                property real total: signalView.durationSec > 0 ? signalView.durationSec : 1
                visible: selection.selectedSegment !== null && signalView.durationSec > 0
                x: selection.selectedSegment ? (selection.selectedSegment.startSec / total) * track.width : 0
                width: selection.selectedSegment
                       ? Math.max(3, ((selection.selectedSegment.endSec - selection.selectedSegment.startSec)
                                      / total) * track.width) : 0
                anchors.bottom: ribbon.top
                anchors.bottomMargin: 3
                height: 4
                radius: 2
                color: Theme.textPrimary            // high-contrast vs both the tier bands and the accent viewport
            }

            // viewport indicator — bound to the real current view window
            Rectangle {
                id: viewportRect
                property real totalDur: signalView.durationSec > 0 ? signalView.durationSec : 1
                y: -2
                height: parent.height + 4
                x: (signalView.viewStartSec / totalDur) * track.width
                width: Math.max(6, ((signalView.viewEndSec - signalView.viewStartSec) / totalDur) * track.width)
                color: Qt.rgba(Theme.accent.r, Theme.accent.g, Theme.accent.b, 0.10)
                border.color: Theme.accent
                border.width: 1
                radius: Theme.radiusChip
            }

            // click / drag anywhere on the track to recentre the viewport there
            MouseArea {
                anchors.fill: parent
                cursorShape: Qt.PointingHandCursor
                function navTo(mx) {
                    var span = signalView.viewEndSec - signalView.viewStartSec
                    var total = viewportRect.totalDur
                    var center = Math.max(0, Math.min(mx, track.width)) / track.width * total
                    var ns = Math.max(0, Math.min(center - span / 2, Math.max(0, total - span)))
                    if (ns + span > ns)
                        signalView.setView(ns, ns + span)
                }
                onPressed: (m) => navTo(m.x)
                onPositionChanged: (m) => { if (pressed) navTo(m.x) }
            }
        }
    }
}
