import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// design spec (a): "ActivityRail (56px) icon nav: Workspace / Overview /
// Segment Inspector / Quality Segmentation / Settings"
Rectangle {
    id: root
    color: Theme.bgRail

    // Single-edge hairline: right border only (design mockup).
    Rectangle {
        anchors.top: parent.top
        anchors.bottom: parent.bottom
        anchors.right: parent.right
        width: 1
        color: Theme.borderColor
    }

    property string currentView: "workspace"
    signal navigate(string view)
    // Settings is a modal/side-panel affordance, not one of AppController's
    // four `VIEWS` (design spec (a) lists it as a 5th rail icon distinct from
    // the Workspace/Overview/Inspector/Segmentation view-switch group).
    signal settingsRequested()

    readonly property var items: [
        { view: "workspace", icon: "activity", label: "Workspace" },
        { view: "overview", icon: "grid", label: "Overview" },
        { view: "inspector", icon: "search", label: "Segment Inspector" },
        { view: "segmentation", icon: "list", label: "Quality Segmentation" }
    ]

    // Rounded-rectangle path helper for the Canvas icons (Qt context2d has no
    // native roundRect on all versions).
    function rrect(ctx, x, y, w, h, r) {
        ctx.beginPath()
        ctx.moveTo(x + r, y)
        ctx.lineTo(x + w - r, y)
        ctx.arcTo(x + w, y, x + w, y + r, r)
        ctx.lineTo(x + w, y + h - r)
        ctx.arcTo(x + w, y + h, x + w - r, y + h, r)
        ctx.lineTo(x + r, y + h)
        ctx.arcTo(x, y + h, x, y + h - r, r)
        ctx.lineTo(x, y + r)
        ctx.arcTo(x, y, x + r, y, r)
        ctx.closePath()
    }

    ColumnLayout {
        anchors.top: parent.top
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.topMargin: 12
        spacing: 6

        Repeater {
            model: root.items
            delegate: ToolButton {
                id: navBtn
                required property var modelData
                implicitWidth: 40
                implicitHeight: 40
                checkable: true
                checked: root.currentView === modelData.view
                onClicked: root.navigate(modelData.view)

                // Active = accent tint only (no border); hover = bgPanel fill.
                background: Rectangle {
                    radius: Theme.radiusNav
                    color: navBtn.checked
                        ? Qt.rgba(Theme.accent.r, Theme.accent.g, Theme.accent.b, 0.14)
                        : (navBtn.hovered ? Theme.bgPanel : "transparent")
                }

                contentItem: Canvas {
                    width: 19
                    height: 19
                    property string iconType: navBtn.modelData.icon
                    property color stroke: navBtn.checked ? Theme.accent : Theme.textMuted
                    onStrokeChanged: requestPaint()
                    onPaint: {
                        var ctx = getContext("2d")
                        ctx.reset()
                        ctx.save()
                        ctx.scale(width / 20, height / 20)
                        ctx.strokeStyle = stroke
                        ctx.fillStyle = stroke
                        ctx.lineWidth = 1.5
                        ctx.lineCap = "round"
                        ctx.lineJoin = "round"
                        if (iconType === "activity") {
                            // Lucide "activity" — an ECG-style pulse polyline
                            ctx.beginPath()
                            ctx.moveTo(2, 10); ctx.lineTo(5.5, 10); ctx.lineTo(8, 3)
                            ctx.lineTo(12, 17); ctx.lineTo(14.5, 10); ctx.lineTo(18, 10)
                            ctx.stroke()
                        } else if (iconType === "grid") {
                            // Lucide "layout-grid" — 2x2 rounded squares
                            root.rrect(ctx, 3, 3, 6.5, 6.5, 1.6); ctx.stroke()
                            root.rrect(ctx, 10.5, 3, 6.5, 6.5, 1.6); ctx.stroke()
                            root.rrect(ctx, 3, 10.5, 6.5, 6.5, 1.6); ctx.stroke()
                            root.rrect(ctx, 10.5, 10.5, 6.5, 6.5, 1.6); ctx.stroke()
                        } else if (iconType === "search") {
                            // Lucide "search" — magnifier (inspect)
                            ctx.beginPath(); ctx.arc(8.5, 8.5, 5.2, 0, 2 * Math.PI); ctx.stroke()
                            ctx.beginPath(); ctx.moveTo(12.4, 12.4); ctx.lineTo(17, 17); ctx.stroke()
                        } else if (iconType === "list") {
                            // Lucide "list" — 3 rows with leading bullets
                            var ys = [4.5, 10, 15.5]
                            for (var j = 0; j < 3; j++) {
                                ctx.beginPath(); ctx.arc(3.5, ys[j], 1.0, 0, 2 * Math.PI); ctx.fill()
                                ctx.beginPath(); ctx.moveTo(7, ys[j]); ctx.lineTo(18, ys[j]); ctx.stroke()
                            }
                        }
                        ctx.restore()
                    }
                }

                ToolTip.visible: hovered
                ToolTip.text: modelData.label
                ToolTip.delay: 400
            }
        }
    }

    // Settings entry pinned to the bottom of the rail (design spec (a)).
    ToolButton {
        id: settingsBtn
        anchors.bottom: parent.bottom
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.bottomMargin: 12
        implicitWidth: 40
        implicitHeight: 40
        onClicked: root.settingsRequested()

        background: Rectangle {
            radius: Theme.radiusNav
            color: settingsBtn.hovered ? Theme.bgPanel : "transparent"
        }

        contentItem: Canvas {
            width: 18
            height: 18
            property color stroke: settingsBtn.hovered ? Theme.textPrimary : Theme.textMuted
            onStrokeChanged: requestPaint()
            onPaint: {
                // Lucide "sliders-horizontal" — 3 tracks with knobs (a clean settings icon)
                var ctx = getContext("2d")
                ctx.reset()
                ctx.save()
                ctx.scale(width / 20, height / 20)
                ctx.strokeStyle = stroke
                ctx.fillStyle = stroke
                ctx.lineWidth = 1.4
                ctx.lineCap = "round"
                var ys = [5, 10, 15]
                var knobs = [13.5, 7, 15]
                for (var i = 0; i < 3; i++) {
                    ctx.beginPath(); ctx.moveTo(3, ys[i]); ctx.lineTo(17, ys[i]); ctx.stroke()
                    ctx.beginPath(); ctx.arc(knobs[i], ys[i], 2.0, 0, 2 * Math.PI); ctx.fill()
                }
                ctx.restore()
            }
        }

        ToolTip.visible: hovered
        ToolTip.text: "Settings"
        ToolTip.delay: 400
    }
}
