import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "components"

// design spec (a): "TopBar (52px) recording chip . search/(cmd)K . model
// status pill . theme . Export"
Rectangle {
    id: root
    color: Theme.bgPanelAlt

    function _fmtDur(sec) {
        sec = Math.max(0, Math.round(sec))
        var h = Math.floor(sec / 3600)
        var m = Math.floor((sec % 3600) / 60)
        var s = sec % 60
        function p(n) { return (n < 10 ? "0" : "") + n }
        return p(h) + ":" + p(m) + ":" + p(s)
    }

    // Single-edge hairline: bottom border only (design mockup).
    Rectangle {
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        height: 1
        color: Theme.borderColor
    }

    RowLayout {
        anchors.fill: parent
        anchors.leftMargin: 14
        anchors.rightMargin: 14
        spacing: 16

        // ---- Brand: gradient logo tile + two-tone wordmark ----------------
        RowLayout {
            spacing: 10

            Rectangle {
                Layout.preferredWidth: 26
                Layout.preferredHeight: 26
                radius: 7
                gradient: Gradient {
                    orientation: Gradient.Vertical
                    GradientStop { position: 0.0; color: Theme.accent }
                    GradientStop { position: 1.0; color: Theme.accentDim }
                }

                // Small white ECG polyline glyph (mockup viewBox 0 0 16 16).
                Canvas {
                    anchors.centerIn: parent
                    width: 16
                    height: 16
                    onPaint: {
                        var ctx = getContext("2d")
                        ctx.reset()
                        ctx.strokeStyle = Theme.bgApp
                        ctx.lineWidth = 1.6
                        ctx.lineJoin = "round"
                        ctx.lineCap = "round"
                        var pts = [[1,8],[4,8],[5.5,3],[7.5,13],[9,8],[11,8],[12.5,5.5],[14,8],[15,8]]
                        ctx.beginPath()
                        ctx.moveTo(pts[0][0], pts[0][1])
                        for (var i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1])
                        ctx.stroke()
                    }
                }
            }

            RowLayout {
                spacing: 0
                Label {
                    text: "BioSQA"
                    color: Theme.textPrimary
                    font.family: Theme.fontUi
                    font.weight: Font.DemiBold
                    font.pixelSize: 14
                }
                Label {
                    text: " Studio"
                    color: Theme.textSecondary
                    font.family: Theme.fontUi
                    font.weight: Font.Medium
                    font.pixelSize: 14
                }
            }
        }

        // Divider between logo and recording chip (mockup 1x22).
        Rectangle {
            Layout.preferredWidth: 1
            Layout.preferredHeight: 22
            color: Theme.borderColor
        }

        // ---- Recording chip -----------------------------------------------
        Rectangle {
            id: recChip
            radius: Theme.radiusControl
            color: recChipHover.hovered ? Theme.hoverBg : Theme.bgPanel
            border.color: Theme.borderColor
            border.width: 1
            Layout.preferredHeight: 30
            implicitWidth: recRow.implicitWidth + 22
            visible: recordings.count > 0

            HoverHandler { id: recChipHover }

            RowLayout {
                id: recRow
                anchors.centerIn: parent
                spacing: 9

                Rectangle {
                    Layout.preferredWidth: 7; Layout.preferredHeight: 7; radius: 3.5
                    color: Theme.statusRed
                }
                Label {
                    text: recordings.currentName || "—"
                    color: Theme.textPrimary
                    font.family: Theme.fontMono
                    font.pixelSize: 12
                    elide: Text.ElideMiddle
                    Layout.maximumWidth: 260
                }
                Label {
                    text: root._fmtDur(recordings.currentDurationSec)
                    color: Theme.textMuted
                    font.family: Theme.fontMono
                    font.pixelSize: 12
                }
                Label {
                    text: "⌄"  // caret
                    color: Theme.textMuted
                    font.pixelSize: 12
                    Layout.leftMargin: 2
                }
            }
        }

        Item { Layout.fillWidth: true } // spacer

        // ---- Model status pill (design spec (b)) --------------------------
        Rectangle {
            id: statusPill
            radius: height / 2
            color: Theme.pillBg
            border.width: 1
            border.color: Qt.rgba(Theme.accent.r, Theme.accent.g, Theme.accent.b, 0.28)
            Layout.preferredHeight: 26
            implicitWidth: statusRow.implicitWidth + 22

            RowLayout {
                id: statusRow
                anchors.centerIn: parent
                spacing: 8

                // Accent dot with soft glow (larger translucent rect behind).
                Item {
                    Layout.preferredWidth: 6
                    Layout.preferredHeight: 6
                    Rectangle {
                        anchors.centerIn: parent
                        width: 14; height: 14; radius: 7
                        color: Qt.rgba(Theme.accent.r, Theme.accent.g, Theme.accent.b, 0.35)
                    }
                    Rectangle {
                        anchors.centerIn: parent
                        width: 6; height: 6; radius: 3
                        color: Theme.accent
                    }
                }
                Label {
                    // Compact: status + precision + latency. The model version lives in
                    // the Overview model card, so it's dropped here to keep the pill short.
                    // Precision and latency are FACTS ABOUT THE MODEL: each clause appears only
                    // when the engine actually reported it. The pill used to hardcode
                    // "FP32 · 2.1 ms/win" — a latency nothing had ever measured.
                    objectName: "modelStatusText"
                    text: {
                        var parts = [inference.statusText || "no model"]
                        var prec = inference.precision !== undefined ? inference.precision : ""
                        if (prec.length > 0)
                            parts.push(prec)
                        if (inference.latencyMs > 0)
                            parts.push(inference.latencyMs.toFixed(1) + " ms/win")
                        return parts.join(" · ")
                    }
                    color: Theme.tealText
                    font.family: Theme.fontMono
                    font.pixelSize: 11
                    elide: Text.ElideRight
                    Layout.maximumWidth: 360
                }
            }
        }

        // ---- Theme / color-blind-palette toggle (32x32) -------------------
        Button {
            id: themeBtn
            implicitWidth: 32
            implicitHeight: 32
            padding: 0
            onClicked: {
                settings.setThemeDark(!settings.themeDark)
                AppController.toast(settings.themeDark ? "Dark theme" : "Light theme")
            }

            ToolTip.visible: hovered
            ToolTip.text: Theme.dark ? "Switch to light theme" : "Switch to dark theme"
            ToolTip.delay: 400

            Accessible.role: Accessible.Button
            Accessible.name: themeBtn.ToolTip.text

            background: Rectangle {
                radius: Theme.radiusControl
                color: themeBtn.hovered ? Theme.hoverBg : Theme.bgPanel
                border.width: 1
                border.color: Theme.borderColor
            }
            contentItem: Canvas {
                property color stroke: themeBtn.hovered ? Theme.textPrimary : Theme.textSecondary
                property bool isDark: Theme.dark
                onStrokeChanged: requestPaint()
                onIsDarkChanged: requestPaint()
                onPaint: {
                    var ctx = getContext("2d")
                    ctx.reset()
                    var cx = width / 2, cy = height / 2
                    ctx.strokeStyle = stroke
                    ctx.fillStyle = stroke
                    ctx.lineWidth = 1.3
                    ctx.lineCap = "round"
                    if (isDark) {
                        // crescent moon (currently dark → click for light)
                        ctx.beginPath()
                        ctx.arc(cx, cy, 6, 0, 2 * Math.PI)
                        ctx.fill()
                        ctx.fillStyle = themeBtn.hovered ? Theme.hoverBg : Theme.bgPanel
                        ctx.beginPath()
                        ctx.arc(cx + 3, cy - 2, 6, 0, 2 * Math.PI)
                        ctx.fill()
                    } else {
                        // sun (currently light → click for dark)
                        ctx.beginPath()
                        ctx.arc(cx, cy, 3.4, 0, 2 * Math.PI)
                        ctx.stroke()
                        for (var i = 0; i < 8; i++) {
                            var a = i * Math.PI / 4
                            ctx.beginPath()
                            ctx.moveTo(cx + Math.cos(a) * 5.5, cy + Math.sin(a) * 5.5)
                            ctx.lineTo(cx + Math.cos(a) * 7.5, cy + Math.sin(a) * 7.5)
                            ctx.stroke()
                        }
                    }
                }
            }
        }

        // ---- Export (accent CTA + format menu) ----------------------------
        AccentButton {
            id: exportBtn
            text: "Export"
            glyph: "⤓"
            onClicked: exportMenu.popup(exportBtn, 0, exportBtn.height + 2)
        }
        Menu {
            id: exportMenu
            Repeater {
                model: [
                    { label: "CSV table", fmt: "csv" },
                    { label: "TSV (BIDS events)", fmt: "tsv" },
                    { label: "JSON quality report", fmt: "json" },
                    { label: "Parquet", fmt: "parquet" },
                    { label: "WFDB annotation", fmt: "wfdb" },
                    { label: "MATLAB .mat", fmt: "mat" }
                ]
                delegate: MenuItem {
                    required property var modelData
                    text: modelData.label
                    onTriggered: exporter.exportSelection(modelData.fmt)
                }
            }
        }
    }
}
