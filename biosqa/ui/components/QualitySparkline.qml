import QtQuick
import "../"

// Shared per-tier quality ribbon: colored bands across the item width, driven by a
// `bands` array of {start, width, tier} in 0..1 fractions. Q0 bands get a diagonal
// hatch overlay (the redundant non-color encoding, design spec (c)). Used by the
// channel-row sparkline, minimap ribbon, and per-modality overview timeline.
Item {
    id: root
    property var bands: []          // [{ start: 0..1, width: 0..1, tier: "Q0".."Q3" }]
    property real radius: 0
    clip: true

    Repeater {
        model: root.bands
        Rectangle {
            required property var modelData
            readonly property var _p: (Theme.currentQualityPalette()[modelData.tier]
                                       || Theme.currentQualityPalette()["Q3"])
            x: modelData.start * root.width
            width: Math.max(1, modelData.width * root.width)
            height: root.height
            radius: root.radius
            color: _p.color

            // Q0 diagonal hatch (redundant encoding)
            Canvas {
                anchors.fill: parent
                visible: !!parent._p.hatch
                onPaint: {
                    var ctx = getContext("2d")
                    ctx.reset()
                    ctx.strokeStyle = "rgba(10,13,19,0.45)"
                    ctx.lineWidth = 1.2
                    for (var i = -height; i < width; i += 5) {
                        ctx.beginPath(); ctx.moveTo(i, height); ctx.lineTo(i + height, 0); ctx.stroke()
                    }
                }
            }
        }
    }
}
