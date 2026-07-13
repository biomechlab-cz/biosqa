import QtQuick
import "../"

// Shared per-tier quality ribbon: colored bands across the item width, driven by a
// `bands` array of {start, width, tier} in 0..1 fractions. Q0 bands get a diagonal
// hatch overlay (the redundant non-color encoding, design spec (c)). Used by the
// channel-row sparkline, minimap ribbon, and per-modality overview timeline.
//
// ONE Canvas rather than a Repeater of Rectangles (each of which nested a further hatch
// Canvas): a long recording has thousands of bands, and the ribbon is only a few pixels
// tall -- N QQuickItems for it is pure overhead. Nothing here is clickable or hoverable,
// so no hit-testing is lost.
Item {
    id: root
    property var bands: []          // [{ start: 0..1, width: 0..1, tier: "Q0".."Q3" }]
    property real radius: 0
    clip: true

    Canvas {
        id: ribbon
        anchors.fill: parent
        readonly property var watch: root.bands
        readonly property bool cbPal: Theme.useColorBlindPalette
        onWatchChanged: requestPaint()
        onCbPalChanged: requestPaint()
        readonly property real rr: root.radius
        onRrChanged: requestPaint()
        // A Canvas clears its backing store on resize without re-emitting paint, so the ribbon went
        // blank whenever its row was re-laid out (channel list, segment cards, responsive grid).
        onWidthChanged: requestPaint()
        onHeightChanged: requestPaint()
        onPaint: {
            var ctx = getContext("2d")
            ctx.reset()
            var bs = root.bands
            if (!bs || bs.length === 0)
                return
            if (ribbon.rr > 0) {                 // rounded ribbon ends (Item.clip can't round)
                var r = Math.min(ribbon.rr, width / 2, height / 2)
                ctx.beginPath()
                ctx.moveTo(r, 0)
                ctx.lineTo(width - r, 0)
                ctx.arcTo(width, 0, width, r, r)
                ctx.lineTo(width, height - r)
                ctx.arcTo(width, height, width - r, height, r)
                ctx.lineTo(r, height)
                ctx.arcTo(0, height, 0, height - r, r)
                ctx.lineTo(0, r)
                ctx.arcTo(0, 0, r, 0, r)
                ctx.closePath()
                ctx.clip()
            }
            for (var i = 0; i < bs.length; i++) {
                var p = Theme.tierInfo(bs[i].tier)
                var x = bs[i].start * width
                var w = Math.max(1, bs[i].width * width)
                ctx.fillStyle = p.color
                ctx.fillRect(x, 0, w, height)
                if (!p.hatch)
                    continue
                // Q0 diagonal hatch (redundant encoding), clipped to this band.
                ctx.save()
                ctx.beginPath()
                ctx.rect(x, 0, w, height)
                ctx.clip()
                ctx.strokeStyle = "rgba(10,13,19,0.45)"
                ctx.lineWidth = 1.2
                for (var d = x - height; d < x + w; d += 5) {
                    ctx.beginPath()
                    ctx.moveTo(d, height)
                    ctx.lineTo(d + height, 0)
                    ctx.stroke()
                }
                ctx.restore()
            }
        }
    }
}
