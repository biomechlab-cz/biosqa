pragma Singleton
import QtQuick

// Palette + typography singleton (design spec (c)). Referenced directly as
// `Theme.<token>` from any QML file in this directory (implicit) or from a
// subdirectory via `import "../"` (see workspace/overview/inspector/
// segmentation QML). `pragma Singleton` gives every reference the same
// QtObject instance without needing `Theme { }` instantiation anywhere.
QtObject {
    id: theme

    // ---- dark / light theme (toggled from the top bar) ---------------------
    property bool dark: true

    readonly property color bgApp: dark ? "#0A0D13" : "#F4F6FA"        // canvas / plot bg
    readonly property color bgPanel: dark ? "#161B25" : "#FFFFFF"      // right rail, cards
    readonly property color bgPanelAlt: dark ? "#10141C" : "#EEF1F6"    // header, left rail, table head
    readonly property color bgRail: dark ? "#0C0F16" : "#E9EDF3"        // activity rail, axis strip
    readonly property color borderColor: dark ? "#262D3A" : "#D5DCE6"   // primary hairlines
    readonly property color borderSubtle: dark ? "#171d28" : "#E3E8EF"  // inner lane separators
    readonly property color textPrimary: dark ? "#E6EAF2" : "#1A1F2A"   // headings, values
    readonly property color textSecondary: dark ? "#9AA4B6" : "#55607A" // labels
    readonly property color textMuted: dark ? "#626C7E" : "#8A93A6"     // captions, ticks
    readonly property color textBody: dark ? "#c5cdda" : "#3A4256"      // prose (rationale/audit/bar labels)

    // user-configurable accent (default teal); Settings can rebind this to
    // one of `accentOptions` at runtime.
    property color accent: "#35D0BA"
    readonly property var accentOptions: ["#35D0BA", "#5B9CFF", "#C08CF2", "#E0A32E"]

    // ---- fonts --------------------------------------------------------------
    readonly property string fontUi: "Geist"
    readonly property string fontMono: "JetBrains Mono"

    // ---- chrome geometry ------------------------------------------------
    readonly property int radiusControl: 7    // buttons/inputs
    readonly property int radiusPanel: 11      // cards/panels (9-11px)
    readonly property int radiusChip: 5        // chips (3-6px)
    readonly property int radiusNav: 9         // activity-rail nav buttons

    // ---- extra design tokens (from design_files mockup) --------------------
    readonly property color hoverBg: dark ? "#1E2430" : "#E9EDF3"        // control hover fill
    readonly property color borderRow: dark ? "#1a2029" : "#E8ECF2"      // table-row hairlines
    readonly property color chipDark: "#062521"                          // text/icon on accent fills (both themes)
    readonly property color tealText: dark ? "#8fe6d9" : "#12836f"       // model-pill / override text
    readonly property color pillBg: dark ? "#0F1E1B" : "#E4F6F1"         // teal status-pill background
    readonly property color statusRed: "#FF7A85"                         // recording live dot
    readonly property color accentDim: "#1c8f80"                         // logo-gradient dark stop
    readonly property color traceColor: dark ? "#C6D0E2" : "#3A4256"     // waveform stroke
    readonly property color chipBorderMuted: dark ? "#313a49" : "#C7CEDB" // artifact-chip border
    readonly property color borderHover: dark ? "#33415a" : "#B9C3D2"    // control-hover hairline
    // Semantic status tones. The raw amber/red read at ~2:1 on the light theme's white
    // panels, so each is darkened there instead of being hardcoded per call site.
    readonly property color warnColor: dark ? "#E0A32E" : "#8A5A00"
    readonly property color dangerColor: dark ? "#E5484D" : "#B3261E"

    readonly property int topBarHeight: 52
    readonly property int activityRailWidth: 56
    readonly property int fileTreeWidth: 248
    readonly property int inspectorWidth: 322
    readonly property int minimapHeight: 74

    // ---- glassmorphism (hover tooltip / minimap overlays) -------------------
    // Prefer the cheap flat-translucent variant over the plot area; reserve
    // MultiEffect/FastBlur for truly-static overlays (design spec (c)).
    readonly property color glassBg: dark ? Qt.rgba(16 / 255, 20 / 255, 28 / 255, 0.7)
                                          : Qt.rgba(1, 1, 1, 0.82)
    readonly property color glassBorder: dark ? Qt.rgba(1, 1, 1, 0.09) : Qt.rgba(0, 0, 0, 0.08)
    readonly property int glassBlurRadius: 15

    // ---- quality bands Q0-Q3 (default palette) ------------------------------
    // Redundant, load-bearing non-color encoding (design spec (c)) is carried
    // by QualityBandDelegate/QualityLegend, not by this color table alone:
    // each tier also has a glyph and a mono text code, and Q0 additionally
    // gets a diagonal hatch fill.
    readonly property var qualityPalette: ({
        "Q3": { color: "#2FBF71", glyph: "✓", label: "Excellent" },
        "Q2": { color: "#86C440", glyph: "✓", label: "Acceptable" },
        "Q1": { color: "#E0A32E", glyph: "⚠", label: "Poor" },
        "Q0": { color: "#E5484D", glyph: "⊘", label: "Unacceptable", hatch: true }
    })

    // Alternate color-blind-safe diverging palette (blue<->orange), does not
    // rely on red/green discrimination (design spec (c)). Toggle via
    // `theme.useColorBlindPalette` from Settings.
    readonly property var qualityPaletteColorBlind: ({
        "Q3": { color: "#2E86FF", glyph: "✓", label: "Excellent" },
        "Q2": { color: "#7FB3FF", glyph: "✓", label: "Acceptable" },
        "Q1": { color: "#FFB020", glyph: "⚠", label: "Poor" },
        "Q0": { color: "#FF5A36", glyph: "⊘", label: "Unacceptable", hatch: true }
    })

    property bool useColorBlindPalette: false

    function currentQualityPalette() {
        return useColorBlindPalette ? qualityPaletteColorBlind : qualityPalette
    }

    // Neutral descriptor for an ABSENT or unrecognised tier. Every lookup that can miss
    // must land here: falling back to a real tier (the old `?? palette["Q3"]`) paints an
    // invented grade -- for an assessment tool a blank "unknown" is the only honest answer.
    readonly property var unknownTier: ({ color: theme.textMuted, glyph: "", label: "Unknown" })

    function tierInfo(code) {
        return currentQualityPalette()[code] || unknownTier
    }

    function isTier(code) {
        return currentQualityPalette()[code] !== undefined
    }

    // ---- modality colors (channel identity, not quality) --------------------
    readonly property var modalityColors: ({
        "ecg": "#FF7A85",
        "ppg": "#35D0BA",
        "eeg": "#6E8BFF",
        "eda": "#C08CF2"
    })
}
