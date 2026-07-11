import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../"
import "../components"

// design spec (b): "ModelCardPanel | ModelCardModel | parsed model_card.json k/v
// rows" (mockup lines 392-403). Bound to the Python `modelCard` context property.
// A GlassPanel with a custom accent "lens" header instead of the plain title slot.
GlassPanel {
    id: root
    pad: 20
    spacing: 9

    // ---- accent lens header --------------------------------------------------
    RowLayout {
        Layout.fillWidth: true
        Layout.bottomMargin: 5
        spacing: 8

        Item {
            width: 14; height: 14
            Rectangle {
                anchors.fill: parent
                radius: 3
                color: "transparent"
                border.width: 1
                border.color: Theme.accent
            }
            Rectangle {
                anchors.centerIn: parent
                width: 5; height: 5; radius: 2.5
                color: Theme.accent
            }
        }
        Text {
            text: "Model card"
            color: Theme.textPrimary
            font.family: Theme.fontUi
            font.pixelSize: 13
            font.weight: Font.DemiBold
        }
        Item { Layout.fillWidth: true }
    }

    // ---- key/value rows ------------------------------------------------------
    ListView {
        Layout.fillWidth: true
        Layout.fillHeight: true
        clip: true
        spacing: 9
        interactive: false
        model: modelCard

        delegate: RowLayout {
            id: mcRow
            required property string key
            required property string value
            width: ListView.view.width
            spacing: 10

            Label {
                text: mcRow.key
                color: Theme.textSecondary
                font.family: Theme.fontUi
                font.pixelSize: 11
            }
            Item { Layout.fillWidth: true }
            Label {
                text: mcRow.value
                color: Theme.textPrimary
                font.family: Theme.fontMono
                font.pixelSize: 11
                horizontalAlignment: Text.AlignRight
                // wrap long values (e.g. class_order) instead of eliding mid-token.
                wrapMode: Text.Wrap
                Layout.maximumWidth: mcRow.width * 0.62
                Layout.alignment: Qt.AlignTop
            }
        }
    }
}
