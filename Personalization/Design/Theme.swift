import SwiftUI
import UIKit

extension Color {
    /// 0xRRGGBB
    init(hex: UInt32) {
        self.init(red: Double((hex >> 16) & 0xFF) / 255, green: Double((hex >> 8) & 0xFF) / 255, blue: Double(hex & 0xFF) / 255)
    }

    /// A color that switches with light and dark mode.
    init(light: UInt32, dark: UInt32) {
        self.init(UIColor { traits in
            UIColor(Color(hex: traits.userInterfaceStyle == .dark ? dark : light))
        })
    }
}

/// The look of the app: warm paper, thick ink outlines, and one loud color per screen.
enum Theme {
    // Accents (bright enough to carry dark text on top of them)
    static let tomato = Color(hex: 0xFF6B57)
    static let sunflower = Color(hex: 0xFFC83D)
    static let mint = Color(hex: 0x38D6A8)
    static let sky = Color(hex: 0x4DB2FF)
    static let grape = Color(hex: 0x9B7BFF)
    static let bubblegum = Color(hex: 0xFF86BE)
    static let palette = [tomato, sunflower, mint, sky, grape, bubblegum]

    // Surfaces and text
    static let paper = Color(light: 0xFFF4E0, dark: 0x1F1A26)
    static let card = Color(light: 0xFFFFFF, dark: 0x2C2535)
    static let ink = Color(light: 0x2A2130, dark: 0xF6EEDF)
    /// Text that sits on a bright accent fill: always dark, in both modes.
    static let onAccent = Color(hex: 0x2A2130)
}

/// Warm paper with a couple of big soft blobs in the screen's color.
struct ScreenBackground: View {
    let color: Color
    var body: some View {
        ZStack {
            Theme.paper
            Circle().fill(color.opacity(0.22)).frame(width: 300).offset(x: 150, y: -260)
            Circle().fill(Theme.palette[((Theme.palette.firstIndex(of: color) ?? 0) + 3) % Theme.palette.count].opacity(0.14))
                .frame(width: 260).offset(x: -170, y: 330)
        }
        .ignoresSafeArea()
    }
}

/// A card that looks like a sticker: thick ink outline and a hard, offset shadow in a bright color.
struct StickerCard<Content: View>: View {
    var tint: Color = Theme.sky
    var tilt: Double = 0
    @ViewBuilder var content: Content

    var body: some View {
        content
            .padding(16)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Theme.card, in: RoundedRectangle(cornerRadius: 22, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 22, style: .continuous).stroke(Theme.ink, lineWidth: 2.5))
            .background(RoundedRectangle(cornerRadius: 22, style: .continuous).fill(tint).offset(x: 5, y: 6))
            .rotationEffect(.degrees(tilt))
            .padding(.trailing, 5)
            .padding(.bottom, 6)
    }
}

/// A chunky button that physically presses into the page.
struct StickerButtonStyle: ButtonStyle {
    var color: Color
    @Environment(\.isEnabled) private var isEnabled

    func makeBody(configuration: Configuration) -> some View {
        let pressed = configuration.isPressed
        configuration.label
            .font(.headline.weight(.heavy))
            .foregroundStyle(Theme.onAccent)
            .padding(.vertical, 14)
            .padding(.horizontal, 18)
            .frame(maxWidth: .infinity)
            .background(isEnabled ? color : Color.gray.opacity(0.35), in: Capsule())
            .overlay(Capsule().stroke(Theme.ink, lineWidth: 2.5))
            .background(Capsule().fill(Theme.ink).offset(x: pressed ? 1 : 4, y: pressed ? 1 : 5))
            .offset(x: pressed ? 3 : 0, y: pressed ? 4 : 0)
            .opacity(isEnabled ? 1 : 0.55)
            .padding(.trailing, 4)
            .padding(.bottom, 5)
            .animation(.snappy(duration: 0.12), value: pressed)
    }
}

/// Big friendly screen title with a small italic aside under it.
struct ScreenHeader: View {
    let title: String
    let subtitle: String
    let color: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.system(size: 36, weight: .black, design: .rounded))
                .foregroundStyle(Theme.ink)
            Text(subtitle)
                .font(.system(.callout, design: .serif))
                .italic()
                .foregroundStyle(Theme.ink.opacity(0.7))
            Capsule().fill(color).frame(width: 72, height: 9)
                .overlay(Capsule().stroke(Theme.ink, lineWidth: 2))
                .padding(.top, 2)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// A collapsible "how does this work?" card. Tap to open for the longer explanation.
struct Explainer<Content: View>: View {
    let title: String
    var color: Color = Theme.sunflower
    var icon = "lightbulb.fill"
    @ViewBuilder var content: Content
    @State private var isOpen = false

    var body: some View {
        StickerCard(tint: color) {
            DisclosureGroup(isExpanded: $isOpen) {
                VStack(alignment: .leading, spacing: 10) { content }
                    .font(.callout)
                    .foregroundStyle(Theme.ink.opacity(0.85))
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.top, 10)
            } label: {
                Label {
                    Text(title).font(.subheadline.weight(.bold)).foregroundStyle(Theme.ink)
                } icon: {
                    Image(systemName: icon).foregroundStyle(color)
                        .shadow(color: Theme.ink, radius: 0, x: 1, y: 1)
                }
            }
            .tint(Theme.ink)
        }
    }
}

/// A round colored badge with the class's first letter. The color comes from the name, so it stays the same.
struct ClassAvatar: View {
    let name: String
    var size: CGFloat = 48

    static func color(for name: String) -> Color {
        let seed = name.unicodeScalars.reduce(7) { ($0 &* 31 &+ Int($1.value)) % 1_000_003 }
        return Theme.palette[seed % Theme.palette.count]
    }
    private var tilt: Double {
        Double(name.unicodeScalars.reduce(3) { ($0 &* 17 &+ Int($1.value)) % 997 } % 9) - 4
    }

    var body: some View {
        Text(String(name.trimmingCharacters(in: .whitespaces).first.map { String($0).uppercased() } ?? "?"))
            .font(.system(size: size * 0.5, weight: .black, design: .rounded))
            .foregroundStyle(Theme.onAccent)
            .frame(width: size, height: size)
            .background(Self.color(for: name), in: Circle())
            .overlay(Circle().stroke(Theme.ink, lineWidth: 2.5))
            .rotationEffect(.degrees(tilt))
    }
}

/// A chunky 10-block meter for confidence.
struct ConfidenceMeter: View {
    let value: Float
    let color: Color

    var body: some View {
        HStack(spacing: 4) {
            ForEach(0..<10, id: \.self) { i in
                RoundedRectangle(cornerRadius: 4, style: .continuous)
                    .fill(Float(i) < value * 10 ? color : Theme.ink.opacity(0.08))
                    .frame(height: 16)
                    .overlay(RoundedRectangle(cornerRadius: 4, style: .continuous).stroke(Theme.ink, lineWidth: 1.5))
            }
        }
    }
}

/// A small pill for numbers like "0.84 s".
struct StatChip: View {
    let label: String
    let value: String
    let color: Color

    var body: some View {
        VStack(spacing: 2) {
            Text(value).font(.system(.headline, design: .rounded).weight(.heavy))
            Text(label).font(.caption2.weight(.semibold)).opacity(0.75)
        }
        .foregroundStyle(Theme.onAccent)
        .padding(.vertical, 8)
        .frame(maxWidth: .infinity)
        .background(color, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).stroke(Theme.ink, lineWidth: 2))
    }
}

/// A selectable option card (used for choosing the head).
struct ChoiceCard: View {
    let title: String
    let detail: String
    let icon: String
    let color: Color
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 12) {
                Image(systemName: icon).font(.title2.weight(.bold))
                    .frame(width: 44, height: 44)
                    .background(color, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                    .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).stroke(Theme.ink, lineWidth: 2))
                    .foregroundStyle(Theme.onAccent)
                VStack(alignment: .leading, spacing: 2) {
                    Text(title).font(.headline.weight(.bold))
                    Text(detail).font(.caption).opacity(0.7)
                }
                .foregroundStyle(Theme.ink)
                Spacer()
                Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                    .font(.title3)
                    .foregroundStyle(isSelected ? color : Theme.ink.opacity(0.3))
                    .shadow(color: isSelected ? Theme.ink : .clear, radius: 0, x: 1, y: 1)
            }
            .padding(12)
            .background(isSelected ? color.opacity(0.22) : Color.clear, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous)
                .stroke(Theme.ink.opacity(isSelected ? 1 : 0.25), lineWidth: isSelected ? 2.5 : 1.5))
        }
        .buttonStyle(.plain)
        .animation(.snappy(duration: 0.15), value: isSelected)
    }
}
