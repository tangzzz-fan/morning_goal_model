import SwiftUI

extension Color {
    enum Design {
        static let deepIndigo = Color(hex: "2A2F4D") // 主背景色 - 深靛蓝
        static let sunriseGold = Color(hex: "FFB800") // 强调色 - 日出金
        static let softWhite = Color(hex: "F0F0F0") // 主文本色 - 柔白
        static let mutedGray = Color(hex: "A0A0A0") // 次要文本色 - 浅灰
        static let darkIndigo = Color(hex: "1A1F3D") // 深色调背景
        static let lightGold = Color(hex: "FFD700") // 浅金色（用于高光）
        static let accentPink = Color(hex: "FF6B6B") // 强调色 - 粉红
        static let accentCyan = Color(hex: "4ECDC4") // 强调色 - 青色
        static let cardBackground = Color(hex: "252A48") // 卡片背景
    }

    init(hex: String) {
        let hex = hex.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
        var int: UInt64 = 0
        Scanner(string: hex).scanHexInt64(&int)
        let alpha, red, green, blue: UInt64
        switch hex.count {
        case 3:
            (alpha, red, green, blue) = (255, (int >> 8) * 17, (int >> 4 & 0xF) * 17, (int & 0xF) * 17)
        case 6:
            (alpha, red, green, blue) = (255, int >> 16, int >> 8 & 0xFF, int & 0xFF)
        case 8:
            (alpha, red, green, blue) = (int >> 24, int >> 16 & 0xFF, int >> 8 & 0xFF, int & 0xFF)
        default:
            (alpha, red, green, blue) = (255, 0, 0, 0)
        }
        self.init(
            .sRGB,
            red: Double(red) / 255,
            green: Double(green) / 255,
            blue: Double(blue) / 255,
            opacity: Double(alpha) / 255
        )
    }
}

enum Typography {
    static let title = Font.system(size: 28, weight: .regular, design: .default)
    static let headline = Font.system(size: 20, weight: .regular, design: .default)
    static let body = Font.system(size: 16, weight: .regular, design: .default)
    static let caption = Font.system(size: 14, weight: .regular, design: .default)
    static let streakNumber = Font.system(size: 32, weight: .medium, design: .default)
}

enum Spacing {
    static let xs: CGFloat = 4
    static let sm: CGFloat = 8
    static let md: CGFloat = 16
    static let lg: CGFloat = 24
    static let xl: CGFloat = 32
}

enum CornerRadius {
    static let sm: CGFloat = 8
    static let md: CGFloat = 12
    static let lg: CGFloat = 16
    static let circular: CGFloat = 999
}
