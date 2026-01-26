import Foundation

enum InsightType: String, Codable {
    case pattern // Pattern Recognition (e.g., "Monday = Work")
    case balance // Balance Advice (e.g., "Too much work")
    case trend // Trend Analysis (e.g., "Positive streak")
    case achievability // Achievability (e.g., "Goal too vague")
    case completion // Completion Rate (e.g., "History says 50%")
    case comparison // Comparison (e.g., "VS last month")
    case encouragement // Encouragement (e.g., "Keep it up!")
    case nostalgia // Nostalgia (e.g., "On this day...")
}

struct Insight: Identifiable, Equatable {
    let id = UUID()
    let type: InsightType
    let title: String
    let description: String
    let priority: Int // Display priority (1-5, 5 is highest)
    let actionSuggestion: String? // Optional actionable advice

    // Metadata for UI
    var iconName: String {
        switch type {
        case .pattern: return "waveform.path.ecg"
        case .balance: return "scale.3d"
        case .trend: return "chart.xyaxis.line"
        case .achievability: return "target"
        case .completion: return "checkmark.circle"
        case .comparison: return "arrow.left.and.right"
        case .encouragement: return "sparkles"
        case .nostalgia: return "clock.arrow.circlepath"
        }
    }
}
