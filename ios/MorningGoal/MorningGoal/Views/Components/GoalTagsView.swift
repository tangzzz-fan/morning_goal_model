import SwiftUI

struct GoalTagsView: View {
    let entry: GoalEntry

    var body: some View {
        FlowLayout(spacing: 8) {
            // Topic
            if let topic = entry.effectiveCategory {
                TagPill(text: topic, icon: "tag.fill", color: .blue)
            }

            // Sentiment
            if let sentiment = entry.effectiveSentiment {
                TagPill(text: sentiment, icon: sentimentIcon(for: sentiment), color: sentimentColor(for: sentiment))
            }

            // Urgency
            if let urgency = entry.effectiveUrgency {
                TagPill(text: urgency, icon: "exclamationmark.circle", color: .red)
            }

            // ActionType
            if let type = entry.effectiveActionType {
                TagPill(text: type, icon: "figure.walk", color: .green)
            }

            // Difficulty
            if let diff = entry.effectiveDifficulty {
                TagPill(text: diff, icon: "speedometer", color: .orange)
            }
        }
    }

    private func sentimentIcon(for sentiment: String) -> String {
        switch sentiment {
        case "积极": return "sun.max.fill"
        case "消极": return "cloud.rain.fill"
        default: return "cloud.fill"
        }
    }

    private func sentimentColor(for sentiment: String) -> Color {
        switch sentiment {
        case "积极": return .green
        case "消极": return .gray
        default: return .blue
        }
    }
}

struct TagPill: View {
    let text: String
    let icon: String
    let color: Color

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: icon)
                .font(.system(size: 10))
            Text(text)
                .font(.system(size: 12, weight: .medium))
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(color.opacity(0.2))
        .foregroundColor(color)
        .clipShape(Capsule())
    }
}
