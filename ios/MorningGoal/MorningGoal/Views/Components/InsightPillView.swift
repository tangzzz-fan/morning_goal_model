import SwiftUI

struct InsightPillView: View {
    let insight: Insight

    var body: some View {
        HStack(spacing: Spacing.sm) {
            Image(systemName: insight.iconName)
                .font(.system(size: 12))
                .foregroundColor(Color.Design.deepIndigo)

            Text(insight.description)
                .font(Typography.caption)
                .foregroundColor(Color.Design.deepIndigo)
                .lineLimit(1)
        }
        .padding(.horizontal, Spacing.md)
        .padding(.vertical, Spacing.xs)
        .background(
            Capsule()
                .fill(Color.Design.sunriseGold.opacity(0.9))
                .shadow(color: Color.Design.sunriseGold.opacity(0.3), radius: 4, x: 0, y: 2)
        )
        .transition(.scale.combined(with: .opacity))
    }
}
