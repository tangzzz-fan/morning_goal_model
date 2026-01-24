import CoreData
import SwiftUI

struct StreakCounterView: View {
    @Environment(\.managedObjectContext) private var context
    @State private var streakDays: Int = 0
    @Binding var trigger: Bool

    private var hasStreak: Bool { streakDays > 0 }

    func refresh(animated: Bool) {
        streakDays = StreakService.count(in: context)
    }

    var body: some View {
        HStack(spacing: Spacing.sm) {
            // 火焰图标
            if hasStreak {
                Image(systemName: "flame.fill")
                    .font(.title3)
                    .foregroundColor(Color.Design.sunriseGold)
            }

            // Streak 数字
            Text("\(streakDays)")
                .font(Typography.streakNumber)
                .foregroundColor(Color.Design.sunriseGold)

            // 天数标签
            Text(LocalizedStringKey("days"))
                .font(Typography.caption)
                .foregroundColor(Color.Design.mutedGray)
        }
        .padding(.horizontal, Spacing.md)
        .padding(.vertical, Spacing.sm)
        .background(
            Capsule()
                .fill(Color.Design.darkIndigo.opacity(0.8))
                .overlay(
                    Capsule()
                        .stroke(Color.Design.sunriseGold.opacity(0.3), lineWidth: 1)
                )
        )
        .onAppear { refresh(animated: false) }
        .onChange(of: trigger) { _, _ in refresh(animated: true) }
    }
}
