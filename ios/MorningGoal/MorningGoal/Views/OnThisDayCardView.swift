import SwiftUI

struct OnThisDayCardView: View {
    let entry: GoalEntry
    @Environment(\.dismiss) private var dismiss
    @State private var appeared = false

    var timeLabel: String {
        let calendar = Calendar.current
        let today = calendar.startOfDay(for: Date())
        let entryDate = calendar.startOfDay(for: entry.lastUpdated)

        let components = calendar.dateComponents([.year, .month, .day], from: entryDate, to: today)

        if let year = components.year, year >= 1 {
            return String(format: NSLocalizedString("on_this_day_title_years", comment: ""), year)
        } else if let month = components.month, month >= 1 {
            return NSLocalizedString("on_this_day_title_month", comment: "")
        } else if let day = components.day, day >= 7 {
            return NSLocalizedString("on_this_day_title_week", comment: "")
        }

        return String(format: NSLocalizedString("on_this_day_title_years", comment: ""), 1) // Fallback
    }

    var body: some View {
        ZStack {
            // 半透明背景
            Color.black.opacity(0.9)
                .ignoresSafeArea()
                .onTapGesture {
                    dismissCard()
                }

            // 卡片内容
            VStack(spacing: Spacing.lg) {
                // 顶部装饰
                Image(systemName: "sparkles")
                    .font(.system(size: 32))
                    .foregroundColor(Color.Design.sunriseGold)
                    .rotationEffect(.degrees(appeared ? 0 : -180))
                    .scaleEffect(appeared ? 1 : 0.5)
                    .opacity(appeared ? 1 : 0)

                // 标题
                Text(timeLabel)
                    .font(Typography.title)
                    .foregroundColor(Color.Design.sunriseGold)
                    .multilineTextAlignment(.center)

                // 分隔线
                Rectangle()
                    .fill(Color.Design.sunriseGold.opacity(0.3))
                    .frame(height: 1)
                    .frame(maxWidth: 200)

                // 历史目标内容
                ScrollView {
                    Text(entry.goalText)
                        .font(Typography.body)
                        .foregroundColor(Color.Design.softWhite)
                        .lineSpacing(6)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, Spacing.lg)
                }
                .frame(maxHeight: 200)

                // 日期
                Text(formatDate(entry.lastUpdated))
                    .font(Typography.caption)
                    .foregroundColor(Color.Design.mutedGray)

                // 关闭按钮
                Button(action: dismissCard) {
                    Text(LocalizedStringKey("close"))
                        .font(Typography.body)
                        .foregroundColor(Color.Design.sunriseGold)
                        .padding(.horizontal, Spacing.xl)
                        .padding(.vertical, Spacing.md)
                        .background(
                            Capsule()
                                .fill(Color.Design.darkIndigo.opacity(0.8))
                                .overlay(
                                    Capsule()
                                        .stroke(Color.Design.sunriseGold.opacity(0.3), lineWidth: 1)
                                )
                        )
                }
            }
            .padding(Spacing.xl)
            .background(
                RoundedRectangle(cornerRadius: CornerRadius.lg)
                    .fill(Color.Design.deepIndigo)
                    .overlay(
                        RoundedRectangle(cornerRadius: CornerRadius.lg)
                            .stroke(Color.Design.sunriseGold.opacity(0.3), lineWidth: 2)
                    )
                    .shadow(color: Color.Design.sunriseGold.opacity(0.2), radius: 20, x: 0, y: 10)
            )
            .padding(.horizontal, Spacing.xl)
            .scaleEffect(appeared ? 1 : 0.8)
            .opacity(appeared ? 1 : 0)
        }
        .onAppear {
            withAnimation(.spring(response: 0.6, dampingFraction: 0.8)) {
                appeared = true
            }
        }
    }

    private func dismissCard() {
        withAnimation(.easeOut(duration: 0.3)) {
            appeared = false
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
            dismiss()
        }
    }

    private func formatDate(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy年MM月dd日"
        return formatter.string(from: date)
    }
}
