import CoreData
import SwiftUI

struct GoalDetailView: View {
    @Environment(\.managedObjectContext) private var context
    @ObservedObject var entry: GoalEntry
    @Environment(\.dismiss) private var dismiss

    // Labels from InsightModelManager (duplicated here for UI independence, or we could inject them)
    // Ideally we should get these from a source of truth, but for now hardcoding to match Manager is fine.

    let topicLabels = [
        "工作", "健康", "家庭", "个人发展", "理财", "社交", "家务", "学习",
        "睡眠", "饮食", "心态", "娱乐", "出行", "职业发展", "沟通", "育儿"
    ]
    let sentimentLabels = ["积极", "中性", "消极"]
    let urgencyLabels = ["低", "中", "高"]
    let timeframeLabels = ["今天", "本周", "本月", "长期"]
    let actiontypeLabels = ["学习", "运动", "工作", "生活", "社交"]
    let difficultyLabels = ["简单", "中等", "困难"]
    let specificityLabels = ["模糊", "一般", "具体"]

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Spacing.lg) {
                // 1. Header with Goal Text
                VStack(alignment: .leading, spacing: Spacing.xs) {
                    Text("GOAL")
                        .font(Typography.caption)
                        .foregroundColor(Color.Design.mutedGray)

                    Text(entry.goalText)
                        .font(Typography.headline)
                        .foregroundColor(Color.Design.softWhite)
                        .padding()
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color.Design.inputBackground)
                        .cornerRadius(CornerRadius.md)
                }

                Divider().background(Color.Design.mutedGray.opacity(0.3))

                Text("Correction Help Improve AI")
                    .font(Typography.caption)
                    .foregroundColor(Color.Design.mutedGray)
                    .padding(.bottom, -Spacing.md)

                // 2. Correction Sections for all 7 Dimensions
                VStack(spacing: Spacing.md) {
                    Group {
                        CorrectionSection(
                            title: "Topic (主题)",
                            currentValue: entry.effectiveCategory,
                            confidence: entry.categoryConfidence,
                            options: topicLabels,
                            color: .blue
                        ) { newValue in
                            updateEntry {
                                entry.categoryUserCorrected = newValue
                            }
                        }

                        CorrectionSection(
                            title: "Sentiment (情感)",
                            currentValue: entry.effectiveSentiment,
                            confidence: 0.0,
                            options: sentimentLabels,
                            color: sentimentColor(entry.effectiveSentiment)
                        ) { newValue in
                            updateEntry {
                                entry.sentimentUserCorrected = newValue
                            }
                        }

                        CorrectionSection(
                            title: "Urgency (紧急度)",
                            currentValue: entry.urgencyUserCorrected ?? entry.urgency,
                            confidence: entry.urgencyConfidence,
                            options: urgencyLabels,
                            color: .orange
                        ) { newValue in
                            updateEntry {
                                entry.urgencyUserCorrected = newValue
                            }
                        }

                        CorrectionSection(
                            title: "Time Frame (时间)",
                            currentValue: entry.timeFrameUserCorrected ?? entry.timeFrame,
                            confidence: entry.timeFrameConfidence,
                            options: timeframeLabels,
                            color: .purple
                        ) { newValue in
                            updateEntry {
                                entry.timeFrameUserCorrected = newValue
                            }
                        }
                    }

                    Group {
                        CorrectionSection(
                            title: "Action Type (行动)",
                            currentValue: entry.actionTypeUserCorrected ?? entry.actionType,
                            confidence: entry.actionTypeConfidence,
                            options: actiontypeLabels,
                            color: .green
                        ) { newValue in
                            updateEntry {
                                entry.actionTypeUserCorrected = newValue
                            }
                        }

                        CorrectionSection(
                            title: "Difficulty (难度)",
                            currentValue: entry.difficultyUserCorrected ?? entry.difficulty,
                            confidence: entry.difficultyConfidence,
                            options: difficultyLabels,
                            color: .red
                        ) { newValue in
                            updateEntry {
                                entry.difficultyUserCorrected = newValue
                            }
                        }

                        CorrectionSection(
                            title: "Specificity (明确度)",
                            currentValue: entry.specificityUserCorrected ?? entry.specificity,
                            confidence: entry.specificityConfidence,
                            options: specificityLabels,
                            color: .teal
                        ) { newValue in
                            updateEntry {
                                entry.specificityUserCorrected = newValue
                            }
                        }
                    }
                }
                .padding(.bottom, Spacing.xl)

                // Footer removed to eliminate "timer"
            }
            .padding(Spacing.md)
        }
        .background(Color.Design.deepIndigo.ignoresSafeArea())
        .navigationTitle("Detail & Correct")
        .navigationBarTitleDisplayMode(.inline)
    }

    private func updateEntry(_ changes: () -> Void) {
        changes()
        entry.correctedAt = Date()
        entry.isTrainingSample = true

        do {
            try context.save()
            // Trigger feedback logic
            ModelUpdateScheduler.shared.scheduleModelUpdate()
            print("✅ Correction saved and model update scheduled.")
        } catch {
            print("❌ Failed to save correction: \(error)")
        }
    }

    private func sentimentColor(_ sentiment: String?) -> Color {
        switch sentiment {
        case "积极": return .green
        case "消极": return .red
        default: return .gray
        }
    }
}

// MARK: - Reusable Correction Section

struct CorrectionSection: View {
    let title: String
    let currentValue: String?
    let confidence: Double
    let options: [String]
    let color: Color
    let onCorrect: (String) -> Void

    @State private var isEditing = false

    var body: some View {
        VStack(alignment: .leading, spacing: Spacing.xs) {
            HStack {
                Text(title)
                    .font(Typography.caption)
                    .foregroundColor(Color.Design.mutedGray)

                Spacer()

                if confidence > 0 {
                    Text("\(Int(confidence * 100))%")
                        .font(Typography.caption)
                        .foregroundColor(Color.Design.mutedGray.opacity(0.7))
                }
            }

            HStack {
                if let value = currentValue {
                    Text(value)
                        .font(Typography.body)
                        .foregroundColor(.white)
                        .padding(.horizontal, Spacing.sm)
                        .padding(.vertical, 4)
                        .background(color.opacity(0.8))
                        .cornerRadius(CornerRadius.sm)
                } else {
                    Text("Unknown")
                        .font(Typography.body)
                        .foregroundColor(Color.Design.mutedGray)
                }

                Spacer()

                Button(action: { isEditing.toggle() }) {
                    Text("Modify")
                        .font(Typography.caption)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(Color.Design.darkIndigo)
                        .foregroundColor(Color.Design.sunriseGold)
                        .cornerRadius(CornerRadius.sm)
                        .overlay(
                            RoundedRectangle(cornerRadius: CornerRadius.sm)
                                .stroke(Color.Design.sunriseGold.opacity(0.3), lineWidth: 1)
                        )
                }
            }

            if isEditing {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: Spacing.sm) {
                        ForEach(options, id: \.self) { option in
                            Button(action: {
                                onCorrect(option)
                                withAnimation {
                                    isEditing = false
                                }
                            }) {
                                Text(option)
                                    .font(Typography.caption)
                                    .foregroundColor(currentValue == option ? .white : Color.Design.mutedGray)
                                    .padding(.horizontal, 12)
                                    .padding(.vertical, 6)
                                    .background(
                                        currentValue == option ? color : Color.Design.inputBackground
                                    )
                                    .cornerRadius(CornerRadius.sm)
                                    .overlay(
                                        RoundedRectangle(cornerRadius: CornerRadius.sm)
                                            .stroke(currentValue == option ? Color.white : Color.clear, lineWidth: 1)
                                    )
                            }
                        }
                    }
                    .padding(.top, Spacing.xs)
                    .padding(.bottom, 4)
                }
            }
        }
        .padding()
        .background(Color.Design.cardBackground)
        .cornerRadius(CornerRadius.md)
    }
}
