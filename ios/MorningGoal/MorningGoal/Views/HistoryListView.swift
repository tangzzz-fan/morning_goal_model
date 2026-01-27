import CoreData
import Observation
import SwiftUI

struct HistoryListView: View {
    @Environment(\.managedObjectContext) private var context
    @FetchRequest(
        sortDescriptors: [NSSortDescriptor(keyPath: \GoalEntry.lastUpdated, ascending: false)],
        animation: .default
    ) private var entries: FetchedResults<GoalEntry>

    @State private var streakTrigger = false
    @State private var previousCount: Int = 0
    @State private var showSettings = false

    @State private var selectedEntryForCorrection: GoalEntry?

    // Celebration trigger from parent
    @Binding var shouldCelebrate: Bool

    var body: some View {
        ZStack {
            Color.Design.deepIndigo.ignoresSafeArea()

            if entries.isEmpty {
                EmptyStateView()
            } else {
                ScrollView {
                    LazyVStack(spacing: Spacing.md, pinnedViews: .sectionHeaders) {
                        // Streak 计数器在顶部
                        HStack {
                            Button(action: { showSettings = true }, label: {
                                Image(systemName: "gearshape")
                                    .foregroundColor(Color.Design.mutedGray)
                            })
                            Spacer()
                            StreakCounterView(trigger: $streakTrigger)
                                .padding(.trailing, Spacing.md)
                        }
                        .padding(.top, Spacing.sm)
                        .padding(.horizontal, Spacing.md)
                        ForEach(groupedEntries.keys.sorted(by: >), id: \.self) { date in
                            Section(header: SectionHeader(date: date)) {
                                ForEach(groupedEntries[date] ?? [], id: \.self) { entry in
                                    HistoryRow(entry: entry, onCorrect: {
                                        selectedEntryForCorrection = entry
                                    })
                                    .padding(.horizontal, Spacing.md)
                                }
                            }
                        }
                    }
                    .padding(.top, Spacing.sm)
                }
                .scrollIndicators(.hidden)
            }
        }
        .onAppear {
            previousCount = entries.count

            // 如果需要庆祝，触发简单更新
            if shouldCelebrate {
                streakTrigger.toggle()
                shouldCelebrate = false
            }
        }
        .onChange(of: entries.count) { _, newCount in
            if newCount > previousCount {
                streakTrigger.toggle()
            }
            previousCount = newCount
        }
        .sheet(isPresented: $showSettings) {
            SettingsView()
        }
        .sheet(item: $selectedEntryForCorrection) { entry in
            NavigationView {
                GoalDetailView(entry: entry)
                    .toolbar {
                        ToolbarItem(placement: .navigationBarLeading) {
                            Button(LocalizedStringKey("history_close")) {
                                selectedEntryForCorrection = nil
                            }
                        }
                    }
            }
        }
    }

    private var groupedEntries: [Date: [GoalEntry]] {
        let calendar = Calendar.current
        return Dictionary(grouping: entries) { entry in
            calendar.startOfDay(for: entry.lastUpdated)
        }
    }
}

struct SettingsView: View {
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        ZStack {
            Color.Design.deepIndigo.ignoresSafeArea()

            VStack(spacing: Spacing.lg) {
                Text("Settings")
                    .font(Typography.headline)
                    .foregroundColor(Color.Design.softWhite)
                    .padding(.top, Spacing.lg)

                List {
                    Section {
                        if let url1 = URL(string: "https://mcnrn1su375m.feishu.cn/wiki/WXKxwZ3l6iusE2k5PI8ceF50nOf") {
                            Link("Terms of Service", destination: url1)
                        }
                        if let url2 = URL(string: "https://mcnrn1su375m.feishu.cn/wiki/MX9Nw1u5uiFuaSk1aeCcI0Slnrd") {
                            Link("Privacy Policy", destination: url2)
                        }
                    } header: {
                        Text("Legal")
                            .foregroundColor(Color.Design.mutedGray)
                    }
                    .listRowBackground(Color.Design.darkIndigo.opacity(0.6))

                    Section {
                        HStack {
                            Text("Version")
                                .foregroundColor(Color.Design.softWhite)
                            Spacer()
                            Text(Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "1.0")
                                .foregroundColor(Color.Design.mutedGray)
                        }
                    } header: {
                        Text("About")
                            .foregroundColor(Color.Design.mutedGray)
                    }
                    .listRowBackground(Color.Design.darkIndigo.opacity(0.6))
                }
                .scrollContentBackground(.hidden)
                .foregroundColor(Color.Design.softWhite) // 设置 List 默认文本颜色

                Button(LocalizedStringKey("close")) {
                    dismiss()
                }
                .buttonStyle(SunriseGoldButtonStyle())
                .padding(.bottom, Spacing.xl)
            }
        }
    }
}

struct HistoryRow: View {
    let entry: GoalEntry
    let onCorrect: () -> Void
    @State private var isExpanded = false
    @State private var isGeneratingInsight = false
    @Environment(InsightModelManagerWrapper.self) var modelManager

    var body: some View {
        VStack(alignment: .leading, spacing: Spacing.sm) {
            HStack {
                // ... (existing code for header)
                // 日期和时间
                VStack(alignment: .leading, spacing: Spacing.xs) {
                    Text(formatDate(entry.lastUpdated))
                        .font(Typography.caption)
                        .foregroundColor(Color.Design.mutedGray)

                    Text(formatTime(entry.lastUpdated))
                        .font(Typography.caption)
                        .foregroundColor(Color.Design.mutedGray.opacity(0.7))
                }
                .frame(width: 60, alignment: .leading)

                // 内容预览
                VStack(alignment: .leading, spacing: Spacing.xs) {
                    Text(entry.goalText)
                        .font(Typography.body)
                        .foregroundColor(Color.Design.softWhite)
                        .lineLimit(isExpanded ? nil : 2) // Expand text if needed
                        .fixedSize(horizontal: false, vertical: true)
                }

                Spacer()

                // Expand Icon
                Image(systemName: "chevron.right")
                    .rotationEffect(.degrees(isExpanded ? 90 : 0))
                    .foregroundColor(Color.Design.mutedGray)
                    .font(.caption)
            }

            // Progressive Disclosure Content
            if isExpanded {
                Divider()
                    .background(Color.Design.mutedGray.opacity(0.3))
                    .padding(.vertical, Spacing.xs)

                // 1. Tags
                GoalTagsView(entry: entry)

                // 2. Saved Insight (if any)
                if isGeneratingInsight {
                    // Loading State
                    HStack(spacing: Spacing.sm) {
                        ProgressView()
                            .scaleEffect(0.7)
                        Text(LocalizedStringKey("history_generating_insight"))
                            .font(Typography.caption)
                            .foregroundColor(Color.Design.mutedGray)
                    }
                    .padding(.top, Spacing.xs)
                } else if let insightText = entry.insightText {
                    HStack(spacing: Spacing.sm) {
                        Image(systemName: "sparkles")
                            .foregroundColor(Color.Design.sunriseGold)
                        Text(insightText)
                            .font(Typography.caption)
                            .foregroundColor(Color.Design.sunriseGold)
                    }
                    .padding(.top, Spacing.xs)
                } else {
                    // Empty state (should not happen often as we trigger generation immediately)
                    Text(LocalizedStringKey("history_no_insight"))
                        .font(Typography.caption)
                        .foregroundColor(Color.Design.mutedGray.opacity(0.5))
                        .padding(.top, Spacing.xs)
                }

                // 3. User Correction Action
                Button(action: {
                    onCorrect()
                }) {
                    Text(LocalizedStringKey("history_wrong_classification"))
                        .font(.caption2)
                        .foregroundColor(Color.Design.mutedGray)
                        .underline()
                }
                .padding(.top, Spacing.sm)
            }
        }
        .padding(Spacing.md)
        .background(
            RoundedRectangle(cornerRadius: CornerRadius.md)
                .fill(Color.Design.darkIndigo.opacity(0.6))
                .overlay(
                    RoundedRectangle(cornerRadius: CornerRadius.md)
                        .stroke(Color.Design.sunriseGold.opacity(isExpanded ? 0.3 : 0.1), lineWidth: 1)
                )
        )
        // Handle Tap
        .contentShape(Rectangle()) // Make entire area tappable
        .onTapGesture {
            withAnimation(.spring()) {
                isExpanded.toggle()
            }
            // Trigger generation if needed
            if isExpanded {
                generateRetroactiveInsight()
            }
        }
    }

    private func generateRetroactiveInsight() {
        guard let context = entry.managedObjectContext else { return }
        isGeneratingInsight = true

        Task {
            // Ensure models are loaded
            if !modelManager.isInitialized {
                await modelManager.loadModels()
            }

            // 1. Perform Analysis (Serialized)
            // We force analysis to ensure tags are generated if missing or updated
            if let result = try? await modelManager.performSerially({
                try await modelManager.analyze(text: entry.goalText)
            }) {
                await MainActor.run {
                    // Update entry with analysis results
                    entry.analyzedAt = Date()
                    if let topic = result.topic { entry.category = topic.label
                        entry.categoryConfidence = topic.confidence
                    }
                    if let sentiment = result.sentiment {
                        entry.sentiment = sentiment.label
                        entry.sentimentScore = sentiment.label == "积极" ? 0.9 : (sentiment.label == "消极" ? -0.9 : 0.0)
                    }
                    if let urgency = result.urgency { entry.urgency = urgency.label
                        entry.urgencyConfidence = urgency.confidence
                    }
                    if let timeFrame = result.timeFrame { entry.timeFrame = timeFrame.label
                        entry.timeFrameConfidence = timeFrame.confidence
                    }
                    if let actionType = result.actionType { entry.actionType = actionType.label
                        entry.actionTypeConfidence = actionType.confidence
                    }
                    if let difficulty = result.difficulty { entry.difficulty = difficulty.label
                        entry.difficultyConfidence = difficulty.confidence
                    }
                    if let specificity = result.specificity { entry.specificity = specificity.label
                        entry.specificityConfidence = specificity.confidence
                    }

                    // Save analysis results immediately
                    try? context.save()
                }
            }

            // 2. Generate Insight Whisper
            let engine = InsightEngine(viewContext: context)
            if let insight = await engine.generateDailyInsight(for: entry) {
                await MainActor.run {
                    entry.insightText = insight.description
                    entry.insightShownAt = Date()

                    do {
                        try context.save()
                    } catch {
                        print("Failed to save retroactive insight: \(error)")
                    }

                    withAnimation {
                        isGeneratingInsight = false
                    }
                }
            } else {
                await MainActor.run {
                    isGeneratingInsight = false
                }
            }
        }
    }

    private func formatDate(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "MM/dd"
        return formatter.string(from: date)
    }

    private func formatTime(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm"
        return formatter.string(from: date)
    }
}

struct TagView: View {
    let text: String

    var body: some View {
        Text(text)
            .font(Typography.caption)
            .foregroundColor(Color.Design.sunriseGold)
            .padding(.horizontal, Spacing.sm)
            .padding(.vertical, Spacing.xs)
            .background(
                Capsule()
                    .fill(Color.Design.sunriseGold.opacity(0.1))
                    .overlay(
                        Capsule()
                            .stroke(Color.Design.sunriseGold.opacity(0.3), lineWidth: 1)
                    )
            )
    }
}

struct SectionHeader: View {
    let date: Date

    var body: some View {
        HStack {
            Text(formatSectionDate(date))
                .font(Typography.caption)
                .foregroundColor(Color.Design.mutedGray)
                .padding(.horizontal, Spacing.md)
                .padding(.vertical, Spacing.sm)
                .background(
                    Capsule()
                        .fill(Color.Design.deepIndigo)
                        .overlay(
                            Capsule()
                                .stroke(Color.Design.sunriseGold.opacity(0.2), lineWidth: 1)
                        )
                )

            Spacer()
        }
        .padding(.horizontal, Spacing.md)
        .padding(.vertical, Spacing.sm)
        .background(Color.Design.deepIndigo)
    }

    private func formatSectionDate(_ date: Date) -> String {
        let calendar = Calendar.current
        if calendar.isDateInToday(date) {
            return NSLocalizedString("history_today", comment: "")
        } else if calendar.isDateInYesterday(date) {
            return NSLocalizedString("history_yesterday", comment: "")
        } else {
            let formatter = DateFormatter()
            formatter.dateFormat = "MM月dd日"
            return formatter.string(from: date)
        }
    }
}

struct EmptyStateView: View {
    var body: some View {
        VStack(spacing: Spacing.lg) {
            Image(systemName: "book.closed")
                .font(.system(size: 64))
                .foregroundColor(Color.Design.sunriseGold.opacity(0.6))

            Text(LocalizedStringKey("history_empty_title"))
                .font(Typography.headline)
                .foregroundColor(Color.Design.softWhite)

            Text(LocalizedStringKey("history_empty_message"))
                .font(Typography.body)
                .foregroundColor(Color.Design.mutedGray)
                .multilineTextAlignment(.center)

            // 42% Theory Hint
            Text(LocalizedStringKey("onboarding_page1_subtitle"))
                .font(Typography.caption)
                .foregroundColor(Color.Design.sunriseGold.opacity(0.8))
                .multilineTextAlignment(.center)
                .padding(.horizontal, Spacing.xl)
                .padding(.top, Spacing.md)
        }
        .padding()
    }
}

// 流式布局
struct FlowLayout: Layout {
    var spacing: CGFloat = 8

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let result = FlowResult(
            in: proposal.replacingUnspecifiedDimensions().width,
            subviews: subviews,
            spacing: spacing
        )
        return result.size
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let result = FlowResult(
            in: bounds.width,
            subviews: subviews,
            spacing: spacing
        )

        for (index, subview) in subviews.enumerated() {
            subview.place(
                at: CGPoint(
                    x: bounds.minX + result.positions[index].x,
                    y: bounds.minY + result.positions[index].y
                ),
                proposal: .unspecified
            )
        }
    }

    struct FlowResult {
        var size: CGSize = .zero
        var positions: [CGPoint] = []

        init(in maxWidth: CGFloat, subviews: Subviews, spacing: CGFloat) {
            var x: CGFloat = 0
            var y: CGFloat = 0
            var lineHeight: CGFloat = 0

            for subview in subviews {
                let size = subview.sizeThatFits(.unspecified)

                if x + size.width > maxWidth && x > 0 {
                    x = 0
                    y += lineHeight + spacing
                    lineHeight = 0
                }

                positions.append(CGPoint(x: x, y: y))
                x += size.width + spacing
                lineHeight = max(lineHeight, size.height)

                self.size.width = max(self.size.width, x - spacing)
                self.size.height = max(self.size.height, y + lineHeight)
            }
        }
    }
}
