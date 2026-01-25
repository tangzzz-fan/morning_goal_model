import CoreData
import SwiftUI

struct InsightEngineDebugView: View {
    @State private var selectedScenario: Scenario = .balanced
    @State private var insights: [Insight] = []
    @State private var mockEntries: [GoalEntry] = []

    // 独立的内存 Core Data Stack 用于生成测试数据
    private let container: NSPersistentContainer = {
        let container = NSPersistentContainer(name: "MorningGoal")
        let description = NSPersistentStoreDescription()
        description.type = NSInMemoryStoreType
        container.persistentStoreDescriptions = [description]
        container.loadPersistentStores { _, error in
            if let error = error {
                print("Failed to load in-memory store: \(error)")
            }
        }
        return container
    }()

    enum Scenario: String, CaseIterable, Identifiable {
        case balanced = "均衡发展"
        case overworked = "工作狂模式"
        case decliningMotivation = "动力下降"
        case highUrgency = "焦虑状态"
        case consistentLearner = "持续学习"

        var id: String { rawValue }
    }

    var body: some View {
        List {
            Section("测试场景") {
                Picker("选择场景", selection: $selectedScenario) {
                    ForEach(Scenario.allCases) { scenario in
                        Text(scenario.rawValue).tag(scenario)
                    }
                }
                .pickerStyle(.menu)

                Button("生成数据并分析") {
                    runScenario()
                }
                .buttonStyle(.borderedProminent)
            }

            if !insights.isEmpty {
                Section("生成的洞察 (\(insights.count))") {
                    ForEach(insights.indices, id: \.self) { index in
                        let insight = insights[index]
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                Text(insightTitle(for: insight))
                                    .font(.headline)
                                Spacer()
                                priorityBadge(for: insight.priority)
                            }
                            Text(insightDescription(for: insight))
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                        }
                        .padding(.vertical, 4)
                    }
                }
            }

            Section("模拟数据概览 (\(mockEntries.count)条)") {
                if mockEntries.isEmpty {
                    Text("暂无数据")
                        .foregroundColor(.secondary)
                } else {
                    ForEach(mockEntries.prefix(5)) { entry in
                        VStack(alignment: .leading) {
                            Text(entry.goalText)
                                .font(.caption)
                                .lineLimit(1)
                            HStack {
                                Text(entry.dateString)
                                Text(entry.effectiveCategory ?? "-")
                                Text(entry.effectiveSentiment ?? "-")
                            }
                            .font(.caption2)
                            .foregroundColor(.secondary)
                        }
                    }
                    if mockEntries.count > 5 {
                        Text("... 还有 \(mockEntries.count - 5) 条")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
            }
        }
        .navigationTitle("洞察引擎实验室")
    }

    // MARK: - Logic

    private func runScenario() {
        // 1. 清除旧数据
        mockEntries = []
        insights = []

        let context = container.viewContext
        context.reset()

        // 2. 生成新数据
        let entries = generateScenarioData(in: context)
        self.mockEntries = entries

        // 3. 运行分析
        let engine = InsightEngine(context: context)
        self.insights = engine.generateInsights(from: entries, topK: 10)
    }

    private func generateScenarioData(in context: NSManagedObjectContext) -> [GoalEntry] {
        switch selectedScenario {
        case .balanced: return generateBalancedData(in: context)
        case .overworked: return generateOverworkedData(in: context)
        case .decliningMotivation: return generateDecliningData(in: context)
        case .highUrgency: return generateHighUrgencyData(in: context)
        case .consistentLearner: return generateConsistentLearnerData(in: context)
        }
    }

    private func generateBalancedData(in context: NSManagedObjectContext) -> [GoalEntry] {
        var entries: [GoalEntry] = []
        let calendar = Calendar.current
        let today = Date()

        for i in 0 ..< 14 {
            guard let date = calendar.date(byAdding: .day, value: -i, to: today) else { continue }
            let categories = ["工作", "健康", "学习", "社交", "娱乐"]
            let sentiments = ["积极", "中性"]

            let count = Int.random(in: 1 ... 2)
            for _ in 0 ..< count {
                let entry = createEntry(in: context, date: date)
                entry.category = categories.randomElement()
                entry.sentiment = sentiments.randomElement()
                entry.urgency = "中"
                entries.append(entry)
            }
        }
        return entries
    }

    private func generateOverworkedData(in context: NSManagedObjectContext) -> [GoalEntry] {
        var entries: [GoalEntry] = []
        let calendar = Calendar.current
        let today = Date()

        for i in 0 ..< 14 {
            guard let date = calendar.date(byAdding: .day, value: -i, to: today) else { continue }
            let count = Int.random(in: 2 ... 4)
            for _ in 0 ..< count {
                let entry = createEntry(in: context, date: date)
                if Double.random(in: 0 ... 1) < 0.8 {
                    entry.category = "工作"
                    entry.urgency = "高"
                    entry.sentiment = ["消极", "中性"].randomElement()
                } else {
                    entry.category = "休息"
                    entry.urgency = "低"
                }
                entries.append(entry)
            }
        }
        return entries
    }

    private func generateDecliningData(in context: NSManagedObjectContext) -> [GoalEntry] {
        var entries: [GoalEntry] = []
        let calendar = Calendar.current
        let today = Date()

        for i in 0 ..< 20 {
            guard let date = calendar.date(byAdding: .day, value: -i, to: today) else { continue }

            let isRecent = i < 7
            let count = isRecent ? Int.random(in: 0 ... 1) : Int.random(in: 2 ... 3)

            if count > 0 {
                for _ in 0 ..< count {
                    let entry = createEntry(in: context, date: date)
                    entry.sentiment = isRecent ? "消极" : "积极"
                    entry.category = "学习"
                    entries.append(entry)
                }
            }
        }
        return entries
    }

    private func generateHighUrgencyData(in context: NSManagedObjectContext) -> [GoalEntry] {
        var entries: [GoalEntry] = []
        let calendar = Calendar.current
        let today = Date()

        for i in 0 ..< 10 {
            guard let date = calendar.date(byAdding: .day, value: -i, to: today) else { continue }
            let entry = createEntry(in: context, date: date)
            entry.urgency = "高"
            entry.category = "工作"
            entry.goalText = "紧急任务 \(i)"
            entries.append(entry)
        }
        return entries
    }

    private func generateConsistentLearnerData(in context: NSManagedObjectContext) -> [GoalEntry] {
        var entries: [GoalEntry] = []
        let calendar = Calendar.current
        let today = Date()

        for i in 0 ..< 14 {
            guard let date = calendar.date(byAdding: .day, value: -i, to: today) else { continue }
            let entry = createEntry(in: context, date: date)
            entry.category = "学习"
            entry.sentiment = "积极"
            entry.goalText = "学习 Swift 第 \(i) 章"
            entries.append(entry)
        }
        return entries
    }

    private func createEntry(in context: NSManagedObjectContext, date: Date) -> GoalEntry {
        let entry = GoalEntry(context: context)
        entry.dateString = GoalEntry.dateStringFrom(date)
        entry.lastUpdated = date
        entry.goalText = "Mock Goal \(UUID().uuidString.prefix(4))"

        // 默认值，会被覆盖
        entry.category = "其他"
        entry.categoryConfidence = 0.9
        entry.sentiment = "中性"
        entry.sentimentScore = 0.8
        entry.urgency = "中"
        entry.urgencyConfidence = 0.8
        entry.timeFrame = "今天"
        entry.timeFrameConfidence = 0.9
        entry.actionType = "生活"
        entry.actionTypeConfidence = 0.8
        entry.difficulty = "中等"
        entry.difficultyConfidence = 0.8
        entry.specificity = "一般"
        entry.specificityConfidence = 0.8

        // 设置用户修正值以确保 effective 属性工作
        entry.userCorrectedCategory = nil

        return entry
    }

    // MARK: - Helpers

    private func insightTitle(for insight: Insight) -> String {
        switch insight {
        case .categoryDistribution: return "主要关注点"
        case .streakBooster: return "习惯养成建议"
        case .consistencyPattern: return "坚持模式"
        case .sentimentTrend: return "情绪趋势"
        case .balance: return "生活平衡"
        case .volumeTrend: return "动力变化"
        case .weekdayPattern: return "黄金时间"
        case .encouragement: return "里程碑鼓励"
        case .comparison: return "周对比"
        case .recurringGoal: return "重复目标"
        case .achievability: return "可达成性分析"
        }
    }

    private func insightDescription(for insight: Insight) -> String {
        switch insight {
        case let .categoryDistribution(top, pct):
            return "你 \(pct)% 的目标都集中在 \(top) 领域"
        case let .streakBooster(cat, imp):
            return "如果在 \(cat) 上再坚持一下，完成率可提升 \(imp)%"
        case let .consistencyPattern(cat, days):
            return "你已经在 \(cat) 领域连续坚持了 \(days) 天"
        case let .sentimentTrend(dir):
            return "最近你的情绪呈现 \(dir) 趋势"
        case let .balance(main, pct, sugg):
            return "\(main) 占比 \(pct)%，建议适当增加 \(sugg) 相关活动"
        case let .volumeTrend(dir, change):
            return "目标数量较上周 \(dir) \(change)%"
        case let .weekdayPattern(busy, _, quiet):
            return "\(busy) 是你最忙碌的时候，\(quiet) 相对轻松"
        case let .encouragement(streak, msg):
            return "连续记录 \(streak) 天！\(msg)"
        case let .comparison(thisWeek, lastWeek, _):
            return "本周 \(thisWeek) 个目标，上周 \(lastWeek) 个"
        case let .recurringGoal(text, count, _):
            return "目标 '\(text)' 重复出现了 \(count) 次"
        case let .achievability(level, sugg):
            return "目标整体难度 \(level)，建议 \(sugg)"
        }
    }

    private func priorityBadge(for priority: InsightPriority) -> some View {
        Text(priorityLabel(priority))
            .font(.caption2)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(priorityColor(priority).opacity(0.2))
            .foregroundColor(priorityColor(priority))
            .cornerRadius(4)
    }

    private func priorityLabel(_ priority: InsightPriority) -> String {
        switch priority {
        case .high: return "高优先级"
        case .medium: return "中优先级"
        case .low: return "低优先级"
        }
    }

    private func priorityColor(_ priority: InsightPriority) -> Color {
        switch priority {
        case .high: return Color.Design.accentPink
        case .medium: return Color.Design.accentCyan
        case .low: return Color.Design.mutedGray
        }
    }
}
