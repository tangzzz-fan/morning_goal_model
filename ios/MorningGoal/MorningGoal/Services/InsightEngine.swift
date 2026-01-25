import CoreData
import Foundation

// MARK: - 洞察优先级

enum InsightPriority: Int, Comparable {
    case low = 0
    case medium = 1
    case high = 2

    static func < (lhs: InsightPriority, rhs: InsightPriority) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}

// MARK: - 洞察类型

enum Insight: Equatable {
    // 原有类型
    case categoryDistribution(topCategory: String, percentage: Int)
    case streakBooster(category: String, improvement: Int)
    case consistencyPattern(category: String, days: Int)
    case sentimentTrend(direction: String)

    // 新增类型 - Phase 5.1
    case balance(mainType: String, percentage: Int, suggestedType: String)
    case volumeTrend(direction: String, changePercent: Int)
    case weekdayPattern(busyDay: String, busyCount: Int, quietDay: String)
    case encouragement(streakDays: Int, message: String)
    case comparison(thisWeek: Int, lastWeek: Int, changePercent: Int)

    // 新增类型 - Phase 5.2 补充
    case recurringGoal(goalText: String, count: Int, category: String?)
    case achievability(level: String, suggestion: String)

    var id: String {
        switch self {
        case .categoryDistribution: return "category_dist"
        case .streakBooster: return "streak_boost"
        case .consistencyPattern: return "consistency"
        case .sentimentTrend: return "sentiment"
        case .balance: return "balance"
        case .volumeTrend: return "volume_trend"
        case .weekdayPattern: return "weekday_pattern"
        case .encouragement: return "encouragement"
        case .comparison: return "comparison"
        case .recurringGoal: return "recurring_goal"
        case .achievability: return "achievability"
        }
    }

    /// 洞察优先级
    var priority: InsightPriority {
        switch self {
        case .encouragement: return .high
        case .streakBooster, .balance, .achievability: return .high
        case .sentimentTrend, .volumeTrend, .comparison, .recurringGoal: return .medium
        case .categoryDistribution, .consistencyPattern, .weekdayPattern: return .low
        }
    }
}

// MARK: - 洞察引擎

class InsightEngine {
    private let context: NSManagedObjectContext

    init(context: NSManagedObjectContext) {
        self.context = context
    }

    // MARK: - 生成洞察

    /// 生成洞察（带Top-K优先级排序）
    /// - Parameters:
    ///   - period: 分析周期（天数）
    ///   - topK: 返回的最大洞察数量
    /// - Returns: 按优先级排序的洞察数组
    func generateInsights(for period: Int = 30, topK: Int = 5) -> [Insight] {
        // 获取最近N天的数据
        let entries = fetchRecentEntries(days: period)
        return generateInsights(from: entries, topK: topK)
    }

    /// 根据给定的目标条目生成洞察
    /// - Parameters:
    ///   - entries: 目标条目数组
    ///   - topK: 返回的最大洞察数量
    /// - Returns: 按优先级排序的洞察数组
    func generateInsights(from entries: [GoalEntry], topK: Int = 5) -> [Insight] {
        var insights: [Insight] = []

        guard entries.count >= 7 else {
            print("⏭️ 数据不足（<7天），跳过洞察生成")
            return insights
        }

        print("📊 生成洞察：基于 \(entries.count) 条数据")

        // 原有洞察
        if let topCategoryInsight = analyzeTopCategory(entries) {
            insights.append(topCategoryInsight)
        }

        if let streakInsight = analyzeStreakCorrelation(entries) {
            insights.append(streakInsight)
        }

        if let consistencyInsight = analyzeConsistency(entries) {
            insights.append(consistencyInsight)
        }

        if let sentimentInsight = analyzeSentimentTrend(entries) {
            insights.append(sentimentInsight)
        }

        // 新增洞察 (Phase 5.2)
        if let balanceInsight = analyzeBalance(entries) {
            insights.append(balanceInsight)
        }

        if let weekdayInsight = analyzeWeekdayPattern(entries) {
            insights.append(weekdayInsight)
        }

        if let volumeInsight = analyzeVolumeTrend(entries) {
            insights.append(volumeInsight)
        }

        if let encouragementInsight = generateEncouragement(entries) {
            insights.append(encouragementInsight)
        }

        if let comparisonInsight = analyzeWeekComparison(entries) {
            insights.append(comparisonInsight)
        }

        // Phase 5.2 补充分析
        if let recurringInsight = findRecurringGoals(entries) {
            insights.append(recurringInsight)
        }

        if let achievabilityInsight = analyzeAchievability(entries) {
            insights.append(achievabilityInsight)
        }

        // 按优先级排序（高优先级在前）
        let sortedInsights = insights.sorted { $0.priority > $1.priority }

        // Top-K 筛选
        let topInsights = Array(sortedInsights.prefix(topK))

        print("✅ 生成了 \(insights.count) 条洞察，返回 Top-\(topK): \(topInsights.count) 条")

        return topInsights
    }

    // MARK: - 分析方法

    private func analyzeTopCategory(_ entries: [GoalEntry]) -> Insight? {
        guard !entries.isEmpty else { return nil }

        // 统计每个类别的出现次数
        let categoryGroups = Dictionary(grouping: entries) {
            $0.effectiveCategory ?? "未分类"
        }

        guard let topCategory = categoryGroups.max(by: { $0.value.count < $1.value.count }),
              !topCategory.value.isEmpty
        else {
            return nil
        }

        let percentage = Int(Double(topCategory.value.count) / Double(entries.count) * 100)

        return .categoryDistribution(topCategory: topCategory.key, percentage: percentage)
    }

    private func analyzeStreakCorrelation(_ entries: [GoalEntry]) -> Insight? {
        // 按日期排序
        let sorted = entries.sorted { $0.dateString < $1.dateString }

        guard sorted.count > 7 else { return nil }

        // 计算每个类别对Streak的影响
        var categoryStreakImpact: [String: (maintained: Int, total: Int)] = [:]

        for (index, entry) in sorted.enumerated() where index < sorted.count - 1 {
            guard let category = entry.effectiveCategory else { continue }

            let nextEntry = sorted[index + 1]
            let daysDiff = daysBetween(entry.dateString, nextEntry.dateString)

            // 如果连续记录（相邻天），贡献+1；否则中断
            let isMaintained = daysDiff == 1

            var impact = categoryStreakImpact[category] ?? (maintained: 0, total: 0)
            if isMaintained {
                impact.maintained += 1
            }
            impact.total += 1
            categoryStreakImpact[category] = impact
        }

        // 计算每个类别的维持率
        var categoryRates: [String: Double] = [:]
        for (category, impact) in categoryStreakImpact {
            guard impact.total > 0 else { continue }
            let rate = Double(impact.maintained) / Double(impact.total)
            categoryRates[category] = rate
        }

        // 找到维持率最高的类别
        guard let bestCategory = categoryRates.max(by: { $0.value < $1.value }),
              bestCategory.value > 0.5, // 至少50%维持率
              let impact = categoryStreakImpact[bestCategory.key],
              impact.total >= 3
        else { // 至少3次记录
            return nil
        }

        let improvement = Int((bestCategory.value - 0.5) * 200) // 转换为相对改善百分比

        return .streakBooster(category: bestCategory.key, improvement: max(improvement, 10))
    }

    private func analyzeConsistency(_ entries: [GoalEntry]) -> Insight? {
        let categoryGroups = Dictionary(grouping: entries) {
            $0.effectiveCategory ?? "未分类"
        }

        // 找到出现最频繁的类别
        guard let consistentCategory = categoryGroups.max(by: { $0.value.count < $1.value.count }),
              consistentCategory.value.count >= 7
        else {
            return nil
        }

        return .consistencyPattern(
            category: consistentCategory.key,
            days: consistentCategory.value.count
        )
    }

    private func analyzeSentimentTrend(_ entries: [GoalEntry]) -> Insight? {
        // 取最近的记录
        let recentEntries = entries.suffix(min(30, entries.count))

        // 提取情感得分
        let sentimentScores = recentEntries.compactMap { entry -> Double? in
            guard entry.sentimentScore != 0 else { return nil }
            return entry.sentimentScore
        }

        guard sentimentScores.count > 10 else { return nil }

        // 计算前半段和后半段的平均情感得分
        let midPoint = sentimentScores.count / 2
        let firstHalf = sentimentScores.prefix(midPoint)
        let secondHalf = sentimentScores.suffix(sentimentScores.count - midPoint)

        let firstAvg = firstHalf.reduce(0, +) / Double(firstHalf.count)
        let secondAvg = secondHalf.reduce(0, +) / Double(secondHalf.count)

        let direction: String
        let threshold = 0.1

        if secondAvg > firstAvg + threshold {
            direction = "上升"
        } else if secondAvg < firstAvg - threshold {
            direction = "下降"
        } else {
            direction = "稳定"
        }

        return .sentimentTrend(direction: direction)
    }

    // MARK: - 新增分析方法 (Phase 5.2)

    /// 分析目标类型平衡度 (基于 ActionType)
    private func analyzeBalance(_ entries: [GoalEntry]) -> Insight? {
        guard entries.count >= 10 else { return nil }

        // 统计行动类型分布
        let actionTypeGroups = Dictionary(grouping: entries) {
            $0.effectiveActionType ?? "未分类"
        }

        guard let topType = actionTypeGroups.max(by: { $0.value.count < $1.value.count }) else {
            return nil
        }

        let percentage = Int(Double(topType.value.count) / Double(entries.count) * 100)

        // 只有当占比超过 60% 时才建议平衡
        guard percentage >= 60 else { return nil }

        // 建议互补类型 (基于 ActionType)
        let complementaryTypes: [String: String] = [
            "工作": "生活", // Work -> Lifestyle
            "学习": "社交", // Learning -> Social
            "运动": "学习", // Exercise -> Learning
            "生活": "工作", // Lifestyle -> Work
            "社交": "运动", // Social -> Exercise
            "未分类": "生活"
        ]

        let suggestedType = complementaryTypes[topType.key] ?? "其他"

        return .balance(mainType: topType.key, percentage: percentage, suggestedType: suggestedType)
    }

    /// 分析周期模式（周几最活跃）
    private func analyzeWeekdayPattern(_ entries: [GoalEntry]) -> Insight? {
        guard entries.count >= 14 else { return nil } // 至少2周数据

        let calendar = Calendar.current
        var weekdayCounts: [Int: Int] = [:]

        for entry in entries {
            if let date = GoalEntry.dateFrom(entry.dateString) {
                let weekday = calendar.component(.weekday, from: date)
                weekdayCounts[weekday, default: 0] += 1
            }
        }

        guard !weekdayCounts.isEmpty else { return nil }

        let busyDay = weekdayCounts.max(by: { $0.value < $1.value })
        let quietDay = weekdayCounts.min(by: { $0.value < $1.value })

        guard let busy = busyDay, let quiet = quietDay, busy.key != quiet.key else {
            return nil
        }

        let weekdayNames = ["", "周日", "周一", "周二", "周三", "周四", "周五", "周六"]

        return .weekdayPattern(
            busyDay: weekdayNames[busy.key],
            busyCount: busy.value,
            quietDay: weekdayNames[quiet.key]
        )
    }

    /// 分析目标数量趋势（本周 vs 上周）
    private func analyzeVolumeTrend(_ entries: [GoalEntry]) -> Insight? {
        let calendar = Calendar.current
        let today = calendar.startOfDay(for: Date())

        guard let thisWeekStart = calendar.date(byAdding: .day, value: -6, to: today),
              let lastWeekStart = calendar.date(byAdding: .day, value: -13, to: today),
              let lastWeekEnd = calendar.date(byAdding: .day, value: -7, to: today)
        else {
            return nil
        }

        let thisWeekString = GoalEntry.dateStringFrom(thisWeekStart)
        let lastWeekStartString = GoalEntry.dateStringFrom(lastWeekStart)
        let lastWeekEndString = GoalEntry.dateStringFrom(lastWeekEnd)
        let todayString = GoalEntry.dateStringFrom(today)

        let thisWeekCount = entries.filter {
            $0.dateString >= thisWeekString && $0.dateString <= todayString
        }.count

        let lastWeekCount = entries.filter {
            $0.dateString >= lastWeekStartString && $0.dateString <= lastWeekEndString
        }.count

        guard lastWeekCount > 0 else { return nil }

        let changePercent = Int(Double(thisWeekCount - lastWeekCount) / Double(lastWeekCount) * 100)

        let direction: String
        if changePercent > 10 {
            direction = "上升"
        } else if changePercent < -10 {
            direction = "下降"
        } else {
            direction = "稳定"
        }

        return .volumeTrend(direction: direction, changePercent: abs(changePercent))
    }

    /// 生成鼓励消息
    private func generateEncouragement(_ entries: [GoalEntry]) -> Insight? {
        // 计算连续记录天数
        let sortedDates = Set(entries.map { $0.dateString }).sorted(by: >)

        guard !sortedDates.isEmpty else { return nil }

        var streakDays = 0
        let calendar = Calendar.current
        var currentDate = calendar.startOfDay(for: Date())

        for dateString in sortedDates {
            let expectedDateString = GoalEntry.dateStringFrom(currentDate)
            if dateString == expectedDateString {
                streakDays += 1
                currentDate = calendar.date(byAdding: .day, value: -1, to: currentDate) ?? currentDate
            } else {
                break
            }
        }

        // 只有连续 >= 3 天才生成鼓励
        guard streakDays >= 3 else { return nil }

        let messages: [String]
        switch streakDays {
        case 3 ... 6:
            messages = ["继续保持！", "你做得很棒！", "每一天都在进步！"]
        case 7 ... 13:
            messages = ["一周的坚持，了不起！", "习惯正在形成！", "你的毅力令人印象深刻！"]
        case 14 ... 29:
            messages = ["两周的坚持，太棒了！", "你已经是习惯大师了！", "持续的力量！"]
        default:
            messages = ["传奇！超过一个月的坚持！", "你是真正的目标达人！", "无与伦比的毅力！"]
        }

        let message = messages.randomElement() ?? "继续加油！"

        return .encouragement(streakDays: streakDays, message: message)
    }

    /// 分析周对比
    private func analyzeWeekComparison(_ entries: [GoalEntry]) -> Insight? {
        let calendar = Calendar.current
        let today = calendar.startOfDay(for: Date())

        guard let thisWeekStart = calendar.date(byAdding: .day, value: -6, to: today),
              let lastWeekStart = calendar.date(byAdding: .day, value: -13, to: today),
              let lastWeekEnd = calendar.date(byAdding: .day, value: -7, to: today)
        else {
            return nil
        }

        let thisWeekString = GoalEntry.dateStringFrom(thisWeekStart)
        let lastWeekStartString = GoalEntry.dateStringFrom(lastWeekStart)
        let lastWeekEndString = GoalEntry.dateStringFrom(lastWeekEnd)
        let todayString = GoalEntry.dateStringFrom(today)

        let thisWeekCount = entries.filter {
            $0.dateString >= thisWeekString && $0.dateString <= todayString
        }.count

        let lastWeekCount = entries.filter {
            $0.dateString >= lastWeekStartString && $0.dateString <= lastWeekEndString
        }.count

        guard lastWeekCount > 0 || thisWeekCount > 0 else { return nil }

        let changePercent: Int
        if lastWeekCount > 0 {
            changePercent = Int(Double(thisWeekCount - lastWeekCount) / Double(lastWeekCount) * 100)
        } else {
            changePercent = 100
        }

        return .comparison(thisWeek: thisWeekCount, lastWeek: lastWeekCount, changePercent: changePercent)
    }

    // MARK: - Phase 5.2 补充分析方法

    /// 识别高频重复目标
    private func findRecurringGoals(_ entries: [GoalEntry]) -> Insight? {
        guard entries.count >= 10 else { return nil }

        // 统计目标文本的出现次数（简化：取前 10 个字符作为关键词）
        var goalCounts: [String: (count: Int, fullText: String, category: String?)] = [:]

        for entry in entries {
            let goalText = entry.goalText
            // 使用简化的目标关键词（去除空格、取前 15 个字符）
            let keyword = String(goalText.prefix(15)).trimmingCharacters(in: .whitespaces).lowercased()

            if let existing = goalCounts[keyword] {
                goalCounts[keyword] = (count: existing.count + 1, fullText: existing.fullText, category: existing.category)
            } else {
                goalCounts[keyword] = (count: 1, fullText: goalText, category: entry.effectiveCategory)
            }
        }

        // 找到出现次数最多的目标（至少 3 次）
        guard let topRecurring = goalCounts.max(by: { $0.value.count < $1.value.count }),
              topRecurring.value.count >= 3
        else {
            return nil
        }

        return .recurringGoal(
            goalText: topRecurring.value.fullText,
            count: topRecurring.value.count,
            category: topRecurring.value.category
        )
    }

    /// 分析目标可达成性（基于历史完成模式）
    private func analyzeAchievability(_ entries: [GoalEntry]) -> Insight? {
        guard entries.count >= 14 else { return nil }

        // 分析目标的复杂度分布
        var difficultyStats: [String: Int] = [:]
        for entry in entries {
            if let difficulty = entry.effectiveDifficulty {
                difficultyStats[difficulty, default: 0] += 1
            }
        }

        guard !difficultyStats.isEmpty else { return nil }

        // 计算难度分布
        let total = Double(difficultyStats.values.reduce(0, +))
        let highDifficultyCount = Double(difficultyStats["困难"] ?? 0)
        let lowDifficultyCount = Double(difficultyStats["简单"] ?? 0)

        let highRatio = highDifficultyCount / total
        let lowRatio = lowDifficultyCount / total

        let level: String
        let suggestion: String

        if highRatio > 0.5 {
            level = "需调整"
            suggestion = "你设定了较多困难目标。尝试拆分大目标为小步骤，提高可达成性。"
        } else if lowRatio > 0.7 {
            level = "可提升"
            suggestion = "你的目标大多比较简单。可以尝试设定一些更有挑战性的目标！"
        } else {
            level = "良好"
            suggestion = "你的目标难度分布均衡，继续保持！"
        }

        return .achievability(level: level, suggestion: suggestion)
    }

    private func fetchRecentEntries(days: Int) -> [GoalEntry] {
        let calendar = Calendar.current
        let endDate = Date()
        guard let startDate = calendar.date(byAdding: .day, value: -days, to: endDate) else {
            return []
        }

        let startString = GoalEntry.dateStringFrom(startDate)

        let request: NSFetchRequest<GoalEntry> = GoalEntry.fetchRequest()
        request.predicate = NSPredicate(format: "dateString >= %@", startString)
        request.sortDescriptors = [NSSortDescriptor(key: "dateString", ascending: true)]

        do {
            return try context.fetch(request)
        } catch {
            print("❌ 获取历史数据失败: \(error)")
            return []
        }
    }

    private func daysBetween(_ date1: String, _ date2: String) -> Int {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd"

        guard let d1 = formatter.date(from: date1),
              let d2 = formatter.date(from: date2)
        else {
            return Int.max
        }

        let days = Calendar.current.dateComponents([.day], from: d1, to: d2).day ?? Int.max
        return abs(days)
    }
}

// MARK: - 洞察描述扩展

extension Insight {
    var title: String {
        switch self {
        case .categoryDistribution:
            return "主题分布"
        case .streakBooster:
            return "连续记录助推器"
        case .consistencyPattern:
            return "坚持模式"
        case .sentimentTrend:
            return "情感趋势"
        case .balance:
            return "目标平衡"
        case .volumeTrend:
            return "活跃度趋势"
        case .weekdayPattern:
            return "周期规律"
        case .encouragement:
            return "鼓励"
        case .comparison:
            return "周对比"
        case .recurringGoal:
            return "高频目标"
        case .achievability:
            return "可达成性"
        }
    }

    var description: String {
        switch self {
        case let .categoryDistribution(category, percentage):
            return "在过去30天里，你 \(percentage)% 的目标与「\(category)」相关。"

        case let .streakBooster(category, improvement):
            return "我们注意到，当你记录与「\(category)」相关的目标时，你的连续记录保持率提高了 \(improvement)%。"

        case let .consistencyPattern(category, days):
            return "你已经连续 \(days) 天关注「\(category)」，保持得很好！"

        case let .sentimentTrend(direction):
            return "你的整体情感倾向呈现\(direction)趋势。"

        case let .balance(mainType, percentage, suggestedType):
            return "你 \(percentage)% 的目标都是「\(mainType)」类型。尝试添加一些「\(suggestedType)」类型的目标来保持平衡！"

        case let .volumeTrend(direction, changePercent):
            if direction == "上升" {
                return "太棒了！你的目标记录活跃度较上周提升了 \(changePercent)%。"
            } else if direction == "下降" {
                return "你的目标记录活跃度较上周下降了 \(changePercent)%，继续加油！"
            } else {
                return "你的目标记录活跃度保持稳定。"
            }

        case let .weekdayPattern(busyDay, busyCount, quietDay):
            return "你在\(busyDay)最活跃（\(busyCount)个目标），而\(quietDay)相对安静。"

        case let .encouragement(streakDays, message):
            if streakDays > 0 {
                return "🔥 你已经连续记录 \(streakDays) 天了！\(message)"
            } else {
                return message
            }

        case let .comparison(thisWeek, lastWeek, changePercent):
            if thisWeek > lastWeek {
                return "本周你记录了 \(thisWeek) 个目标，比上周多 \(changePercent)%！"
            } else if thisWeek < lastWeek {
                return "本周你记录了 \(thisWeek) 个目标，比上周少 \(abs(changePercent))%。"
            } else {
                return "本周与上周记录数量相同（\(thisWeek) 个目标）。"
            }

        case let .recurringGoal(goalText, count, category):
            let categoryNote = category.map { "【\($0)】" } ?? ""
            return "你已经 \(count) 次记录了「\(goalText)」\(categoryNote)。这是你的核心目标！"

        case let .achievability(level, suggestion):
            return "[可达成性: \(level)] \(suggestion)"
        }
    }

    var iconName: String {
        switch self {
        case .categoryDistribution:
            return "chart.pie.fill"
        case .streakBooster:
            return "flame.fill"
        case .consistencyPattern:
            return "checkmark.circle.fill"
        case .sentimentTrend:
            return "chart.line.uptrend.xyaxis"
        case .balance:
            return "scale.3d"
        case .volumeTrend:
            return "arrow.up.right.circle.fill"
        case .weekdayPattern:
            return "calendar.badge.clock"
        case .encouragement:
            return "star.fill"
        case .comparison:
            return "arrow.left.arrow.right"
        case .recurringGoal:
            return "repeat.circle.fill"
        case .achievability:
            return "target"
        }
    }
}
