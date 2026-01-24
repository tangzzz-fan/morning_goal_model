import CoreData
import Foundation

// MARK: - 洞察类型

enum Insight: Equatable {
    case categoryDistribution(topCategory: String, percentage: Int)
    case streakBooster(category: String, improvement: Int)
    case consistencyPattern(category: String, days: Int)
    case sentimentTrend(direction: String)

    var id: String {
        switch self {
        case .categoryDistribution: return "category_dist"
        case .streakBooster: return "streak_boost"
        case .consistencyPattern: return "consistency"
        case .sentimentTrend: return "sentiment"
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

    func generateInsights(for period: Int = 30) -> [Insight] {
        var insights: [Insight] = []

        // 获取最近N天的数据
        let entries = fetchRecentEntries(days: period)

        guard entries.count >= 7 else {
            print("⏭️ 数据不足（<7天），跳过洞察生成")
            return insights
        }

        print("📊 生成洞察：基于 \(entries.count) 条数据")

        // 洞察1：主题分布
        if let topCategoryInsight = analyzeTopCategory(entries) {
            insights.append(topCategoryInsight)
        }

        // 洞察2：Streak关联
        if let streakInsight = analyzeStreakCorrelation(entries) {
            insights.append(streakInsight)
        }

        // 洞察3：一致性模式
        if let consistencyInsight = analyzeConsistency(entries) {
            insights.append(consistencyInsight)
        }

        // 洞察4：情感趋势
        if let sentimentInsight = analyzeSentimentTrend(entries) {
            insights.append(sentimentInsight)
        }

        print("✅ 生成了 \(insights.count) 条洞察")

        return insights
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

    // MARK: - 辅助方法

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
        }
    }
}
