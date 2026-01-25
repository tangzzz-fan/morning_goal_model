//
//  GoalDataAggregator.swift
//  MorningGoal
//
//  目标数据聚合服务
//  提供时间范围内的统计、分布、趋势等查询功能
//

import Accelerate
import CoreData
import Foundation

// MARK: - 聚合统计结果模型

/// 聚合统计时间周期
enum AggregationPeriod: String, CaseIterable {
    case day
    case week
    case month
    case year

    var displayName: String {
        switch self {
        case .day: return "日"
        case .week: return "周"
        case .month: return "月"
        case .year: return "年"
        }
    }

    /// 获取周期对应的天数（用于默认范围）
    var defaultDays: Int {
        switch self {
        case .day: return 1
        case .week: return 7
        case .month: return 30
        case .year: return 365
        }
    }
}

/// 分组维度
enum AggregationGroupBy: String, CaseIterable {
    case topic
    case sentiment
    case urgency
    case timeFrame
    case actionType
    case difficulty
    case specificity
    case weekday

    var displayName: String {
        switch self {
        case .topic: return "主题"
        case .sentiment: return "情感"
        case .urgency: return "紧急度"
        case .timeFrame: return "时间范围"
        case .actionType: return "行动类型"
        case .difficulty: return "难度"
        case .specificity: return "具体程度"
        case .weekday: return "星期"
        }
    }
}

/// 聚合统计结果
struct AggregatedStats {
    /// 总目标数
    let totalGoals: Int

    /// 已分析的目标数
    let analyzedGoals: Int

    /// 有用户纠正的目标数
    let correctedGoals: Int

    /// 各维度的分布统计
    let topicDistribution: [String: Int]
    let sentimentDistribution: [String: Int]
    let urgencyDistribution: [String: Int]
    let timeFrameDistribution: [String: Int]
    let actionTypeDistribution: [String: Int]
    let difficultyDistribution: [String: Int]
    let specificityDistribution: [String: Int]

    /// 平均置信度
    let averageConfidence: DimensionConfidence

    /// 统计时间范围
    let fromDate: Date
    let toDate: Date
}

/// 各维度平均置信度
struct DimensionConfidence {
    let topic: Double
    let sentiment: Double
    let urgency: Double
    let timeFrame: Double
    let actionType: Double
    let difficulty: Double
    let specificity: Double

    var overall: Double {
        (topic + sentiment + urgency + timeFrame + actionType + difficulty + specificity) / 7.0
    }
}

/// 分布项
struct DistributionItem: Identifiable {
    let id = UUID()
    let label: String
    let count: Int
    let percentage: Double
    let icon: String?
}

/// 趋势数据点
struct TrendDataPoint: Identifiable {
    let id = UUID()
    let date: Date
    let dateString: String
    let value: Double
    let count: Int
}

/// 情感趋势结果
struct SentimentTrendResult {
    let period: AggregationPeriod
    let dataPoints: [TrendDataPoint]
    let averageSentiment: Double // -1 消极, 0 中性, 1 积极
}

/// 完成率结果
struct CompletionRateResult {
    let groupBy: AggregationGroupBy
    let period: AggregationPeriod
    let items: [CompletionRateItem]
    let overallRate: Double
}

/// 完成率项
struct CompletionRateItem: Identifiable {
    let id = UUID()
    let label: String
    let totalCount: Int
    let completedCount: Int
    let rate: Double
}

/// 相似目标结果
struct SimilarGoalResult: Identifiable {
    let id: NSManagedObjectID
    let goalText: String
    let dateString: String
    let similarity: Double
    let category: String?
}

/// 每日情感统计（内部使用）
private struct DailySentimentData {
    var positiveCount: Int = 0
    var neutralCount: Int = 0
    var negativeCount: Int = 0
    var total: Int = 0
}

// MARK: - GoalDataAggregator 服务

/// 目标数据聚合服务
/// 提供统计查询、分布分析、趋势分析等功能
final class GoalDataAggregator {
    private let viewContext: NSManagedObjectContext

    init(viewContext: NSManagedObjectContext) {
        self.viewContext = viewContext
    }

    // MARK: - 时间范围工具方法

    /// 获取日期范围（基于周期）
    func getDateRange(for period: AggregationPeriod, from referenceDate: Date = Date()) -> (from: Date, to: Date) {
        let calendar = Calendar.current
        let endOfDay = calendar.startOfDay(for: referenceDate).addingTimeInterval(86400) // 明天0点

        let startDate: Date
        switch period {
        case .day:
            startDate = calendar.startOfDay(for: referenceDate)
        case .week:
            startDate = calendar.date(byAdding: .day, value: -6, to: calendar.startOfDay(for: referenceDate))!
        case .month:
            startDate = calendar.date(byAdding: .day, value: -29, to: calendar.startOfDay(for: referenceDate))!
        case .year:
            startDate = calendar.date(byAdding: .day, value: -364, to: calendar.startOfDay(for: referenceDate))!
        }

        return (from: startDate, to: endOfDay)
    }

    // MARK: - 1. 获取聚合统计 (getAggregatedStats)

    /// 获取时间范围内的聚合统计
    /// - Parameters:
    ///   - from: 开始日期
    ///   - to: 结束日期
    /// - Returns: 聚合统计结果
    func getAggregatedStats(from: Date, to: Date) -> AggregatedStats {
        let entries = fetchEntries(from: from, to: to)

        // 计算各类统计
        let totalGoals = entries.count
        let analyzedGoals = entries.filter { $0.analyzedAt != nil }.count
        let correctedGoals = entries.filter { $0.hasUserCorrections }.count

        // 统计各维度分布
        let topicDistribution = countDistribution(entries: entries) { $0.effectiveCategory }
        let sentimentDistribution = countDistribution(entries: entries) { $0.effectiveSentiment }
        let urgencyDistribution = countDistribution(entries: entries) { $0.effectiveUrgency }
        let timeFrameDistribution = countDistribution(entries: entries) { $0.effectiveTimeFrame }
        let actionTypeDistribution = countDistribution(entries: entries) { $0.effectiveActionType }
        let difficultyDistribution = countDistribution(entries: entries) { $0.effectiveDifficulty }
        let specificityDistribution = countDistribution(entries: entries) { $0.effectiveSpecificity }

        // 计算平均置信度
        let averageConfidence = calculateAverageConfidence(entries: entries)

        return AggregatedStats(
            totalGoals: totalGoals,
            analyzedGoals: analyzedGoals,
            correctedGoals: correctedGoals,
            topicDistribution: topicDistribution,
            sentimentDistribution: sentimentDistribution,
            urgencyDistribution: urgencyDistribution,
            timeFrameDistribution: timeFrameDistribution,
            actionTypeDistribution: actionTypeDistribution,
            difficultyDistribution: difficultyDistribution,
            specificityDistribution: specificityDistribution,
            averageConfidence: averageConfidence,
            fromDate: from,
            toDate: to
        )
    }

    /// 便捷方法：根据周期获取统计
    func getAggregatedStats(for period: AggregationPeriod) -> AggregatedStats {
        let range = getDateRange(for: period)
        return getAggregatedStats(from: range.from, to: range.to)
    }

    // MARK: - 2. 获取主题分布 (getTopicDistribution)

    /// 获取主题分布
    /// - Parameter period: 时间周期
    /// - Returns: 分布项数组（按数量降序排列）
    func getTopicDistribution(period: AggregationPeriod) -> [DistributionItem] {
        let range = getDateRange(for: period)
        return getDistribution(for: .topic, from: range.from, to: range.to)
    }

    /// 获取指定维度的分布
    /// - Parameters:
    ///   - groupBy: 分组维度
    ///   - from: 开始日期
    ///   - to: 结束日期
    /// - Returns: 分布项数组
    func getDistribution(for groupBy: AggregationGroupBy, from: Date, to: Date) -> [DistributionItem] {
        let entries = fetchEntries(from: from, to: to)
        let distribution = countDistributionByDimension(entries: entries, dimension: groupBy)

        let total = distribution.values.reduce(0, +)
        guard total > 0 else { return [] }

        return distribution.map { key, count in
            DistributionItem(
                label: key,
                count: count,
                percentage: Double(count) / Double(total) * 100,
                icon: getIcon(for: groupBy, label: key)
            )
        }.sorted { $0.count > $1.count }
    }

    // MARK: - 3. 获取情感趋势 (getSentimentTrend)

    /// 获取情感趋势
    /// - Parameter days: 天数
    /// - Returns: 情感趋势结果
    func getSentimentTrend(days: Int) -> SentimentTrendResult {
        let calendar = Calendar.current
        let today = calendar.startOfDay(for: Date())
        let startDate = calendar.date(byAdding: .day, value: -(days - 1), to: today)!
        let endDate = today.addingTimeInterval(86400)

        let entries = fetchEntries(from: startDate, to: endDate)

        // 按日期分组
        var dailyData: [String: DailySentimentData] = [:]

        for entry in entries {
            let dateString = entry.dateString
            var data = dailyData[dateString] ?? DailySentimentData()
            data.total += 1

            if let sentiment = entry.effectiveSentiment {
                switch sentiment {
                case "积极": data.positiveCount += 1
                case "中性": data.neutralCount += 1
                case "消极": data.negativeCount += 1
                default: break
                }
            }

            dailyData[dateString] = data
        }

        // 生成数据点
        var dataPoints: [TrendDataPoint] = []
        var totalSentimentScore = 0.0
        var totalCount = 0

        for dayOffset in 0 ..< days {
            guard let date = calendar.date(byAdding: .day, value: dayOffset - (days - 1), to: today) else { continue }
            let dateString = GoalEntry.dateStringFrom(date)

            if let data = dailyData[dateString], data.total > 0 {
                // 计算情感分数：积极=1, 中性=0, 消极=-1
                let sentimentScore = Double(data.positiveCount - data.negativeCount) / Double(data.total)
                dataPoints.append(TrendDataPoint(
                    date: date,
                    dateString: dateString,
                    value: sentimentScore,
                    count: data.total
                ))
                totalSentimentScore += sentimentScore
                totalCount += 1
            } else {
                // 无数据的日期
                dataPoints.append(TrendDataPoint(
                    date: date,
                    dateString: dateString,
                    value: 0,
                    count: 0
                ))
            }
        }

        let averageSentiment = totalCount > 0 ? totalSentimentScore / Double(totalCount) : 0

        return SentimentTrendResult(
            period: days <= 7 ? .week : (days <= 30 ? .month : .year),
            dataPoints: dataPoints,
            averageSentiment: averageSentiment
        )
    }

    // MARK: - 4. 获取完成率 (getCompletionRate)

    /// 获取完成率（按维度分组）
    /// - Note: 由于当前数据模型没有"完成"字段，这里使用 analyzedAt 是否存在作为代理
    ///         实际项目中应该添加 isCompleted 字段
    /// - Parameters:
    ///   - groupBy: 分组维度
    ///   - period: 时间周期
    /// - Returns: 完成率结果
    func getCompletionRate(groupBy: AggregationGroupBy, period: AggregationPeriod) -> CompletionRateResult {
        let range = getDateRange(for: period)
        let entries = fetchEntries(from: range.from, to: range.to)

        // 按维度分组统计
        var groupedData: [String: (total: Int, analyzed: Int)] = [:]

        for entry in entries {
            let key: String
            switch groupBy {
            case .topic:
                key = entry.effectiveCategory ?? "未分类"
            case .sentiment:
                key = entry.effectiveSentiment ?? "未分类"
            case .urgency:
                key = entry.effectiveUrgency ?? "未分类"
            case .timeFrame:
                key = entry.effectiveTimeFrame ?? "未分类"
            case .actionType:
                key = entry.effectiveActionType ?? "未分类"
            case .difficulty:
                key = entry.effectiveDifficulty ?? "未分类"
            case .specificity:
                key = entry.effectiveSpecificity ?? "未分类"
            case .weekday:
                key = getWeekdayName(from: entry.dateString)
            }

            var data = groupedData[key] ?? (0, 0)
            data.total += 1
            if entry.analyzedAt != nil {
                data.analyzed += 1
            }
            groupedData[key] = data
        }

        // 生成完成率项
        let items = groupedData.map { key, data in
            CompletionRateItem(
                label: key,
                totalCount: data.total,
                completedCount: data.analyzed,
                rate: data.total > 0 ? Double(data.analyzed) / Double(data.total) * 100 : 0
            )
        }.sorted { $0.totalCount > $1.totalCount }

        // 计算整体完成率
        let totalCount = entries.count
        let analyzedCount = entries.filter { $0.analyzedAt != nil }.count
        let overallRate = totalCount > 0 ? Double(analyzedCount) / Double(totalCount) * 100 : 0

        return CompletionRateResult(
            groupBy: groupBy,
            period: period,
            items: items,
            overallRate: overallRate
        )
    }

    // MARK: - 5. 相似目标搜索 (getSimilarGoals)

    /// 搜索相似目标（基于 embedding 余弦相似度）
    /// - Parameters:
    ///   - embedding: 查询目标的 embedding 向量 (Data 格式)
    ///   - limit: 返回结果数量限制
    ///   - excludeObjectID: 排除的目标 ID（通常是查询目标本身）
    /// - Returns: 相似目标结果数组（按相似度降序）
    func getSimilarGoals(embedding: Data, limit: Int = 10, excludeObjectID: NSManagedObjectID? = nil) -> [SimilarGoalResult] {
        // 解析查询 embedding
        guard let queryVector = embeddingDataToFloatArray(embedding) else {
            return []
        }

        // 获取所有有 embedding 的目标
        let fetchRequest: NSFetchRequest<GoalEntry> = GoalEntry.fetchRequest()
        fetchRequest.predicate = NSPredicate(format: "embedding != nil")

        guard let entries = try? viewContext.fetch(fetchRequest) else {
            return []
        }

        // 计算相似度
        var results: [(entry: GoalEntry, similarity: Double)] = []

        for entry in entries {
            // 排除指定 ID
            if let excludeID = excludeObjectID, entry.objectID == excludeID {
                continue
            }

            guard let entryEmbedding = entry.embedding,
                  let entryVector = embeddingDataToFloatArray(entryEmbedding)
            else {
                continue
            }

            let similarity = cosineSimilarity(queryVector, entryVector)
            results.append((entry: entry, similarity: similarity))
        }

        // 按相似度排序并限制数量
        let sortedResults = results.sorted { $0.similarity > $1.similarity }
            .prefix(limit)

        return sortedResults.map { item in
            SimilarGoalResult(
                id: item.entry.objectID,
                goalText: item.entry.goalText,
                dateString: item.entry.dateString,
                similarity: item.similarity,
                category: item.entry.effectiveCategory
            )
        }
    }

    /// 根据目标文本搜索相似目标
    /// - Note: 需要先通过 InsightModelManager 获取文本的 embedding
    /// - Parameters:
    ///   - goalEntry: 目标条目
    ///   - limit: 返回结果数量限制
    /// - Returns: 相似目标结果数组
    func getSimilarGoals(to goalEntry: GoalEntry, limit: Int = 10) -> [SimilarGoalResult] {
        guard let embedding = goalEntry.embedding else {
            return []
        }
        return getSimilarGoals(embedding: embedding, limit: limit, excludeObjectID: goalEntry.objectID)
    }

    // MARK: - 辅助统计方法

    /// 获取按日期分组的目标数量趋势
    func getGoalCountTrend(days: Int) -> [TrendDataPoint] {
        let calendar = Calendar.current
        let today = calendar.startOfDay(for: Date())
        let startDate = calendar.date(byAdding: .day, value: -(days - 1), to: today)!
        let endDate = today.addingTimeInterval(86400)

        let entries = fetchEntries(from: startDate, to: endDate)

        // 按日期分组计数
        var dailyCounts: [String: Int] = [:]
        for entry in entries {
            dailyCounts[entry.dateString, default: 0] += 1
        }

        // 生成数据点
        var dataPoints: [TrendDataPoint] = []
        for dayOffset in 0 ..< days {
            guard let date = calendar.date(byAdding: .day, value: dayOffset - (days - 1), to: today) else { continue }
            let dateString = GoalEntry.dateStringFrom(date)
            let count = dailyCounts[dateString] ?? 0

            dataPoints.append(TrendDataPoint(
                date: date,
                dateString: dateString,
                value: Double(count),
                count: count
            ))
        }

        return dataPoints
    }

    /// 获取星期分布（周几目标最多）
    func getWeekdayDistribution(period: AggregationPeriod) -> [DistributionItem] {
        let range = getDateRange(for: period)
        let entries = fetchEntries(from: range.from, to: range.to)

        var weekdayCounts: [Int: Int] = [:] // 1=周日, 2=周一, ..., 7=周六

        let calendar = Calendar.current
        for entry in entries {
            if let date = GoalEntry.dateFrom(entry.dateString) {
                let weekday = calendar.component(.weekday, from: date)
                weekdayCounts[weekday, default: 0] += 1
            }
        }

        let total = entries.count
        let weekdayNames = ["", "周日", "周一", "周二", "周三", "周四", "周五", "周六"]

        return (1 ... 7).map { weekday in
            let count = weekdayCounts[weekday] ?? 0
            return DistributionItem(
                label: weekdayNames[weekday],
                count: count,
                percentage: total > 0 ? Double(count) / Double(total) * 100 : 0,
                icon: nil
            )
        }
    }

    // MARK: - 私有辅助方法

    /// 获取时间范围内的目标条目
    private func fetchEntries(from: Date, to: Date) -> [GoalEntry] {
        let fromString = GoalEntry.dateStringFrom(from)
        let toString = GoalEntry.dateStringFrom(to)

        let fetchRequest: NSFetchRequest<GoalEntry> = GoalEntry.fetchRequest()
        fetchRequest.predicate = NSPredicate(format: "dateString >= %@ AND dateString <= %@", fromString, toString)
        fetchRequest.sortDescriptors = [NSSortDescriptor(key: "dateString", ascending: false)]

        return (try? viewContext.fetch(fetchRequest)) ?? []
    }

    /// 统计分布
    private func countDistribution(entries: [GoalEntry], keyPath: (GoalEntry) -> String?) -> [String: Int] {
        var distribution: [String: Int] = [:]
        for entry in entries {
            if let key = keyPath(entry) {
                distribution[key, default: 0] += 1
            }
        }
        return distribution
    }

    /// 按维度统计分布
    private func countDistributionByDimension(entries: [GoalEntry], dimension: AggregationGroupBy) -> [String: Int] {
        switch dimension {
        case .topic:
            return countDistribution(entries: entries) { $0.effectiveCategory }
        case .sentiment:
            return countDistribution(entries: entries) { $0.effectiveSentiment }
        case .urgency:
            return countDistribution(entries: entries) { $0.effectiveUrgency }
        case .timeFrame:
            return countDistribution(entries: entries) { $0.effectiveTimeFrame }
        case .actionType:
            return countDistribution(entries: entries) { $0.effectiveActionType }
        case .difficulty:
            return countDistribution(entries: entries) { $0.effectiveDifficulty }
        case .specificity:
            return countDistribution(entries: entries) { $0.effectiveSpecificity }
        case .weekday:
            var distribution: [String: Int] = [:]
            for entry in entries {
                let weekday = getWeekdayName(from: entry.dateString)
                distribution[weekday, default: 0] += 1
            }
            return distribution
        }
    }

    /// 计算平均置信度
    private func calculateAverageConfidence(entries: [GoalEntry]) -> DimensionConfidence {
        guard !entries.isEmpty else {
            return DimensionConfidence(topic: 0, sentiment: 0, urgency: 0, timeFrame: 0, actionType: 0, difficulty: 0, specificity: 0)
        }

        let count = Double(entries.count)
        let topic = entries.reduce(0.0) { $0 + $1.categoryConfidence } / count
        let sentiment = entries.reduce(0.0) { $0 + $1.sentimentScore } / count
        let urgency = entries.reduce(0.0) { $0 + $1.urgencyConfidence } / count
        let timeFrame = entries.reduce(0.0) { $0 + $1.timeFrameConfidence } / count
        let actionType = entries.reduce(0.0) { $0 + $1.actionTypeConfidence } / count
        let difficulty = entries.reduce(0.0) { $0 + $1.difficultyConfidence } / count
        let specificity = entries.reduce(0.0) { $0 + $1.specificityConfidence } / count

        return DimensionConfidence(
            topic: topic,
            sentiment: sentiment,
            urgency: urgency,
            timeFrame: timeFrame,
            actionType: actionType,
            difficulty: difficulty,
            specificity: specificity
        )
    }

    /// 获取维度标签对应的图标
    private func getIcon(for dimension: AggregationGroupBy, label: String) -> String? {
        switch dimension {
        case .topic:
            return InsightLabels.Topic(rawValue: label)?.icon
        case .sentiment:
            return InsightLabels.Sentiment(rawValue: label)?.icon
        case .urgency:
            return InsightLabels.Urgency(rawValue: label)?.icon
        case .timeFrame:
            return InsightLabels.TimeFrame(rawValue: label)?.icon
        case .actionType:
            return InsightLabels.ActionType(rawValue: label)?.icon
        case .difficulty:
            return InsightLabels.Difficulty(rawValue: label)?.icon
        case .specificity:
            return InsightLabels.Specificity(rawValue: label)?.icon
        case .weekday:
            return nil
        }
    }

    /// 从日期字符串获取星期名称
    private func getWeekdayName(from dateString: String) -> String {
        guard let date = GoalEntry.dateFrom(dateString) else {
            return "未知"
        }
        let weekday = Calendar.current.component(.weekday, from: date)
        let weekdayNames = ["", "周日", "周一", "周二", "周三", "周四", "周五", "周六"]
        return weekdayNames[weekday]
    }

    /// 将 Data 格式的 embedding 转换为 Float 数组
    private func embeddingDataToFloatArray(_ data: Data) -> [Float]? {
        let floatCount = data.count / MemoryLayout<Float>.size
        guard floatCount > 0 else { return nil }

        var floatArray = [Float](repeating: 0, count: floatCount)
        _ = floatArray.withUnsafeMutableBytes { ptr in
            data.copyBytes(to: ptr)
        }
        return floatArray
    }

    /// 计算余弦相似度
    private func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Double {
        guard a.count == b.count, !a.isEmpty else { return 0 }

        var dotProduct: Float = 0
        var normA: Float = 0
        var normB: Float = 0

        // 使用 Accelerate 框架加速计算
        vDSP_dotpr(a, 1, b, 1, &dotProduct, vDSP_Length(a.count))
        vDSP_dotpr(a, 1, a, 1, &normA, vDSP_Length(a.count))
        vDSP_dotpr(b, 1, b, 1, &normB, vDSP_Length(b.count))

        let denominator = sqrt(normA) * sqrt(normB)
        guard denominator > 0 else { return 0 }

        return Double(dotProduct / denominator)
    }
}
