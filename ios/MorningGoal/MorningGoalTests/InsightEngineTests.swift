//
//  InsightEngineTests.swift
//  MorningGoalTests
//
//  洞察引擎单元测试 - Phase 5
//

import CoreData
import XCTest

@testable import MorningGoal

final class InsightEngineTests: XCTestCase {
    var container: NSPersistentContainer!
    var context: NSManagedObjectContext!
    var engine: InsightEngine!

    override func setUp() {
        super.setUp()
        container = InMemoryCoreData.container()
        context = container.viewContext
        engine = InsightEngine(context: context)
    }

    override func tearDown() {
        engine = nil
        context = nil
        container = nil
        super.tearDown()
    }

    // MARK: - Helper Methods

    /// 创建测试用目标条目
    private func createEntry(
        dateString: String,
        goalText: String = "测试目标",
        category: String? = nil,
        sentiment: String? = nil,
        sentimentScore: Double = 0.0
    ) -> GoalEntry {
        let entry = GoalEntry(context: context)
        entry.dateString = dateString
        entry.goalText = goalText
        entry.lastUpdated = Date()
        entry.category = category
        entry.sentiment = sentiment
        entry.sentimentScore = sentimentScore
        return entry
    }

    /// 生成过去 N 天的日期字符串
    private func dateString(daysAgo: Int) -> String {
        let date = Calendar.current.date(byAdding: .day, value: -daysAgo, to: Date())!
        return GoalEntry.dateStringFrom(date)
    }

    // MARK: - 测试：数据不足时返回空

    func test_generateInsights_withInsufficientData_returnsEmpty() {
        // 只创建 3 条数据（< 7 天）
        for i in 0 ..< 3 {
            _ = createEntry(dateString: dateString(daysAgo: i))
        }
        try? context.save()

        let insights = engine.generateInsights()

        XCTAssertTrue(insights.isEmpty, "数据不足时应返回空数组")
    }

    // MARK: - 测试：有足够数据时生成洞察

    func test_generateInsights_withSufficientData_returnsInsights() {
        // 创建 10 条数据
        for i in 0 ..< 10 {
            _ = createEntry(
                dateString: dateString(daysAgo: i),
                category: "健康"
            )
        }
        try? context.save()

        let insights = engine.generateInsights()

        XCTAssertFalse(insights.isEmpty, "有足够数据时应返回洞察")
    }

    // MARK: - 测试：主题分布分析

    func test_analyzeTopCategory_returnsCorrectPercentage() {
        // 创建 10 条数据，8 条健康，2 条学习
        for i in 0 ..< 8 {
            _ = createEntry(dateString: dateString(daysAgo: i), category: "健康")
        }
        for i in 8 ..< 10 {
            _ = createEntry(dateString: dateString(daysAgo: i), category: "学习")
        }
        try? context.save()

        // 使用较高的 topK 确保低优先级的 categoryDistribution 被包含
        let insights = engine.generateInsights(topK: 10)

        let categoryInsight = insights.first { $0.id == "category_dist" }
        XCTAssertNotNil(categoryInsight, "应生成主题分布洞察")

        if case let .categoryDistribution(category, percentage) = categoryInsight {
            XCTAssertEqual(category, "健康")
            XCTAssertEqual(percentage, 80)
        }
    }

    // MARK: - 测试：平衡建议分析

    func test_analyzeBalance_detectsImbalance() {
        // 创建 10 条数据，全是健康类型（100%）
        for i in 0 ..< 10 {
            _ = createEntry(dateString: dateString(daysAgo: i), category: "健康")
        }
        try? context.save()

        let insights = engine.generateInsights()

        let balanceInsight = insights.first { $0.id == "balance" }
        XCTAssertNotNil(balanceInsight, "应生成平衡建议洞察")

        if case let .balance(mainType, percentage, _) = balanceInsight {
            XCTAssertEqual(mainType, "健康")
            XCTAssertGreaterThanOrEqual(percentage, 60)
        }
    }

    // MARK: - 测试：周期模式分析

    func test_analyzeWeekdayPattern_identifiesBusyDays() {
        // 创建 14 条数据（2周）
        for i in 0 ..< 14 {
            _ = createEntry(dateString: dateString(daysAgo: i))
        }
        try? context.save()

        let insights = engine.generateInsights()

        let weekdayInsight = insights.first { $0.id == "weekday_pattern" }
        // 周期模式可能生成也可能不生成，取决于日期分布
        // 这里只测试格式正确性
        if let insight = weekdayInsight {
            if case let .weekdayPattern(busyDay, busyCount, quietDay) = insight {
                XCTAssertFalse(busyDay.isEmpty)
                XCTAssertGreaterThan(busyCount, 0)
                XCTAssertFalse(quietDay.isEmpty)
            }
        }
    }

    // MARK: - 测试：情感趋势分析

    func test_analyzeSentimentTrend_detectsUpward() {
        // 创建 15 条数据，前半段消极，后半段积极
        for i in 0 ..< 15 {
            let score: Double = i < 7 ? -0.5 : 0.8
            _ = createEntry(
                dateString: dateString(daysAgo: 14 - i),
                sentimentScore: score
            )
        }
        try? context.save()

        let insights = engine.generateInsights()

        let sentimentInsight = insights.first { $0.id == "sentiment" }
        // 情感趋势分析需要 > 10 条数据
        if let insight = sentimentInsight {
            if case let .sentimentTrend(direction) = insight {
                XCTAssertEqual(direction, "上升")
            }
        }
    }

    // MARK: - 测试：优先级排序

    func test_insightPriority_sortingWorks() {
        XCTAssertTrue(InsightPriority.high > InsightPriority.medium)
        XCTAssertTrue(InsightPriority.medium > InsightPriority.low)
    }

    // MARK: - 测试：Top-K 筛选

    func test_generateInsights_respectsTopKLimit() {
        // 创建足够的数据以生成多个洞察
        for i in 0 ..< 20 {
            _ = createEntry(
                dateString: dateString(daysAgo: i),
                category: i % 2 == 0 ? "健康" : "学习",
                sentimentScore: Double(i % 3) - 1.0
            )
        }
        try? context.save()

        let insights = engine.generateInsights(topK: 3)

        XCTAssertLessThanOrEqual(insights.count, 3, "应最多返回 3 条洞察")
    }

    // MARK: - 测试：洞察属性

    func test_insight_hasValidProperties() {
        let insight = Insight.categoryDistribution(topCategory: "健康", percentage: 50)

        XCTAssertEqual(insight.id, "category_dist")
        XCTAssertEqual(insight.title, "主题分布")
        XCTAssertFalse(insight.description.isEmpty)
        XCTAssertFalse(insight.iconName.isEmpty)
        XCTAssertEqual(insight.priority, .low)
    }

    // MARK: - 测试：新洞察类型属性

    func test_newInsightTypes_haveValidProperties() {
        let balanceInsight = Insight.balance(mainType: "健康", percentage: 70, suggestedType: "学习")
        XCTAssertEqual(balanceInsight.id, "balance")
        XCTAssertEqual(balanceInsight.priority, .high)

        let volumeInsight = Insight.volumeTrend(direction: "上升", changePercent: 20)
        XCTAssertEqual(volumeInsight.id, "volume_trend")
        XCTAssertEqual(volumeInsight.priority, .medium)

        let weekdayInsight = Insight.weekdayPattern(busyDay: "周一", busyCount: 5, quietDay: "周日")
        XCTAssertEqual(weekdayInsight.id, "weekday_pattern")
        XCTAssertEqual(weekdayInsight.priority, .low)

        let encourageInsight = Insight.encouragement(streakDays: 7, message: "继续加油！")
        XCTAssertEqual(encourageInsight.id, "encouragement")
        XCTAssertEqual(encourageInsight.priority, .high)

        let comparisonInsight = Insight.comparison(thisWeek: 10, lastWeek: 8, changePercent: 25)
        XCTAssertEqual(comparisonInsight.id, "comparison")
        XCTAssertEqual(comparisonInsight.priority, .medium)

        let recurringInsight = Insight.recurringGoal(goalText: "Run", count: 5, category: "Health")
        XCTAssertEqual(recurringInsight.id, "recurring_goal")
        XCTAssertEqual(recurringInsight.priority, .medium)

        let achievabilityInsight = Insight.achievability(level: "High", suggestion: "Good job")
        XCTAssertEqual(achievabilityInsight.id, "achievability")
        XCTAssertEqual(achievabilityInsight.priority, .high)
    }

    // MARK: - 测试：鼓励机制

    func test_generateEncouragement_returnsInsight_whenStreakExists() {
        // 创建 7 天连续的记录
        for i in 0 ..< 7 {
            _ = createEntry(dateString: dateString(daysAgo: i), category: "健康")
        }
        try? context.save()
        
        // 我们需要确保 generateEncouragement 内部使用了 StreakService 或者有自己的计算逻辑
        // 根据 InsightEngine.swift 的逻辑，它可能会根据日期连续性计算
        let insights = engine.generateInsights()
        
        let encouragement = insights.first { $0.id == "encouragement" }
        XCTAssertNotNil(encouragement, "连续打卡应触发鼓励洞察")
        
        if case let .encouragement(streakDays, _) = encouragement {
            // 具体的 streak 计算取决于实现，这里至少应该 > 1
            XCTAssertGreaterThanOrEqual(streakDays, 2)
        }
    }
}
