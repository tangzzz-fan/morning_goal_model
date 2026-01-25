//
//  GoalDataAggregatorTests.swift
//  MorningGoalTests
//
//  GoalDataAggregator 单元测试
//

import CoreData
import XCTest

@testable import MorningGoal

final class GoalDataAggregatorTests: XCTestCase {
    var container: NSPersistentContainer!
    var context: NSManagedObjectContext!
    var aggregator: GoalDataAggregator!

    override func setUp() {
        super.setUp()
        container = InMemoryCoreData.container()
        context = container.viewContext
        aggregator = GoalDataAggregator(viewContext: context)
    }

    override func tearDown() {
        aggregator = nil
        context = nil
        container = nil
        super.tearDown()
    }

    // MARK: - Helper Methods

    private func createEntry(
        date: Date,
        goalText: String = "Test Goal",
        category: String? = nil,
        sentiment: String? = nil,
        sentimentScore: Double = 0.0,
        urgency: String? = nil,
        timeFrame: String? = nil,
        actionType: String? = nil,
        difficulty: String? = nil,
        specificity: String? = nil
    ) {
        let entry = GoalEntry(context: context)
        entry.dateString = GoalEntry.dateStringFrom(date)
        entry.goalText = goalText
        entry.lastUpdated = date
        entry.category = category
        entry.sentiment = sentiment
        entry.sentimentScore = sentimentScore
        entry.urgency = urgency
        entry.timeFrame = timeFrame
        entry.actionType = actionType
        entry.difficulty = difficulty
        entry.specificity = specificity
        // Ensure core data recognizes the date string correctly for filtering
    }

    // MARK: - Tests

    func test_getAggregatedStats_weekPeriod_returnsCorrectCounts() async throws {
        // Create 3 entries for today
        for _ in 0..<3 {
            createEntry(date: Date())
        }
        
        // Create 2 entries for yesterday
        let yesterday = Calendar.current.date(byAdding: .day, value: -1, to: Date())!
        for _ in 0..<2 {
            createEntry(date: yesterday)
        }
        
        // Create 1 entry for 8 days ago (outside week range)
        let oldDate = Calendar.current.date(byAdding: .day, value: -8, to: Date())!
        createEntry(date: oldDate)
        
        try context.save()
        
        let stats = try await aggregator.getAggregatedStats(for: .week)
        
        // Should be 3 + 2 = 5
        XCTAssertEqual(stats.totalGoals, 5)
    }

    func test_getTopicDistribution_returnsCorrectDistribution() async throws {
        // 3 Health, 1 Work, 1 null
        createEntry(date: Date(), category: "Health")
        createEntry(date: Date(), category: "Health")
        createEntry(date: Date(), category: "Health")
        createEntry(date: Date(), category: "Work")
        createEntry(date: Date(), category: nil)
        
        try context.save()
        
        let distribution = try await aggregator.getTopicDistribution(period: .week)
        
        let healthItem = distribution.first { $0.label == "Health" }
        let workItem = distribution.first { $0.label == "Work" }
        
        XCTAssertNotNil(healthItem)
        XCTAssertEqual(healthItem?.count, 3)
        // 3/4 classified goals = 75%
        XCTAssertEqual(healthItem?.percentage, 75.0)
        
        XCTAssertNotNil(workItem)
        XCTAssertEqual(workItem?.count, 1)
        XCTAssertEqual(workItem?.percentage, 25.0)
    }
    
    func test_getSentimentTrend_returnsCorrectAverage() async throws {
        // Day 1: "积极" -> Score 1.0
        let day1 = Date()
        createEntry(date: day1, sentiment: "积极")
        
        // Day 2: "中性" -> Score 0.0
        let day2 = Calendar.current.date(byAdding: .day, value: -1, to: Date())!
        createEntry(date: day2, sentiment: "中性")
        
        try context.save()
        
        let trend = try await aggregator.getSentimentTrend(days: 7)
        
        // Overall average (1.0 + 0.0) / 2 = 0.5
        XCTAssertEqual(trend.averageSentiment, 0.5, accuracy: 0.01)
        
        // Check data points
        let day1Str = GoalEntry.dateStringFrom(day1)
        let day2Str = GoalEntry.dateStringFrom(day2)
        
        let p1 = trend.dataPoints.first { GoalEntry.dateStringFrom($0.date) == day1Str }
        let p2 = trend.dataPoints.first { GoalEntry.dateStringFrom($0.date) == day2Str }
        
        XCTAssertNotNil(p1)
        XCTAssertEqual(p1?.value ?? 0, 1.0, accuracy: 0.01)
        
        XCTAssertNotNil(p2)
        XCTAssertEqual(p2?.value ?? 0, 0.0, accuracy: 0.01)
    }
    
    func test_getWeekdayDistribution_returnsCorrectCounts() async throws {
        // Create an entry for today
        createEntry(date: Date())
        
        try context.save()
        
        let distribution = try await aggregator.getWeekdayDistribution(period: .week)
        
        // Should have 7 items
        XCTAssertEqual(distribution.count, 7)
        
        // Find today's weekday
        let calendar = Calendar.current
        let weekday = calendar.component(.weekday, from: Date())
        // Adjust to 0-6 index or however the aggregator implements it.
        // Usually 1=Sun, 2=Mon.
        // Aggregator implementation details might vary, but total count should be 1.
        
        let totalCount = distribution.reduce(0) { $0 + $1.count }
        XCTAssertEqual(totalCount, 1)
    }
    
    func test_getGoalCountTrend_returnsCorrectCounts() async throws {
        // Today: 2 goals
        createEntry(date: Date())
        createEntry(date: Date())
        
        // Yesterday: 1 goal
        let yesterday = Calendar.current.date(byAdding: .day, value: -1, to: Date())!
        createEntry(date: yesterday)
        
        try context.save()
        
        let trend = try await aggregator.getGoalCountTrend(days: 7)
        
        let todayStr = GoalEntry.dateStringFrom(Date())
        let yesterdayStr = GoalEntry.dateStringFrom(yesterday)
        
        let p1 = trend.first { GoalEntry.dateStringFrom($0.date) == todayStr }
        let p2 = trend.first { GoalEntry.dateStringFrom($0.date) == yesterdayStr }
        
        XCTAssertEqual(p1?.count, 2)
        XCTAssertEqual(p2?.count, 1)
    }

    func test_getAllDistributions_returnsCorrectCounts() async throws {
        // Create entries with various attributes
        createEntry(date: Date(), urgency: "High", timeFrame: "Short-term", actionType: "Action", difficulty: "Hard", specificity: "High")
        createEntry(date: Date(), urgency: "High", timeFrame: "Long-term", actionType: "Thinking", difficulty: "Easy", specificity: "Low")
        createEntry(date: Date(), urgency: "Low", timeFrame: "Short-term", actionType: "Action", difficulty: "Medium", specificity: "High")
        
        try context.save()
        
        // Test Urgency Distribution
        let urgencyDist = try await aggregator.getDistribution(for: .urgency, from: Date(), to: Date())
        let highUrgency = urgencyDist.first { $0.label == "High" }
        XCTAssertEqual(highUrgency?.count, 2)
        XCTAssertEqual(highUrgency?.percentage ?? 0.0, 66.6, accuracy: 0.1)
        
        // Test TimeFrame Distribution
        let timeFrameDist = try await aggregator.getDistribution(for: .timeFrame, from: Date(), to: Date())
        let shortTerm = timeFrameDist.first { $0.label == "Short-term" }
        XCTAssertEqual(shortTerm?.count, 2)
        
        // Test ActionType Distribution
        let actionTypeDist = try await aggregator.getDistribution(for: .actionType, from: Date(), to: Date())
        let action = actionTypeDist.first { $0.label == "Action" }
        XCTAssertEqual(action?.count, 2)
        
        // Test Difficulty Distribution
        let difficultyDist = try await aggregator.getDistribution(for: .difficulty, from: Date(), to: Date())
        XCTAssertEqual(difficultyDist.count, 3) // Hard, Easy, Medium
        
        // Test Specificity Distribution
        let specificityDist = try await aggregator.getDistribution(for: .specificity, from: Date(), to: Date())
        let highSpec = specificityDist.first { $0.label == "High" }
        XCTAssertEqual(highSpec?.count, 2)
    }

    func test_getCompletionRate_returnsCorrectRate() async throws {
        // 3 entries, 2 analyzed (simulated completion)
        let e1 = GoalEntry(context: context)
        e1.dateString = GoalEntry.dateStringFrom(Date())
        e1.analyzedAt = Date() // Completed
        e1.category = "Work"
        
        let e2 = GoalEntry(context: context)
        e2.dateString = GoalEntry.dateStringFrom(Date())
        e2.analyzedAt = Date() // Completed
        e2.category = "Work"
        
        let e3 = GoalEntry(context: context)
        e3.dateString = GoalEntry.dateStringFrom(Date())
        e3.analyzedAt = nil // Not completed
        e3.category = "Health"
        
        try context.save()
        
        let rateResult = try await aggregator.getCompletionRate(groupBy: .topic, period: .day)
        
        // Overall rate: 2/3 = 66.6%
        XCTAssertEqual(rateResult.overallRate, 66.6, accuracy: 0.1)
        
        // Work rate: 2/2 = 100%
        let workItem = rateResult.items.first { $0.label == "Work" }
        XCTAssertEqual(workItem?.rate, 100.0)
        
        // Health rate: 0/1 = 0%
        let healthItem = rateResult.items.first { $0.label == "Health" }
        XCTAssertEqual(healthItem?.rate, 0.0)
    }
    
    func test_getSimilarGoals_returnsCorrectMatches() async throws {
        // Create some dummy entries with embeddings
        let e1 = GoalEntry(context: context)
        e1.goalText = "Goal A"
        e1.embedding = Data([0x01, 0x02, 0x03]) // Dummy embedding for A
        
        let e2 = GoalEntry(context: context)
        e2.goalText = "Goal B"
        e2.embedding = Data([0x01, 0x02, 0x04]) // Dummy embedding for B (similar to A)
        
        let e3 = GoalEntry(context: context)
        e3.goalText = "Goal C"
        e3.embedding = Data([0x10, 0x20, 0x30]) // Dummy embedding for C (dissimilar to A)
        
        try context.save()
        
        // Assume dataA is the embedding for "Goal A"
        let dataA = Data([0x01, 0x02, 0x03])
        
        // Search similar to A (using A's embedding)
        // Should return B first, then C (or C might be 0 similarity)
        // Exclude A itself from results
        
        let results = try await aggregator.getSimilarGoals(embedding: dataA, limit: 10, excludeObjectID: e1.objectID)
        
        XCTAssertFalse(results.isEmpty)
        XCTAssertEqual(results.first?.goalText, "Goal B")
        XCTAssertGreaterThan(results.first?.similarity ?? 0, 0.8)
    }

    func test_averageConfidence_ignoresUnanalyzed() async throws {
        // 1. Analyzed Entry (High Confidence)
        let e1 = GoalEntry(context: context)
        e1.dateString = GoalEntry.dateStringFrom(Date())
        e1.analyzedAt = Date()
        e1.categoryConfidence = 0.9
        e1.sentimentScore = 0.9
        e1.category = "Work"
        e1.sentiment = "Positive"
        
        // 2. Unanalyzed Entry (Zero/Nil Confidence)
        let e2 = GoalEntry(context: context)
        e2.dateString = GoalEntry.dateStringFrom(Date())
        e2.analyzedAt = nil
        e2.categoryConfidence = 0.0
        e2.sentimentScore = 0.0
        e2.category = nil
        e2.sentiment = nil
        
        try context.save()
        
        let stats = try await aggregator.getAggregatedStats(for: .day)
        
        // Should be 0.9 (Average of 1 item), NOT 0.45 (Average of 2 items)
        XCTAssertEqual(stats.averageConfidence.topic, 0.9, accuracy: 0.01, "Should ignore unanalyzed entry for topic confidence")
        XCTAssertEqual(stats.averageConfidence.sentiment, 0.9, accuracy: 0.01, "Should ignore unanalyzed entry for sentiment confidence")
    }
}
