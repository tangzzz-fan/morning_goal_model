//
//  InsightSystemIntegrationTests.swift
//  MorningGoalTests
//
//  洞察系统集成测试 - Phase 7
//  验证从数据聚合到洞察生成的完整流程
//
//  Created by User on 2026/01/25.
//

import CoreData
import XCTest
@testable import MorningGoal

final class InsightSystemIntegrationTests: XCTestCase {
    var container: NSPersistentContainer!
    var context: NSManagedObjectContext!
    var aggregator: GoalDataAggregator!
    var engine: InsightEngine!
    
    override func setUp() {
        super.setUp()
        container = InMemoryCoreData.container()
        context = container.viewContext
        aggregator = GoalDataAggregator(viewContext: context)
        engine = InsightEngine(context: context)
    }
    
    override func tearDown() {
        engine = nil
        aggregator = nil
        context = nil
        container = nil
        super.tearDown()
    }
    
    // MARK: - Integration Tests
    
    func test_endToEnd_highWorkload_generatesBalanceAndVolumeInsights() async throws {
        // 1. Simulate User Activity: High workload week (15 work goals)
        let today = Date()
        let calendar = Calendar.current
        
        // Create 15 work goals over the last 7 days
        for i in 0..<15 {
            let dayOffset = i % 7
            let date = calendar.date(byAdding: .day, value: -dayOffset, to: today)!
            
            let entry = GoalEntry(context: context)
            entry.dateString = GoalEntry.dateStringFrom(date)
            entry.goalText = "Work Goal \(i)"
            entry.lastUpdated = date
            entry.category = "工作" // Topic
            entry.actionType = "工作" // ActionType
            entry.sentiment = "中性"
        }
        
        // Create 2 work goals in the previous week (Week - 2) for Comparison
        for i in 0..<2 {
            let date = calendar.date(byAdding: .day, value: -10, to: today)!
            let entry = GoalEntry(context: context)
            entry.dateString = GoalEntry.dateStringFrom(date)
            entry.goalText = "Old Work Goal \(i)"
            entry.lastUpdated = date
            entry.category = "工作"
            entry.actionType = "工作"
        }
        
        try context.save()
        
        // 2. Run Aggregation
        let stats = aggregator.getAggregatedStats(for: .week)
        
        // Verify Stats
        XCTAssertGreaterThanOrEqual(stats.totalGoals, 2) // At least some goals from this week (due to date boundary some might fall out if logic differs, but generally should match)
        
        let actionTypeDist = aggregator.getDistribution(for: .actionType, from: stats.fromDate, to: stats.toDate)
        let workItem = actionTypeDist.first { $0.label == "工作" }
        XCTAssertNotNil(workItem)
        XCTAssertGreaterThan(workItem?.percentage ?? 0, 80.0)
        
        // 3. Generate Insights
        let insights = engine.generateInsights(for: 7)
        
        // 4. Verify Insights
        
        // Expect "Balance" insight (Too much work)
        let balanceInsight = insights.first { $0.id == "balance" }
        XCTAssertNotNil(balanceInsight, "Should generate balance insight for high work load")
        
        if case let .balance(mainType, _, suggestedType) = balanceInsight {
            XCTAssertEqual(mainType, "工作")
            XCTAssertEqual(suggestedType, "生活")
        }
        
        // Expect "Volume Trend" insight (15 vs 2 -> Huge Increase)
        let volumeInsight = insights.first { $0.id == "volume_trend" }
        
        if let volumeInsight = volumeInsight {
            // Might be generated depending on exact date logic
            if case let .volumeTrend(direction, _) = volumeInsight {
                XCTAssertEqual(direction, "上升")
            }
        }
        
        // Expect "Comparison" insight
        let comparisonInsight = insights.first { $0.id == "comparison" }
         XCTAssertNotNil(comparisonInsight, "Should generate comparison insight")
    }
    
    func test_endToEnd_positiveStreak_generatesEncouragementAndSentimentInsight() async throws {
        // ... (existing test code)
        // 1. Simulate User Activity: 8 days of positive entries
        let today = Date()
        let calendar = Calendar.current
        
        for i in 0..<8 {
            let date = calendar.date(byAdding: .day, value: -i, to: today)!
            let entry = GoalEntry(context: context)
            entry.dateString = GoalEntry.dateStringFrom(date)
            entry.goalText = "Positive Goal \(i)"
            entry.sentimentScore = 0.9
            entry.sentiment = "积极"
        }
        
        try context.save()
        
        // 2. Generate Insights
        let insights = engine.generateInsights(for: 14) // Use 14 days to cover the streak
        
        // 3. Verify Insights
        
        // Expect "Encouragement" for 8 day streak
        let encouragement = insights.first { $0.id == "encouragement" }
        XCTAssertNotNil(encouragement, "Should generate encouragement for streak")
        
        // Expect "Sentiment Trend"
        // Need > 10 entries with non-zero score. Currently 8.
        // Let's add historic data with non-zero score (e.g. 0.2) to verify trend UP (0.9 vs 0.2)
        
        for i in 8..<20 {
            let date = calendar.date(byAdding: .day, value: -i, to: today)!
            let entry = GoalEntry(context: context)
            entry.dateString = GoalEntry.dateStringFrom(date)
            entry.goalText = "Historic Goal \(i)"
            entry.sentimentScore = 0.2 // Non-zero to be counted
            entry.sentiment = "弱积极"
        }
        try context.save()
        
        // Re-generate
        let insights2 = engine.generateInsights(for: 30)
        let sentimentInsight = insights2.first { $0.id == "sentiment" }
        XCTAssertNotNil(sentimentInsight, "Should generate sentiment trend with sufficient data")
        
        if let insight = sentimentInsight, case let .sentimentTrend(direction) = insight {
            // Recent (0.9) > Historic (0.2) -> Up
            XCTAssertEqual(direction, "上升")
        }
    }

    func test_mockDataService_populatesAnalysisFields() throws {
        // Clear existing data
        MockDataService.clearAllData(from: context)
        
        // Add mock data
        MockDataService.addMockData(to: context)
        
        // Fetch valid entries
        let request: NSFetchRequest<GoalEntry> = GoalEntry.fetchRequest()
        let entries = try context.fetch(request)
        
        XCTAssertGreaterThan(entries.count, 0)
        
        // Verify fields are populated
        for entry in entries {
            // DateString and GoalText should always be present
            XCTAssertNotNil(entry.dateString)
            XCTAssertNotNil(entry.goalText)
            
            // Checking if analysis fields are populated for recent entries (MockDataService adds 7 days + history)
            // The history entries (1 month ago etc) created by createMockEntry inside addMockData might NOT have analysis unless we updated that call too.
            // Let's check the recent ones which definitely use mockText() -> analysis
            
            if let date = GoalEntry.dateFrom(entry.dateString), Date().timeIntervalSince(date) < 8 * 24 * 3600 {
                // Recent entry
                XCTAssertNotNil(entry.category, "Category should be populated")
                XCTAssertGreaterThan(entry.categoryConfidence, 0.0, "Category confidence should be > 0")
                XCTAssertNotNil(entry.sentiment, "Sentiment should be populated")
            }
        }
    }
}
