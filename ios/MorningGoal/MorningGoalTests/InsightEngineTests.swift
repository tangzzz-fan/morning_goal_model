import XCTest
import CoreData
@testable import MorningGoal

final class InsightEngineTests: XCTestCase {
    var container: NSPersistentContainer!
    var viewContext: NSManagedObjectContext!
    var engine: InsightEngine!

    override func setUp() {
        super.setUp()
        // Use in-memory store
        container = NSPersistentContainer(name: "MorningGoalModel", managedObjectModel: CoreDataModelBuilder.makeModel())
        let description = NSPersistentStoreDescription()
        description.type = NSInMemoryStoreType
        container.persistentStoreDescriptions = [description]
        
        container.loadPersistentStores { _, error in
            if let error = error {
                fatalError("Failed to load in-memory store: \(error)")
            }
        }
        
        viewContext = container.viewContext
        engine = InsightEngine(viewContext: viewContext)
    }

    override func tearDown() {
        engine = nil
        viewContext = nil
        container = nil
        super.tearDown()
    }

    // MARK: - Helper

    private func createEntry(dateString: String, text: String, sentiment: String? = nil, category: String? = nil, urgency: String? = nil) -> GoalEntry {
        let entry = GoalEntry(context: viewContext)
        entry.dateString = dateString
        entry.goalText = text
        entry.lastUpdated = Date()
        entry.sentiment = sentiment
        entry.sentimentScore = sentiment == "积极" ? 0.9 : (sentiment == "消极" ? -0.9 : 0.0)
        entry.category = category
        entry.categoryConfidence = 0.9
        entry.urgency = urgency
        // entry.effectiveUrgency = urgency // Mock computed property if needed, or rely on logic
        return entry
    }
    
    private func dateString(daysAgo: Int) -> String {
        let date = Calendar.current.date(byAdding: .day, value: -daysAgo, to: Date())!
        return GoalEntry.dateStringFrom(date)
    }

    // MARK: - Tests: generateInsights (List)

    func testGenerateInsights_ReturnsEmpty_WhenNoData() async {
        let insights = await engine.generateInsights(for: 7)
        XCTAssertTrue(insights.isEmpty)
    }

    func testGenerateInsights_DetectsPositiveTrend() async {
        // Create 5 positive entries in last 7 days
        for i in 0..<5 {
            _ = createEntry(dateString: dateString(daysAgo: i), text: "Good goal \(i)", sentiment: "积极")
        }
        try? viewContext.save()
        
        let insights = await engine.generateInsights(for: 7)
        
        // Should find at least one insight
        XCTAssertFalse(insights.isEmpty)
        
        // Should contain a Trend insight
        let trendInsight = insights.first { $0.type == .trend }
        XCTAssertNotNil(trendInsight)
        XCTAssertTrue(trendInsight?.title.contains("Positive") ?? false)
    }
    
    func testGenerateInsights_DetectsBalanceIssue_WhenMonotonous() async {
        // Create 10 entries all with same category "Work"
        for i in 0..<10 {
            _ = createEntry(dateString: dateString(daysAgo: i), text: "Work \(i)", category: "工作")
        }
        try? viewContext.save()
        
        let insights = await engine.generateInsights(for: 14) // Need >7 days for balance check
        
        let balanceInsight = insights.first { $0.type == .balance }
        XCTAssertNotNil(balanceInsight)
        XCTAssertTrue(balanceInsight?.title.contains("Focus") ?? false) // "Heavy Focus"
    }

    // MARK: - Tests: generateDailyInsight (Single "Whisper")

    func testGenerateDailyInsight_ReturnsFallback_WhenNoConditionsMet() async {
        // Given: A normal goal with no history
        // Use a date that won't trigger milestone (assuming milestone checks history count)
        // But wait, if history is 0, adding 1 makes count 1, which IS a milestone.
        // So we need to seed some history to avoid "First Step" milestone (count == 1)
        // And avoid 10, 30, 100. Let's make total count 5.
        
        for i in 1...4 {
            _ = createEntry(dateString: dateString(daysAgo: i), text: "Old goal \(i)")
        }
        
        let currentEntry = createEntry(dateString: dateString(daysAgo: 0), text: "Today's goal")
        try? viewContext.save()
        
        // When
        let insight = await engine.generateDailyInsight(for: currentEntry)
        
        // Then
        XCTAssertNotNil(insight)
        XCTAssertEqual(insight?.type, .encouragement)
        XCTAssertEqual(insight?.title, "Well Done")
    }

    func testGenerateDailyInsight_ReturnsMilestone_AtSpecificCounts() async {
        // Test count = 1 (First Step)
        let firstEntry = createEntry(dateString: dateString(daysAgo: 0), text: "First")
        try? viewContext.save()
        
        let insight1 = await engine.generateDailyInsight(for: firstEntry)
        XCTAssertEqual(insight1?.title, "First Step")
        
        // Clear for next test or just accumulate
        // Let's accumulate to reach 10. Currently 1. Need 9 more.
        for i in 1...8 {
            _ = createEntry(dateString: dateString(daysAgo: 1), text: "Filler \(i)")
        }
        // Now count is 9. Next one is 10.
        let tenthEntry = createEntry(dateString: dateString(daysAgo: 0), text: "Tenth")
        try? viewContext.save()
        
        let insight10 = await engine.generateDailyInsight(for: tenthEntry)
        XCTAssertEqual(insight10?.title, "Milestone Reversed")
        XCTAssertTrue(insight10?.description.contains("10") ?? false)
    }

    func testGenerateDailyInsight_PrioritizesMilestoneOverUrgency() async {
        // Scenario: 10th goal (Milestone) AND High Urgency.
        // Milestone (Priority 5) should beat Urgency (Priority 3).
        
        // 1. Setup 9 existing goals
        for i in 1...9 {
            _ = createEntry(dateString: dateString(daysAgo: 1), text: "Old \(i)")
        }
        
        // 2. Create 10th goal with High Urgency
        let urgentGoal = createEntry(dateString: dateString(daysAgo: 0), text: "Urgent 10th", urgency: "high")
        try? viewContext.save()
        
        // 3. Generate
        let insight = await engine.generateDailyInsight(for: urgentGoal)
        
        // 4. Verify
        XCTAssertEqual(insight?.type, .encouragement, "Should be Milestone type (Encouragement)")
        XCTAssertEqual(insight?.title, "Milestone Reversed", "Should be Milestone title")
        // Should NOT be "High Urgency"
    }
    
    func testGenerateDailyInsight_PrioritizesUrgencyOverBalance() async {
        // Scenario: High Urgency (Priority 3) vs Balance Issue (Priority 4? Wait)
        // Let's check InsightEngine.swift priorities:
        // Milestone: 5
        // Urgency: 3
        // Balance: 4
        // Pattern: ?
        // Trend: 2
        
        // Wait, in InsightEngine.swift:
        // 1. Milestone (Returns if found)
        // 2. Urgency (Returns if found)
        // 3. Balance (Returns if found)
        
        // So strict order is: Milestone -> Urgency -> Balance.
        // Even if Balance has Priority 4 and Urgency has Priority 3 in struct, 
        // the CODE returns Urgency FIRST.
        // So Urgency > Balance in execution order.
        
        // Setup:
        // Create condition for Balance Warning: >5 goals in week, >80% same topic.
        // Let's create 6 "Work" goals in past 6 days.
        for i in 1...6 {
            _ = createEntry(dateString: dateString(daysAgo: i), text: "Work \(i)", category: "工作")
        }
        
        // Create today's goal: "Work 7" AND High Urgency.
        // Total 7 goals, all work. >80% work. -> Balance Warning.
        // Also High Urgency. -> Urgency Warning.
        // Milestone count = 7 (not 1, 10, 30...). -> No Milestone.
        
        let urgentWorkGoal = createEntry(dateString: dateString(daysAgo: 0), text: "Urgent Work", category: "工作", urgency: "high")
        try? viewContext.save()
        
        let insight = await engine.generateDailyInsight(for: urgentWorkGoal)
        
        // Expectation: Code checks Urgency BEFORE Balance.
        XCTAssertEqual(insight?.type, .achievability) // Urgency type
        XCTAssertEqual(insight?.title, "High Urgency")
    }

    func testGenerateDailyInsight_ReturnsBalanceWarning_Thresholds() async {
        // Thresholds: Total > 5 AND Top% > 80%.
        
        // Case 1: 5 goals, 100% same topic. (Total 5 not > 5? Code says total > 5)
        // Let's check code: if total > 5 && topItem.percentage > 80.0
        
        // Setup 4 old goals + 1 new goal = 5 total. Should NOT trigger.
        for i in 1...4 {
             _ = createEntry(dateString: dateString(daysAgo: i), text: "Work \(i)", category: "工作")
        }
        let goal5 = createEntry(dateString: dateString(daysAgo: 0), text: "Work 5", category: "工作")
        try? viewContext.save()
        
        let insight5 = await engine.generateDailyInsight(for: goal5)
        // Should fallback or trend, but NOT Balance
        XCTAssertNotEqual(insight5?.type, .balance)
        
        // Case 2: 6 goals, 100% same topic. Should trigger.
        let goal6 = createEntry(dateString: dateString(daysAgo: 0), text: "Work 6", category: "工作")
        try? viewContext.save()
        // Note: we need to reset or manage context. 
        // But here we just added goal6. Now we have 5+1=6 goals (Wait, goal5 is still there).
        // Total 6 goals. All Work.
        
        let insight6 = await engine.generateDailyInsight(for: goal6)
        XCTAssertEqual(insight6?.type, .balance)
        XCTAssertEqual(insight6?.title, "Heavy Focus")
    }
    
    func testGenerateDailyInsight_ReturnsTrend_WhenPositive() async {
        // Setup: No Milestone, No Urgency, No Balance (mix categories).
        // But Positive Sentiment > 0.6
        
        // 3 Positive Work, 3 Positive Health.
        // Total 6. Balance: 50/50 (No dominance). 
        // Sentiment: All Positive.
        
        for i in 1...3 {
             _ = createEntry(dateString: dateString(daysAgo: i), text: "Work \(i)", sentiment: "积极", category: "工作")
        }
        for i in 4...6 {
             _ = createEntry(dateString: dateString(daysAgo: i), text: "Health \(i)", sentiment: "积极", category: "健康")
        }
        
        let todayGoal = createEntry(dateString: dateString(daysAgo: 0), text: "Today", sentiment: "积极", category: "工作")
        try? viewContext.save()
        
        let insight = await engine.generateDailyInsight(for: todayGoal)
        
        XCTAssertEqual(insight?.type, .trend)
        XCTAssertEqual(insight?.title, "Positive Streak")
    }
}
