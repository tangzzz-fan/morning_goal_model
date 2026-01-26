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

    private func createEntry(dateString: String, text: String, sentiment: String? = nil, category: String? = nil) {
        let entry = GoalEntry(context: viewContext)
        entry.dateString = dateString
        entry.goalText = text
        entry.lastUpdated = Date()
        entry.sentiment = sentiment
        entry.sentimentScore = sentiment == "积极" ? 0.9 : (sentiment == "消极" ? -0.9 : 0.0)
        entry.category = category
        entry.categoryConfidence = 0.9
        // Add required default values if any
    }
    
    private func dateString(daysAgo: Int) -> String {
        let date = Calendar.current.date(byAdding: .day, value: -daysAgo, to: Date())!
        return GoalEntry.dateStringFrom(date)
    }

    // MARK: - Tests

    func testGenerateInsights_ReturnsEmpty_WhenNoData() async {
        let insights = await engine.generateInsights(for: 7)
        XCTAssertTrue(insights.isEmpty)
    }

    func testGenerateInsights_DetectsPositiveTrend() async {
        // Create 5 positive entries in last 7 days
        for i in 0..<5 {
            createEntry(dateString: dateString(daysAgo: i), text: "Good goal \(i)", sentiment: "积极")
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
            createEntry(dateString: dateString(daysAgo: i), text: "Work \(i)", category: "工作")
        }
        try? viewContext.save()
        
        let insights = await engine.generateInsights(for: 14) // Need >7 days for balance check
        
        let balanceInsight = insights.first { $0.type == .balance }
        XCTAssertNotNil(balanceInsight)
        XCTAssertTrue(balanceInsight?.title.contains("Focus") ?? false) // "Heavy Focus"
    }
}
