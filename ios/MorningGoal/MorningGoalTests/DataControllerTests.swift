import XCTest
import CoreData
@testable import MorningGoal

final class DataControllerTests: XCTestCase {
    func test_user_settings_defaults() {
        let container = InMemoryCoreData.container()
        let ctx = container.viewContext
        let settings = UserSettings.fetchOrCreate(in: ctx)
        XCTAssertEqual(settings.morningStartHour, 7)
        XCTAssertEqual(settings.morningEndHour, 9)
        XCTAssertEqual(settings.analyticsOptIn, false)
        XCTAssertEqual(settings.committed, false)
    }

    func test_goalentry_unique_per_day() {
        let container = InMemoryCoreData.container()
        let ctx = container.viewContext
        let key = GoalEntry.todayString()
        let entry1 = GoalEntry(context: ctx)
        entry1.dateString = key
        entry1.goalText = "a"
        entry1.lastUpdated = Date()
        let entry2 = GoalEntry(context: ctx)
        entry2.dateString = key
        entry2.goalText = "b"
        entry2.lastUpdated = Date()
        do { try ctx.save() } catch {}
        let req = NSFetchRequest<GoalEntry>(entityName: "GoalEntry")
        req.predicate = NSPredicate(format: "dateString == %@", key)
        let rows = try? ctx.fetch(req)
        XCTAssertEqual(rows?.count, 1)
    }
}
