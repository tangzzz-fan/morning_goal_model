import XCTest
import CoreData
@testable import MorningGoal

final class GoalFlowTests: XCTestCase {
    func test_flow_onboarding_commit_and_save_today() {
        let container = InMemoryCoreData.container()
        let ctx = container.viewContext

        var settings = UserSettings.fetchOrCreate(in: ctx)
        settings.morningStartHour = 7
        settings.morningStartMinute = 0
        settings.morningEndHour = 9
        settings.morningEndMinute = 0
        settings.analyticsOptIn = false
        try? ctx.save()

        settings = UserSettings.fetchOrCreate(in: ctx)
        settings.committed = true
        try? ctx.save()

        let key = GoalEntry.todayString()
        let req = NSFetchRequest<GoalEntry>(entityName: "GoalEntry")
        req.predicate = NSPredicate(format: "dateString == %@", key)
        XCTAssertNil(try? ctx.fetch(req).first)

        let entry = GoalEntry(context: ctx)
        entry.dateString = key
        entry.goalText = "focus"
        entry.lastUpdated = Date()
        try? ctx.save()

        let count = StreakService.count(in: ctx)
        XCTAssertEqual(count, 1)
        let rows = try? ctx.fetch(req)
        XCTAssertEqual(rows?.count, 1)
    }
}
