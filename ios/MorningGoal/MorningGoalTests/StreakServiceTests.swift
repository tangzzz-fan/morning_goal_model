import XCTest
import CoreData
@testable import MorningGoal

final class StreakServiceTests: XCTestCase {
    func test_streak_counts_consecutive_days() {
        let container = InMemoryCoreData.container()
        let ctx = container.viewContext
        let cal = Calendar(identifier: .gregorian)
        for i in 0..<5 {
            guard let date = cal.date(byAdding: .day, value: -i, to: Date()) else { continue }
            let key = GoalEntry.dateStringFrom(date)
            let entry = GoalEntry(context: ctx)
            entry.dateString = key
            entry.goalText = "x"
            entry.lastUpdated = date
        }
        try? ctx.save()
        let count = StreakService.count(in: ctx)
        XCTAssertEqual(count, 5)
    }

    func test_streak_stops_on_gap() {
        let container = InMemoryCoreData.container()
        let ctx = container.viewContext
        let cal = Calendar(identifier: .gregorian)
        let today = Date()
        for i in [0, 2, 3] {
            guard let date = cal.date(byAdding: .day, value: -i, to: today) else { continue }
            let key = GoalEntry.dateStringFrom(date)
            let entry = GoalEntry(context: ctx)
            entry.dateString = key
            entry.goalText = "x"
            entry.lastUpdated = date
        }
        try? ctx.save()
        let count = StreakService.count(in: ctx)
        XCTAssertEqual(count, 1)
    }

    func test_streak_counts_yesterday_if_today_missing() {
        let container = InMemoryCoreData.container()
        let ctx = container.viewContext
        let cal = Calendar(identifier: .gregorian)
        // Add 5 days ending yesterday (i=1 to 5)
        for i in 1...5 {
            guard let date = cal.date(byAdding: .day, value: -i, to: Date()) else { continue }
            let key = GoalEntry.dateStringFrom(date)
            let entry = GoalEntry(context: ctx)
            entry.dateString = key
            entry.goalText = "x"
            entry.lastUpdated = date
        }
        try? ctx.save()
        let count = StreakService.count(in: ctx)
        XCTAssertEqual(count, 5)
    }
}
