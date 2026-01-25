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
}
