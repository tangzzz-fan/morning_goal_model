import Foundation

final class PromptStateService {
    static let shared = PromptStateService()
    private let defaults = UserDefaults.standard
    private let lastPromptedKey = "morning_goal_last_prompted_date"

    func hasPromptedToday() -> Bool {
        let today = GoalEntry.todayString()
        return defaults.string(forKey: lastPromptedKey) == today
    }

    func markPromptedToday() {
        defaults.set(GoalEntry.todayString(), forKey: lastPromptedKey)
    }
}
