import CoreData
import Foundation

final class InsightEngine {
    private let aggregator: GoalDataAggregator
    private let viewContext: NSManagedObjectContext

    init(viewContext: NSManagedObjectContext) {
        self.viewContext = viewContext
        self.aggregator = GoalDataAggregator(viewContext: viewContext)
    }

    /// Main entry point: Generates the "Whisper" for a newly committed goal
    /// Returns a single, highest-priority insight
    func generateDailyInsight(for goal: GoalEntry) async -> Insight? {
        // Run analysis in parallel or sequence based on cost
        // For v1, we run a sequence of checks from High Priority to Low Priority

        // 1. Milestone Check (Highest Priority)
        let goalDate = GoalEntry.dateFrom(goal.dateString) ?? Date()

        if let milestone = checkMilestones(referenceDate: goalDate) {
            return milestone
        }

        // 2. Urgent/Important Warning
        if let urgencyInsight = checkUrgency(goal: goal) {
            return urgencyInsight
        }

        // 3. Balance Check (Weekly)
        if let balance = checkBalance(referenceDate: goalDate) {
            return balance
        }

        // 4. Pattern Recognition (Day of Week)
        if let pattern = checkWeekdayPattern(for: goal, referenceDate: goalDate) {
            return pattern
        }

        // 5. Trend (Sentiment)
        if let trend = checkSentimentTrend(referenceDate: goalDate) {
            return trend
        }

        // 6. Generic Encouragement (Fallback)
        return Insight(
            type: .encouragement,
            title: "Well Done",
            description: "Goal recorded. Make it happen!",
            priority: 1,
            actionSuggestion: nil
        )
    }

    /// Generates a list of insights for a specific period (e.g., last 7 days)
    /// Used by InsightViewModel for the Stats Tab
    func generateInsights(for days: Int) async -> [Insight] {
        var insights: [Insight] = []

        // 1. Trend Analysis
        let trend = aggregator.getSentimentTrend(days: days, referenceDate: Date())
        if trend.averageSentiment > 0.6 {
            insights.append(Insight(
                type: .trend,
                title: "Positive Streak",
                description: "You've been very positive lately!",
                priority: 3,
                actionSuggestion: nil
            ))
        } else if trend.averageSentiment < -0.3 {
            insights.append(Insight(
                type: .trend,
                title: "Tough Times?",
                description: "Seems like a challenging period. Be kind to yourself.",
                priority: 3,
                actionSuggestion: nil
            ))
        }

        // 2. Balance Analysis (if period is long enough)
        if days >= 7 {
            let distribution = aggregator.getTopicDistribution(period: .week, referenceDate: Date())
            if let top = distribution.first, top.percentage > 70 {
                insights.append(Insight(
                    type: .balance,
                    title: "Heavy Focus",
                    description: "Most goals are '\(top.label)'.",
                    priority: 4,
                    actionSuggestion: "Try diversifying?"
                ))
            }
        }

        return insights
    }

    // MARK: - Analyzers

    private func checkMilestones(referenceDate: Date) -> Insight? {
        // Calculate lifetime count up to this goal
        // We use a distant past date to ensure we catch everything
        let stats = aggregator.getAggregatedStats(from: .distantPast, to: referenceDate)
        let count = stats.totalGoals

        if count == 1 {
            return Insight(
                type: .encouragement,
                title: "First Step",
                description: "Your journey begins today.",
                priority: 5,
                actionSuggestion: nil
            )
        }
        if count % 10 == 0 || count == 30 || count == 100 {
            return Insight(
                type: .encouragement,
                title: "Milestone Reversed",
                description: "You've recorded \(count) goals!",
                priority: 5,
                actionSuggestion: nil
            )
        }
        return nil
    }

    private func checkUrgency(goal: GoalEntry) -> Insight? {
        if goal.effectiveUrgency == "high" {
            // Only warn if they have TOO MANY high urgency goals today?
            // For now, simple reflection
            return Insight(
                type: .achievability,
                title: "High Urgency",
                description: "Focus is key. Tackle this first.",
                priority: 3,
                actionSuggestion: nil
            )
        }
        return nil
    }

    private func checkBalance(referenceDate: Date) -> Insight? {
        // Only run on Sundays? Or every day?
        // Let's run it if today has > 3 goals already recorded for the week
        // Simply check the last 7 days distribution
        let distribution = aggregator.getTopicDistribution(period: .week, referenceDate: referenceDate)
        guard let topItem = distribution.first else { return nil }

        // If one topic > 80% and count > 5
        let total = distribution.reduce(0) { $0 + $1.count }
        if total > 5 && topItem.percentage > 80.0 {
            return Insight(
                type: .balance,
                title: "Heavy Focus",
                description: "80% of your goals this week are '\(topItem.label)'.",
                priority: 4,
                actionSuggestion: "Consider a break or a different focus?"
            )
        }
        return nil
    }

    private func checkWeekdayPattern(for goal: GoalEntry, referenceDate: Date) -> Insight? {
        // e.g., "Monday is usually Work day"
        let distribution = aggregator.getWeekdayDistribution(period: .month) // TODO: Add referenceDate support to aggregator if needed
        // Check if today (e.g. Monday) matches the user's historical Peak for this Category
        // This is complex. Simplified:
        // "You usually focus on [Category] on [Weekday]s."

        let calendar = Calendar.current
        guard let goalDate = GoalEntry.dateFrom(goal.dateString) else { return nil }
        let weekday = calendar.component(.weekday, from: goalDate)

        // Fetch all goals for this weekday in last month
        // This requires a custom fetch. For MVP, skip complex query.
        return nil
    }

    private func checkSentimentTrend(referenceDate: Date) -> Insight? {
        let trend = aggregator.getSentimentTrend(days: 7, referenceDate: referenceDate)
        if trend.averageSentiment > 0.6 {
            return Insight(
                type: .trend,
                title: "Positive Streak",
                description: "You've been very positive this week!",
                priority: 2,
                actionSuggestion: nil
            )
        }
        return nil
    }
}
