import Combine
import CoreData
import Foundation

@MainActor
class InsightViewModel: ObservableObject {
    // MARK: - Published Properties

    @Published var selectedPeriod: AggregationPeriod = .week {
        didSet {
            loadData()
        }
    }

    @Published var stats: AggregatedStats?
    @Published var topicDistribution: [DistributionItem] = []
    @Published var sentimentTrend: SentimentTrendResult?
    @Published var goalCountTrend: [TrendDataPoint] = []
    @Published var weekdayDistribution: [DistributionItem] = []

    @Published var isLoading = false
    @Published var errorMessage: String?

    // MARK: - Dependencies

    private let aggregator: GoalDataAggregator

    // MARK: - Initialization

    init(context: NSManagedObjectContext) {
        self.aggregator = GoalDataAggregator(viewContext: context)
    }

    // MARK: - Public Methods

    func loadData() {
        isLoading = true
        errorMessage = nil

        Task {
            do {
                let days = selectedPeriod.defaultDays

                // 并行加载所有数据以提高性能
                async let statsResult = aggregator.getAggregatedStats(for: selectedPeriod)
                async let topicResult = aggregator.getTopicDistribution(period: selectedPeriod)
                async let sentimentResult = aggregator.getSentimentTrend(days: days)
                async let countResult = aggregator.getGoalCountTrend(days: days)
                async let weekdayResult = aggregator.getWeekdayDistribution(period: selectedPeriod)

                // 等待所有任务完成
                let (newStats, newTopics, newSentiment, newCounts, newWeekdays) = try await(
                    statsResult,
                    topicResult,
                    sentimentResult,
                    countResult,
                    weekdayResult
                )

                // 更新 UI 状态
                self.stats = newStats
                self.topicDistribution = newTopics
                self.sentimentTrend = newSentiment
                self.goalCountTrend = newCounts
                self.weekdayDistribution = newWeekdays
                self.isLoading = false

            } catch {
                self.errorMessage = "加载数据失败: \(error.localizedDescription)"
                self.isLoading = false
            }
        }
    }

    /// 刷新数据
    func refresh() {
        loadData()
    }
}
