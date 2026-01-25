import BackgroundTasks
import CoreData
import Foundation

/// 负责调度和处理后台数据聚合任务
final class AggregationScheduler {
    static let shared = AggregationScheduler()

    // 任务标识符 - 必须与 Info.plist 中的配置一致
    private let dailyTaskIdentifier = "com.morninggoal.dailyAggregation"
    private let weeklyTaskIdentifier = "com.morninggoal.weeklyAggregation"

    private init() {}

    /// 注册后台任务
    /// 必须在 App 启动时调用 (application:didFinishLaunchingWithOptions:)
    func register() {
        BGTaskScheduler.shared.register(forTaskWithIdentifier: dailyTaskIdentifier, using: nil) { task in
            if let processingTask = task as? BGProcessingTask {
                self.handleDailyAggregation(task: processingTask)
            }
        }

        BGTaskScheduler.shared.register(forTaskWithIdentifier: weeklyTaskIdentifier, using: nil) { task in
            if let processingTask = task as? BGProcessingTask {
                self.handleWeeklyAggregation(task: processingTask)
            }
        }
    }

    /// 调度下一个后台任务
    /// 通常在 App 进入后台时调用 (sceneDidEnterBackground)
    func scheduleNextTasks() {
        scheduleDailyTask()
        scheduleWeeklyTask()
    }

    // MARK: - Scheduling Logic

    private func scheduleDailyTask() {
        let request = BGProcessingTaskRequest(identifier: dailyTaskIdentifier)
        request.requiresNetworkConnectivity = false
        request.requiresExternalPower = false

        // 调度到明天凌晨 2 点左右执行
        let today = Calendar.current.startOfDay(for: Date())
        guard let tomorrow = Calendar.current.date(byAdding: .day, value: 1, to: today),
              let executionDate = Calendar.current.date(bySettingHour: 2, minute: 0, second: 0, of: tomorrow)
        else {
            return
        }

        request.earliestBeginDate = executionDate

        do {
            try BGTaskScheduler.shared.submit(request)
            print("Successfully scheduled daily aggregation for \(executionDate)")
        } catch {
            print("Could not schedule daily aggregation: \(error)")
        }
    }

    private func scheduleWeeklyTask() {
        let request = BGProcessingTaskRequest(identifier: weeklyTaskIdentifier)
        request.requiresNetworkConnectivity = false
        request.requiresExternalPower = false

        // 调度到下周一凌晨 3 点执行
        let today = Calendar.current.startOfDay(for: Date())
        let weekday = Calendar.current.component(.weekday, from: today)
        // 计算距离下周一的天数 (1=周日, 2=周一)
        var daysUntilMonday = 9 - weekday
        if daysUntilMonday > 7 { daysUntilMonday -= 7 }

        guard let nextMonday = Calendar.current.date(byAdding: .day, value: daysUntilMonday, to: today),
              let executionDate = Calendar.current.date(bySettingHour: 3, minute: 0, second: 0, of: nextMonday)
        else {
            return
        }

        request.earliestBeginDate = executionDate

        do {
            try BGTaskScheduler.shared.submit(request)
            print("Successfully scheduled weekly aggregation for \(executionDate)")
        } catch {
            print("Could not schedule weekly aggregation: \(error)")
        }
    }

    // MARK: - Task Handling

    private func handleDailyAggregation(task: BGProcessingTask) {
        scheduleDailyTask() // 重新调度下一次任务

        let context = DataController.shared.container.newBackgroundContext()

        task.expirationHandler = {
            // 任务即将过期时的清理逻辑
            // 实际上 Core Data 操作很难中断，这里主要做标记
            print("Daily aggregation task expiring...")
        }

        context.perform {
            let aggregator = GoalDataAggregator(viewContext: context)

            // 执行昨天的日统计
            let yesterday = Calendar.current.date(byAdding: .day, value: -1, to: Date())!
            let range = aggregator.getDateRange(for: .day, from: yesterday)

            // 计算统计数据 (这里只是执行计算，实际上可能需要将结果保存到数据库或缓存中)
            // 目前 AggregatedStats 是结构体，如果需要持久化，需要扩展数据模型
            // 这里我们模拟一个计算过程，并打印日志
            let stats = aggregator.getAggregatedStats(from: range.from, to: range.to)

            print("Daily Aggregation Completed: \(stats.totalGoals) goals processed for \(range.from)")

            // 模拟保存结果 (实际项目中应该保存到 DailyStats 实体)
            // saveDailyStats(stats, context: context)

            task.setTaskCompleted(success: true)
        }
    }

    private func handleWeeklyAggregation(task: BGProcessingTask) {
        scheduleWeeklyTask() // 重新调度下一次任务

        let context = DataController.shared.container.newBackgroundContext()

        task.expirationHandler = {
            print("Weekly aggregation task expiring...")
        }

        context.perform {
            let aggregator = GoalDataAggregator(viewContext: context)

            // 执行上周的统计
            let lastWeek = Calendar.current.date(byAdding: .day, value: -7, to: Date())!
            let range = aggregator.getDateRange(for: .week, from: lastWeek)

            let stats = aggregator.getAggregatedStats(from: range.from, to: range.to)

            print("Weekly Aggregation Completed: \(stats.totalGoals) goals processed for week of \(range.from)")

            task.setTaskCompleted(success: true)
        }
    }
}
