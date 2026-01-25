import BackgroundTasks
import CoreData
import Foundation

/// 负责调度和处理后台数据聚合任务
final class AggregationScheduler {
    static let shared = AggregationScheduler()

    // 任务标识符 - 必须与 Info.plist 中的配置一致
    private let dailyTaskIdentifier = "com.morninggoal.dailyAggregation"
    private let weeklyTaskIdentifier = "com.morninggoal.weeklyAggregation"
    // private let refreshTaskIdentifier = "com.morninggoal.refresh_stats" // 新增 (暂时注释掉)

    private init() {}

    /// 注册后台任务
    /// 必须在 App 启动时调用 (application:didFinishLaunchingWithOptions:)
    func register() {
        // 注册每日处理任务
        BGTaskScheduler.shared.register(forTaskWithIdentifier: dailyTaskIdentifier, using: nil) { task in
            if let processingTask = task as? BGProcessingTask {
                self.handleDailyAggregation(task: processingTask)
            }
        }

        // 注册每周处理任务
        BGTaskScheduler.shared.register(forTaskWithIdentifier: weeklyTaskIdentifier, using: nil) { task in
            if let processingTask = task as? BGProcessingTask {
                self.handleWeeklyAggregation(task: processingTask)
            }
        }

        // 注册 App Refresh 任务 (暂时注释掉，因为用户当天输入频率低)
        /*
        BGTaskScheduler.shared.register(forTaskWithIdentifier: refreshTaskIdentifier, using: nil) { task in
            if let refreshTask = task as? BGAppRefreshTask {
                self.handleAppRefresh(task: refreshTask)
            }
        }
        */
    }

    /// 调度下一个后台任务
    /// 通常在 App 进入后台时调用 (sceneDidEnterBackground)
    func scheduleNextTasks() {
        scheduleDailyTask()
        scheduleWeeklyTask()
        // scheduleAppRefresh() // 暂时注释掉
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
            // print("Successfully scheduled daily aggregation for \(executionDate)")
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
            // print("Successfully scheduled weekly aggregation for \(executionDate)")
        } catch {
            print("Could not schedule weekly aggregation: \(error)")
        }
    }
    
    /*
    private func scheduleAppRefresh() {
        let request = BGAppRefreshTaskRequest(identifier: refreshTaskIdentifier)
        // 最早 30 分钟后运行
        request.earliestBeginDate = Date(timeIntervalSinceNow: 30 * 60)
        
        do {
            try BGTaskScheduler.shared.submit(request)
            // print("Scheduled app refresh task")
        } catch {
            print("Could not schedule app refresh: \(error)")
        }
    }
    */
    
    // MARK: - Task Handling

    private func handleDailyAggregation(task: BGProcessingTask) {
        scheduleDailyTask() // 重新调度下一次任务

        let context = DataController.shared.container.newBackgroundContext()

        task.expirationHandler = {
            print("Daily aggregation task expiring...")
        }

        context.perform {
            let aggregator = GoalDataAggregator(viewContext: context)

            // 执行昨天的日统计
            let yesterday = Calendar.current.date(byAdding: .day, value: -1, to: Date())!
            let range = aggregator.getDateRange(for: .day, from: yesterday)

            // 计算并缓存统计数据
            let stats = aggregator.getAggregatedStats(from: range.from, to: range.to)
            print("Daily Aggregation Completed: \(stats.totalGoals) goals processed for \(range.from)")

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

    /*
    private func handleAppRefresh(task: BGAppRefreshTask) {
        scheduleAppRefresh() // 重新调度

        task.expirationHandler = {
            print("App refresh task expired")
            task.setTaskCompleted(success: false)
        }

        let context = DataController.shared.container.newBackgroundContext()
        let aggregator = GoalDataAggregator(viewContext: context)

        // 使用 Task 进行异步操作
        Task {
            do {
                // 简单聚合一下当日数据，确保缓存热度
                let today = Date()
                _ = aggregator.getAggregatedStats(from: today, to: today)

                // 预生成洞察
                let engine = InsightEngine(context: context)
                _ = engine.generateInsights(for: 7)

                print("App Refresh Completed")
                task.setTaskCompleted(success: true)
            }
        }
    }
    */
}
