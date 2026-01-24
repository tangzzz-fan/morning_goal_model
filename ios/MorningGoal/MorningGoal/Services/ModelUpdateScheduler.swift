import BackgroundTasks
import CoreData
import Foundation
import UIKit

/// 模型更新调度器
/// 负责在合适的时机触发设备端模型训练
class ModelUpdateScheduler {
    static let shared = ModelUpdateScheduler()

    private let analysisService: AnalysisService
    private let taskIdentifier = "com.morninggoal.model-update"

    // 更新条件阈值
    private let minCorrectionsForUpdate = 10
    private let minHoursBetweenUpdates = 24.0

    private init() {
        analysisService = RuleBasedAnalysisService()
    }

    // MARK: - 注册后台任务

    func registerBackgroundTask() {
        BGTaskScheduler.shared.register(
            forTaskWithIdentifier: taskIdentifier,
            using: nil
        ) { task in
            if let processingTask = task as? BGProcessingTask {
                self.handleModelUpdate(task: processingTask)
            } else {
                task.setTaskCompleted(success: false)
            }
        }

        print("✅ 后台任务已注册: \(taskIdentifier)")
    }

    // MARK: - 调度更新

    func scheduleModelUpdate() {
        // 检查是否满足更新条件
        guard shouldTriggerUpdate() else {
            print("⏭️ 不满足更新条件，跳过调度")
            return
        }

        let request = BGProcessingTaskRequest(identifier: taskIdentifier)
        request.requiresNetworkConnectivity = false
        request.requiresExternalPower = true // 需要充电
        request.earliestBeginDate = Date(timeIntervalSinceNow: 3600) // 1小时后执行

        do {
            try BGTaskScheduler.shared.submit(request)
            print("✅ 已调度模型更新任务")
        } catch {
            print("❌ 调度失败: \(error.localizedDescription)")
        }
    }

    // MARK: - 执行更新

    private func handleModelUpdate(task: BGProcessingTask) {
        print("🔄 开始执行模型更新任务")

        // 设置过期处理
        task.expirationHandler = {
            print("⚠️ 模型更新任务超时")
            task.setTaskCompleted(success: false)
        }

        // 异步执行更新
        Task {
            do {
                // 获取待训练样本
                let samples = try await fetchTrainingSamples()

                guard !samples.isEmpty else {
                    print("⏭️ 没有待训练样本")
                    task.setTaskCompleted(success: true)
                    return
                }

                print("📝 找到 \(samples.count) 个训练样本")

                // 更新模型
                try await analysisService.updateModel(with: samples)

                // 标记样本已处理
                await markSamplesAsProcessed(samples)

                // 记录更新时间
                UserDefaults.standard.set(Date(), forKey: "LastModelUpdateDate")

                print("✅ 模型更新成功")
                task.setTaskCompleted(success: true)
            } catch {
                print("❌ 模型更新失败: \(error.localizedDescription)")
                task.setTaskCompleted(success: false)
            }
        }
    }

    // MARK: - 辅助方法

    private func shouldTriggerUpdate() -> Bool {
        // 条件1：有足够的纠正样本
        let correctionCount = getCorrectionCount()
        guard correctionCount >= minCorrectionsForUpdate else {
            print("⏭️ 纠正样本不足: \(correctionCount)/\(minCorrectionsForUpdate)")
            return false
        }

        // 条件2：距离上次更新超过24小时
        if let lastUpdate = UserDefaults.standard.object(forKey: "LastModelUpdateDate") as? Date {
            let hoursSinceUpdate = Date().timeIntervalSince(lastUpdate) / 3600
            guard hoursSinceUpdate >= minHoursBetweenUpdates else {
                print("⏭️ 距离上次更新时间太短: \(Int(hoursSinceUpdate))小时")
                return false
            }
        }

        // 条件3：设备正在充电
        UIDevice.current.isBatteryMonitoringEnabled = true
        guard UIDevice.current.batteryState == .charging ||
            UIDevice.current.batteryState == .full
        else {
            print("⏭️ 设备未充电")
            return false
        }

        // 条件4：电量充足
        guard UIDevice.current.batteryLevel > 0.5 || UIDevice.current.batteryLevel == -1 else {
            print("⏭️ 电量不足: \(Int(UIDevice.current.batteryLevel * 100))%")
            return false
        }

        // 条件5：非低功耗模式
        guard !ProcessInfo.processInfo.isLowPowerModeEnabled else {
            print("⏭️ 低功耗模式已启用")
            return false
        }

        print("✅ 满足所有更新条件")
        return true
    }

    private func getCorrectionCount() -> Int {
        let context = DataController.shared.container.viewContext
        let request: NSFetchRequest<GoalEntry> = GoalEntry.fetchRequest()
        request.predicate = NSPredicate(format: "isTrainingSample == YES")

        do {
            let count = try context.count(for: request)
            return count
        } catch {
            print("❌ 查询纠正样本数失败: \(error)")
            return 0
        }
    }

    private func fetchTrainingSamples() async throws -> [TrainingSample] {
        let context = DataController.shared.container.viewContext

        return try await context.perform {
            let request: NSFetchRequest<GoalEntry> = GoalEntry.fetchRequest()
            request.predicate = NSPredicate(format: "isTrainingSample == YES")
            request.sortDescriptors = [NSSortDescriptor(key: "correctedAt", ascending: true)]

            let entries = try context.fetch(request)

            return entries.compactMap { entry in
                guard let correctedCategory = entry.categoryUserCorrected,
                      let correctedSentiment = entry.sentimentUserCorrected
                else {
                    return nil
                }

                return TrainingSample(
                    text: entry.goalText,
                    correctCategory: correctedCategory,
                    correctSentiment: correctedSentiment
                )
            }
        }
    }

    @MainActor
    private func markSamplesAsProcessed(_ samples: [TrainingSample]) async {
        let context = DataController.shared.container.viewContext

        // 将已训练的样本标记为已处理
        let request: NSFetchRequest<GoalEntry> = GoalEntry.fetchRequest()
        request.predicate = NSPredicate(format: "isTrainingSample == YES")

        do {
            let entries = try context.fetch(request)
            for entry in entries {
                entry.isTrainingSample = false
            }
            try context.save()
            print("✅ 已标记 \(entries.count) 个样本为已处理")
        } catch {
            print("❌ 标记样本失败: \(error)")
        }
    }

    // MARK: - 手动触发（用于测试）

    func triggerUpdateImmediately() {
        Task {
            do {
                let samples = try await fetchTrainingSamples()
                guard !samples.isEmpty else {
                    print("⏭️ 没有待训练样本")
                    return
                }

                print("🧪 手动触发更新: \(samples.count) 个样本")
                try await analysisService.updateModel(with: samples)
                await markSamplesAsProcessed(samples)
                UserDefaults.standard.set(Date(), forKey: "LastModelUpdateDate")

                print("✅ 手动更新完成")
            } catch {
                print("❌ 手动更新失败: \(error)")
            }
        }
    }
}
