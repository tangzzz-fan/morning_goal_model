import CoreData
import SwiftUI

/// 模型调试界面
/// 用于实时测试模型预测、查看结果、调整参数和触发训练
@MainActor
struct ModelDebugView: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(\.managedObjectContext) private var context

    // 获取用户已有的目标数据
    @FetchRequest(
        entity: GoalEntry.entity(),
        sortDescriptors: [NSSortDescriptor(keyPath: \GoalEntry.lastUpdated, ascending: false)],
        predicate: NSPredicate(format: "isTrainingSample == NO"), // 只显示真实目标,不包括训练样本
        animation: .default
    )
    private var existingGoals: FetchedResults<GoalEntry>

    @State private var inputText = ""
    @State private var predictionResult: AnalysisResult?
    @State private var isAnalyzing = false
    @State private var errorMessage: String?

    // 训练相关
    @State private var showTrainingSection = false
    @State private var trainingCount = 0
    @State private var isTraining = false
    @State private var trainingLog: [String] = []

    // 纠正数据 - 支持多标签
    @State private var selectedCategories: Set<String> = []
    @State private var correctedSentiment = "积极"

    // 统计信息
    @State private var modelInfo: String = ""
    @State private var correctionCount = 0

    // 已有目标分析
    @State private var showExistingGoals = false
    @State private var isBatchAnalyzing = false
    @State private var batchProgress: Double = 0.0
    @State private var analyzedCount = 0

    private let categories = [
        "工作",
        "健康",
        "家庭",
        "个人发展",
        "财务",
        "学习",
        "社交",
        "休闲",
        "创作",
        "运动",
        "旅行",
        "爱好",
        "志愿",
        "精神",
        "家务",
        "其他"
    ]
    private let sentiments = ["积极", "中性", "消极"]

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 24) {
                    ModelInfoSectionView(
                        modelInfo: modelInfo,
                        correctionCount: correctionCount,
                        refresh: loadModelInfo
                    )

                    PredictionTestSectionView(
                        inputText: $inputText,
                        isAnalyzing: isAnalyzing,
                        errorMessage: errorMessage,
                        runPrediction: runPrediction
                    )

                    if let result = predictionResult {
                        PredictionResultSectionView(
                            result: result,
                            categories: categories,
                            sentiments: sentiments,
                            selectedCategories: $selectedCategories,
                            correctedSentiment: $correctedSentiment,
                            saveCorrection: saveCorrection
                        )
                    }

                    ExistingGoalsSectionView(
                        showExistingGoals: $showExistingGoals,
                        existingGoals: existingGoals,
                        inputText: $inputText,
                        isBatchAnalyzing: isBatchAnalyzing,
                        batchProgress: batchProgress,
                        analyzedCount: analyzedCount,
                        batchAnalyzeGoals: batchAnalyzeGoals
                    )

                    TrainingControlSectionView(
                        showTrainingSection: $showTrainingSection,
                        correctionCount: correctionCount,
                        isTraining: isTraining,
                        trainingCount: trainingCount,
                        trainingLog: trainingLog,
                        triggerTraining: triggerTraining,
                        clearTrainingLog: { trainingLog.removeAll() }
                    )
                }
                .padding()
            }
            .navigationTitle("模型调试")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("关闭") { dismiss() }
                }
            }
        }
        .onAppear {
            loadModelInfo()
            loadCorrectionCount()
        }
    }

    // MARK: - Actions

    private func loadModelInfo() {
        do {
            let service = try AdaptiveModelService()
            modelInfo = service.getModelVersion()
        } catch {
            modelInfo = "加载失败"
        }
    }

    private func loadCorrectionCount() {
        let context = DataController.shared.container.viewContext
        let request: NSFetchRequest<GoalEntry> = GoalEntry.fetchRequest()
        request.predicate = NSPredicate(format: "isTrainingSample == YES")

        do {
            correctionCount = try context.count(for: request)
        } catch {
            correctionCount = 0
        }

        // 计算已分析的目标数量
        let analyzedRequest: NSFetchRequest<GoalEntry> = GoalEntry.fetchRequest()
        analyzedRequest.predicate = NSPredicate(format: "isTrainingSample == NO AND category != nil")
        do {
            analyzedCount = try context.count(for: analyzedRequest)
        } catch {
            analyzedCount = 0
        }
    }

    private func runPrediction() {
        isAnalyzing = true
        errorMessage = nil
        predictionResult = nil

        Task {
            do {
                let service = try AdaptiveModelService()
                let result = try await service.analyzeGoal(inputText)

                await MainActor.run {
                    predictionResult = result
                    // 初始化选中的分类 - 默认选中 Top-1
                    selectedCategories = [result.category]
                    // 如果有 Top-K 结果,也可以预选 Top-2
                    if let topCategories = result.topCategories, topCategories.count > 1 {
                        // 可选:自动选中概率 > 20% 的分类
                        let significantCategories = topCategories.filter { $0.confidence > 0.2 }.map { $0.name }
                        selectedCategories = Set(significantCategories)
                    }
                    correctedSentiment = result.sentiment
                    isAnalyzing = false
                }
            } catch {
                await MainActor.run {
                    errorMessage = "预测失败: \(error.localizedDescription)"
                    isAnalyzing = false
                }
            }
        }
    }

    private func saveCorrection() {
        guard let result = predictionResult else { return }
        guard !selectedCategories.isEmpty else { return }

        let context = DataController.shared.container.viewContext

        // 为每个选中的分类创建一个训练样本
        for category in selectedCategories {
            let entry = GoalEntry(context: context)
            entry.goalText = inputText
            entry.dateString = GoalEntry.todayString() + "_" + UUID().uuidString.prefix(8) // 避免重复
            entry.lastUpdated = Date()
            entry.category = result.category // 原始预测
            entry.sentiment = result.sentiment
            entry.categoryUserCorrected = category // 纠正后的分类
            entry.sentimentUserCorrected = correctedSentiment
            entry.isTrainingSample = true
            entry.correctedAt = Date()
        }

        do {
            try context.save()
            loadCorrectionCount()
            let categoriesStr = selectedCategories.sorted().joined(separator: ", ")
            addLog("✅ 已保存 \(selectedCategories.count) 个纠正: \(categoriesStr) / \(correctedSentiment)")
        } catch {
            addLog("❌ 保存失败: \(error.localizedDescription)")
        }
    }

    private func batchAnalyzeGoals() {
        isBatchAnalyzing = true
        batchProgress = 0.0
        addLog("📊 开始批量分析 \(existingGoals.count) 个目标...")

        Task {
            do {
                let service = try AdaptiveModelService()
                let total = existingGoals.count

                for (index, goal) in existingGoals.enumerated() {
                    // 跳过已经分析过的
                    if goal.category != nil {
                        await MainActor.run {
                            batchProgress = Double(index + 1) / Double(total)
                        }
                        continue
                    }

                    // 分析目标
                    let result = try await service.analyzeGoal(goal.goalText)

                    // 保存结果
                    await MainActor.run {
                        goal.category = result.category
                        goal.categoryConfidence = result.categoryConfidence
                        goal.sentiment = result.sentiment
                        goal.sentimentScore = result.sentimentScore
                        goal.analyzedAt = Date()

                        batchProgress = Double(index + 1) / Double(total)
                    }
                }

                // 保存所有更改
                await MainActor.run {
                    do {
                        try context.save()
                        loadCorrectionCount() // 更新统计
                        addLog("✅ 批量分析完成! 共分析 \(total) 个目标")
                        isBatchAnalyzing = false
                    } catch {
                        addLog("❌ 保存失败: \(error.localizedDescription)")
                        isBatchAnalyzing = false
                    }
                }
            } catch {
                await MainActor.run {
                    addLog("❌ 批量分析失败: \(error.localizedDescription)")
                    isBatchAnalyzing = false
                }
            }
        }
    }

    private func triggerTraining() {
        isTraining = true
        addLog("🎓 开始训练...")

        Task {
            do {
                let samples = try await fetchTrainingSamples()
                addLog("📊 获取到 \(samples.count) 个训练样本")

                let service = try AdaptiveModelService()
                try await service.updateModel(with: samples)

                await MainActor.run {
                    trainingCount += 1
                    isTraining = false
                    addLog("✅ 训练完成!")
                    addLog("🔄 模型已更新,下次预测将使用新模型")

                    // 清除已训练的样本
                    markSamplesAsProcessed(samples)
                    loadCorrectionCount()
                }
            } catch {
                await MainActor.run {
                    isTraining = false
                    addLog("❌ 训练失败: \(error.localizedDescription)")
                }
            }
        }
    }

    private func fetchTrainingSamples() async throws -> [TrainingSample] {
        let context = DataController.shared.container.viewContext
        let request: NSFetchRequest<GoalEntry> = GoalEntry.fetchRequest()
        request.predicate = NSPredicate(format: "isTrainingSample == YES")

        return try await context.perform {
            let entries = try context.fetch(request)
            return entries.compactMap { entry -> TrainingSample? in
                guard let correctedCat = entry.categoryUserCorrected,
                      let correctedSent = entry.sentimentUserCorrected
                else {
                    return nil
                }
                return TrainingSample(
                    text: entry.goalText,
                    correctCategory: correctedCat,
                    correctSentiment: correctedSent
                )
            }
        }
    }

    private func markSamplesAsProcessed(_ samples: [TrainingSample]) {
        let context = DataController.shared.container.viewContext
        let request: NSFetchRequest<GoalEntry> = GoalEntry.fetchRequest()
        request.predicate = NSPredicate(format: "isTrainingSample == YES")

        context.perform {
            if let entries = try? context.fetch(request) {
                for entry in entries {
                    entry.isTrainingSample = false
                }
                try? context.save()
            }
        }
    }

    private func addLog(_ message: String) {
        let timestamp = DateFormatter.localizedString(from: Date(), dateStyle: .none, timeStyle: .medium)
        trainingLog.insert("[\(timestamp)] \(message)", at: 0)
    }
}

// MARK: - Preview

#Preview {
    ModelDebugView()
}
