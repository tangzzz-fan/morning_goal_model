//
//  InsightUpdateManager.swift
//  MorningGoal
//
//  设备端训练管理器 - 使用 MLUpdateTask 实现 Fan-out 架构下的模型更新
//  支持 7 个分类器的独立更新
//

import Combine
import CoreML
import Foundation
import OSLog

// MARK: - Training Sample

/// 训练样本 - 包含嵌入向量和所有 7 个维度的正确标签
struct InsightTrainingSample {
    let embedding: MLMultiArray
    let text: String
    let topicLabel: Int
    let sentimentLabel: Int
    let urgencyLabel: Int
    let timeframeLabel: Int
    let actiontypeLabel: Int
    let difficultyLabel: Int
    let specificityLabel: Int
    let timestamp: Date

    init(
        embedding: MLMultiArray,
        text: String,
        topicLabel: Int,
        sentimentLabel: Int,
        urgencyLabel: Int,
        timeframeLabel: Int,
        actiontypeLabel: Int,
        difficultyLabel: Int,
        specificityLabel: Int
    ) {
        self.embedding = embedding
        self.text = text
        self.topicLabel = topicLabel
        self.sentimentLabel = sentimentLabel
        self.urgencyLabel = urgencyLabel
        self.timeframeLabel = timeframeLabel
        self.actiontypeLabel = actiontypeLabel
        self.difficultyLabel = difficultyLabel
        self.specificityLabel = specificityLabel
        self.timestamp = Date()
    }
}

// MARK: - InsightUpdateManager

/// 设备端模型更新管理器
/// 支持 7 个分类器的独立更新
@MainActor
final class InsightUpdateManager: ObservableObject {
    // MARK: - Properties

    private let logger = Logger(subsystem: "com.morninggoal.app", category: "insight-update")

    @Published var isTraining = false
    @Published var trainingProgress: Float = 0.0
    @Published var trainingMessage: String = ""
    @Published var lastTrainingDate: Date?

    private var trainingSamples: [InsightTrainingSample] = []

    // MARK: - Model URLs in App Support

    private var modelsDirectory: URL {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        return appSupport.appendingPathComponent("Models", isDirectory: true)
    }

    func modelURL(for name: String) -> URL {
        modelsDirectory.appendingPathComponent("\(name)Classifier_Updated.mlmodelc")
    }

    // MARK: - Public Methods

    /// 获取待训练样本数量
    var pendingSampleCount: Int {
        trainingSamples.count
    }

    /// 添加训练样本
    func addTrainingSample(
        embedding: MLMultiArray,
        text: String,
        topicLabel: Int,
        sentimentLabel: Int,
        urgencyLabel: Int,
        timeframeLabel: Int,
        actiontypeLabel: Int,
        difficultyLabel: Int,
        specificityLabel: Int
    ) {
        let sample = InsightTrainingSample(
            embedding: embedding,
            text: text,
            topicLabel: topicLabel,
            sentimentLabel: sentimentLabel,
            urgencyLabel: urgencyLabel,
            timeframeLabel: timeframeLabel,
            actiontypeLabel: actiontypeLabel,
            difficultyLabel: difficultyLabel,
            specificityLabel: specificityLabel
        )
        trainingSamples.append(sample)
        logger.info("Added training sample. Total: \(self.trainingSamples.count)")
    }

    /// 清空所有训练样本
    func clearSamples() {
        trainingSamples.removeAll()
        logger.info("Cleared all training samples")
    }

    /// 获取所有训练样本
    func getSamples() -> [InsightTrainingSample] {
        trainingSamples
    }

    // MARK: - Model Update

    /// 更新所有 7 个分类器
    func updateAllModels() async throws {
        guard !trainingSamples.isEmpty else {
            throw InsightUpdateError.noSamples
        }

        isTraining = true
        trainingProgress = 0.0
        trainingMessage = "准备训练数据..."

        defer {
            isTraining = false
        }

        do {
            let progressPerModel: Float = 1.0 / 7.0
            var currentProgress: Float = 0.0

            // 1. Topic
            trainingMessage = "训练主题分类器 (1/7)..."
            try await updateModel(
                name: "Topic",
                labelKeyPath: \.topicLabel,
                probsInputName: "topic_probs_true"
            )
            currentProgress += progressPerModel
            trainingProgress = currentProgress

            // 2. Sentiment
            trainingMessage = "训练情感分类器 (2/7)..."
            try await updateModel(
                name: "Sentiment",
                labelKeyPath: \.sentimentLabel,
                probsInputName: "sentiment_probs_true"
            )
            currentProgress += progressPerModel
            trainingProgress = currentProgress

            // 3. Urgency
            trainingMessage = "训练紧急度分类器 (3/7)..."
            try await updateModel(
                name: "Urgency",
                labelKeyPath: \.urgencyLabel,
                probsInputName: "urgency_probs_true"
            )
            currentProgress += progressPerModel
            trainingProgress = currentProgress

            // 4. Timeframe
            trainingMessage = "训练时间范围分类器 (4/7)..."
            try await updateModel(
                name: "Timeframe",
                labelKeyPath: \.timeframeLabel,
                probsInputName: "timeFrame_probs_true"
            )
            currentProgress += progressPerModel
            trainingProgress = currentProgress

            // 5. Actiontype
            trainingMessage = "训练行动类型分类器 (5/7)..."
            try await updateModel(
                name: "Actiontype",
                labelKeyPath: \.actiontypeLabel,
                probsInputName: "actionType_probs_true"
            )
            currentProgress += progressPerModel
            trainingProgress = currentProgress

            // 6. Difficulty
            trainingMessage = "训练难度分类器 (6/7)..."
            try await updateModel(
                name: "Difficulty",
                labelKeyPath: \.difficultyLabel,
                probsInputName: "difficulty_probs_true"
            )
            currentProgress += progressPerModel
            trainingProgress = currentProgress

            // 7. Specificity
            trainingMessage = "训练具体程度分类器 (7/7)..."
            try await updateModel(
                name: "Specificity",
                labelKeyPath: \.specificityLabel,
                probsInputName: "specificity_probs_true"
            )

            trainingProgress = 1.0
            trainingMessage = "所有模型训练完成! ✅"
            lastTrainingDate = Date()

            // 训练完成后清空样本
            trainingSamples.removeAll()

            logger.info("All 7 models updated successfully")

        } catch {
            trainingMessage = "训练失败: \(error.localizedDescription)"
            throw error
        }
    }

    // MARK: - Private Methods

    private func updateModel(
        name: String,
        labelKeyPath: KeyPath<InsightTrainingSample, Int>,
        probsInputName: String
    ) async throws {
        let bundleName = "\(name)Classifier_Updatable"
        // 使用与 InsightModelManager 一致的路径命名
        let modelURL = modelsDirectory.appendingPathComponent("\(bundleName)_Updated.mlmodelc")

        // 如果更新后的模型不存在，从 bundle 复制
        if !FileManager.default.fileExists(atPath: modelURL.path) {
            try copyModelFromBundle(name: name, destURL: modelURL)
        }

        let batchProvider = try createBatchProvider(
            labelKeyPath: labelKeyPath,
            probsInputName: probsInputName
        )

        try await performUpdate(
            modelURL: modelURL,
            trainingData: batchProvider,
            taskName: name
        )
    }

    private func copyModelFromBundle(name: String, destURL: URL) throws {
        let bundleName = "\(name)Classifier_Updatable"

        guard let sourceURL = Bundle.main.url(forResource: bundleName, withExtension: "mlmodelc")
            ?? Bundle.main.url(forResource: bundleName, withExtension: "mlpackage")
        else {
            throw InsightUpdateError.modelNotFound(name)
        }

        let fileManager = FileManager.default
        if !fileManager.fileExists(atPath: modelsDirectory.path) {
            try fileManager.createDirectory(at: modelsDirectory, withIntermediateDirectories: true)
        }

        if !fileManager.fileExists(atPath: destURL.path) {
            try fileManager.copyItem(at: sourceURL, to: destURL)
            logger.info("Copied \(name) model to App Support for training")
        }
    }

    private func createBatchProvider(
        labelKeyPath: KeyPath<InsightTrainingSample, Int>,
        probsInputName: String
    ) throws -> MLBatchProvider {
        var featureProviders: [MLFeatureProvider] = []

        for sample in trainingSamples {
            let labelArray = try MLMultiArray(shape: [1], dataType: .int32)
            labelArray[0] = NSNumber(value: sample[keyPath: labelKeyPath])

            let features = try MLDictionaryFeatureProvider(dictionary: [
                "embedding": MLFeatureValue(multiArray: sample.embedding),
                probsInputName: MLFeatureValue(multiArray: labelArray)
            ])

            featureProviders.append(features)
        }

        return MLArrayBatchProvider(array: featureProviders)
    }

    private func performUpdate(
        modelURL: URL,
        trainingData: MLBatchProvider,
        taskName: String
    ) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            let progressHandler: (MLUpdateContext) -> Void = { [weak self] context in
                Task { @MainActor in
                    let epoch = context.metrics[.epochIndex] as? Int ?? 0
                    self?.trainingMessage = "训练 \(taskName) - 第 \(epoch) 轮..."
                }
            }

            let completionHandler: (MLUpdateContext) -> Void = { [weak self] context in
                Task { @MainActor in
                    self?.handleUpdateCompletion(
                        context: context,
                        modelURL: modelURL,
                        taskName: taskName,
                        continuation: continuation
                    )
                }
            }

            do {
                let updateTask = try MLUpdateTask(
                    forModelAt: modelURL,
                    trainingData: trainingData,
                    configuration: nil,
                    progressHandlers: MLUpdateProgressHandlers(
                        forEvents: [.epochEnd],
                        progressHandler: progressHandler,
                        completionHandler: completionHandler
                    )
                )
                updateTask.resume()
            } catch {
                continuation.resume(
                    throwing: InsightUpdateError.trainingFailed("\(taskName) - 创建 MLUpdateTask 失败: \(error.localizedDescription)")
                )
            }
        }
    }

    private func handleUpdateCompletion(
        context: MLUpdateContext,
        modelURL: URL,
        taskName: String,
        continuation: CheckedContinuation<Void, Error>
    ) {
        switch context.task.state {
        case .completed:
            do {
                try context.model.write(to: modelURL)
                logger.info("\(taskName) model updated and saved")
                continuation.resume()
            } catch {
                continuation.resume(
                    throwing: InsightUpdateError.trainingFailed("\(taskName) - 保存失败: \(error.localizedDescription)")
                )
            }

        case .failed:
            let errorMessage = context.task.error?.localizedDescription ?? "未知错误"
            continuation.resume(throwing: InsightUpdateError.trainingFailed("\(taskName) - \(errorMessage)"))

        case .cancelling:
            continuation.resume(throwing: InsightUpdateError.trainingFailed("\(taskName) - 训练被取消"))

        default:
            continuation.resume(throwing: InsightUpdateError.trainingFailed("\(taskName) - 意外状态"))
        }
    }
}

// MARK: - Error Types

enum InsightUpdateError: LocalizedError {
    case noSamples
    case modelNotFound(String)
    case trainingFailed(String)

    var errorDescription: String? {
        switch self {
        case .noSamples: return "没有可用的训练样本"
        case let .modelNotFound(name): return "找不到模型: \(name)"
        case let .trainingFailed(msg): return "训练失败: \(msg)"
        }
    }
}
