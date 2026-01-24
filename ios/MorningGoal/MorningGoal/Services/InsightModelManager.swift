//
//  InsightModelManager.swift
//  MorningGoal
//
//  管理洞察分类器的加载、推理和更新
//  使用 Fan-out 架构: 1 个静态特征提取器 + 7 个可更新分类器
//

import Combine
import CoreML
import Foundation
import OSLog
import Tokenizers

// MARK: - Classification Result

/// 单个维度的分析结果
struct DimensionResult {
    let dimension: String
    let label: String
    let confidence: Double
    let allProbabilities: [String: Double]
}

/// 完整的洞察分析结果
struct InsightAnalysisResult {
    // 主题和情感
    let topic: DimensionResult?
    let sentiment: DimensionResult?

    // 5个洞察维度
    let urgency: DimensionResult?
    let timeFrame: DimensionResult?
    let actionType: DimensionResult?
    let difficulty: DimensionResult?
    let specificity: DimensionResult?

    /// 推理耗时 (毫秒)
    let inferenceTimeMs: Double

    /// 特征提取耗时 (毫秒)
    let featureExtractionTimeMs: Double

    /// 获取所有有效的维度结果
    var allDimensions: [DimensionResult] {
        [topic, sentiment, urgency, timeFrame, actionType, difficulty, specificity].compactMap { $0 }
    }
}

// MARK: - Classifier Config

/// 分类器配置
struct ClassifierConfig {
    let name: String
    let numClasses: Int
    let labels: [String]
    let bundleName: String
    let outputName: String
}

// MARK: - InsightModelManager

/// 洞察模型管理器 - 管理 BertFeatureExtractor + 7 个分类器
/// 使用 swift-transformers 的 Tokenizers 框架
@MainActor
final class InsightModelManager: ObservableObject {
    // MARK: - Properties

    private let logger = Logger(subsystem: "com.morninggoal.app", category: "insight-model")

    /// 共享特征提取器 (静态)
    private var featureExtractor: MLModel?

    /// 分类器字典
    private var classifiers: [String: MLModel] = [:]

    /// Tokenizer (使用 swift-transformers)
    private var tokenizer: Tokenizer?
    private let maxLength: Int = 128

    // Special token IDs (standard BERT)
    private let padTokenId: Int = 0
    private let clsTokenId: Int = 101
    private let sepTokenId: Int = 102

    /// 是否已初始化
    @Published var isInitialized: Bool = false

    /// 加载状态消息
    @Published var statusMessage: String = "正在初始化..."

    /// 已加载的分类器名称
    @Published var loadedClassifiers: [String] = []

    // MARK: - Label Mappings (与训练数据索引一致)

    // Topic: 16 categories
    let topicLabels = [
        "工作", "健康", "家庭", "个人发展", "理财", "社交", "家务", "学习",
        "睡眠", "饮食", "心态", "娱乐", "出行", "职业发展", "沟通", "育儿"
    ]

    // Sentiment: 3 categories
    let sentimentLabels = ["消极", "中性", "积极"]

    // Urgency: 3 categories
    let urgencyLabels = ["低", "中", "高"]

    // Timeframe: 4 categories
    let timeframeLabels = ["今天", "本周", "本月", "长期"]

    // Actiontype: 5 categories
    let actiontypeLabels = ["学习", "运动", "工作", "生活", "社交"]

    // Difficulty: 3 categories
    let difficultyLabels = ["简单", "中等", "困难"]

    // Specificity: 3 categories
    let specificityLabels = ["模糊", "一般", "具体"]

    /// 分类器配置 (7个分类器)
    static let classifierConfigs: [ClassifierConfig] = [
        // 主题分类
        ClassifierConfig(
            name: "topic",
            numClasses: 16,
            labels: [], // 使用 topicLabels
            bundleName: "TopicClassifier_Updatable",
            outputName: "topic_probs"
        ),
        // 情感分类
        ClassifierConfig(
            name: "sentiment",
            numClasses: 3,
            labels: [], // 使用 sentimentLabels
            bundleName: "SentimentClassifier_Updatable",
            outputName: "sentiment_probs"
        ),
        // 紧急度
        ClassifierConfig(
            name: "urgency",
            numClasses: 3,
            labels: [],
            bundleName: "UrgencyClassifier_Updatable",
            outputName: "urgency_probs"
        ),
        // 时间范围
        ClassifierConfig(
            name: "timeFrame",
            numClasses: 4,
            labels: [],
            bundleName: "TimeframeClassifier_Updatable",
            outputName: "timeFrame_probs"
        ),
        // 行动类型
        ClassifierConfig(
            name: "actionType",
            numClasses: 5,
            labels: [],
            bundleName: "ActiontypeClassifier_Updatable",
            outputName: "actionType_probs"
        ),
        // 难度
        ClassifierConfig(
            name: "difficulty",
            numClasses: 3,
            labels: [],
            bundleName: "DifficultyClassifier_Updatable",
            outputName: "difficulty_probs"
        ),
        // 具体程度
        ClassifierConfig(
            name: "specificity",
            numClasses: 3,
            labels: [],
            bundleName: "SpecificityClassifier_Updatable",
            outputName: "specificity_probs"
        )
    ]

    /// Models 目录 (App Support)
    private var modelsDirectory: URL {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        return appSupport.appendingPathComponent("Models", isDirectory: true)
    }

    /// Documents 目录 (用于更新的模型)
    private var documentsDirectory: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    // MARK: - Initialization

    init() {
        // 无参数初始化，tokenizer 在 loadModels 中异步加载
    }

    /// 异步加载所有模型
    func loadModels() async {
        statusMessage = "正在加载 Tokenizer..."

        do {
            // 1. 加载 Tokenizer (从 HuggingFace)
            tokenizer = try await AutoTokenizer.from(pretrained: "bert-base-chinese")
            logger.info("Tokenizer loaded from bert-base-chinese")

            // 2. 加载特征提取器
            statusMessage = "正在加载特征提取器..."
            try loadFeatureExtractor()

            // 3. 加载所有分类器
            statusMessage = "正在加载分类器..."
            for config in Self.classifierConfigs {
                do {
                    try loadClassifier(config: config)
                    loadedClassifiers.append(config.name)
                } catch {
                    logger.warning("Failed to load \(config.name): \(error.localizedDescription)")
                }
            }

            isInitialized = true
            statusMessage = "已加载 \(loadedClassifiers.count)/7 个分类器"
            logger.info("InsightModelManager initialized with \(self.classifiers.count) classifiers")

        } catch {
            statusMessage = "加载失败: \(error.localizedDescription)"
            logger.error("Failed to initialize: \(error.localizedDescription)")
        }
    }

    // MARK: - Model Loading

    private func loadFeatureExtractor() throws {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine

        // 尝试多个路径加载
        let possiblePaths = [
            Bundle.main.url(forResource: "BertFeatureExtractor", withExtension: "mlmodelc"),
            Bundle.main.url(forResource: "BertFeatureExtractor", withExtension: "mlpackage")
        ]

        for url in possiblePaths {
            if let url = url {
                self.featureExtractor = try MLModel(contentsOf: url, configuration: config)
                logger.info("Loaded BertFeatureExtractor from: \(url.lastPathComponent)")
                return
            }
        }

        throw AnalysisError.modelNotLoaded
    }

    private func loadClassifier(config: ClassifierConfig) throws {
        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = .cpuAndNeuralEngine

        // 检查 App Support 中是否有已更新的模型
        let updatedPath = modelsDirectory.appendingPathComponent("\(config.bundleName)_Updated.mlmodelc")

        if FileManager.default.fileExists(atPath: updatedPath.path) {
            classifiers[config.name] = try MLModel(contentsOf: updatedPath, configuration: mlConfig)
            logger.info("Loaded updated \(config.name) classifier from App Support")
            return
        }

        // 从 Bundle 加载
        let possiblePaths = [
            Bundle.main.url(forResource: config.bundleName, withExtension: "mlmodelc"),
            Bundle.main.url(forResource: config.bundleName, withExtension: "mlpackage")
        ]

        for url in possiblePaths {
            if let url = url {
                classifiers[config.name] = try MLModel(contentsOf: url, configuration: mlConfig)
                logger.info("Loaded \(config.name) classifier from: \(url.lastPathComponent)")

                // 复制到 App Support 以便后续更新
                try copyModelToAppSupport(from: url, config: config)
                return
            }
        }

        throw AnalysisError.modelNotLoaded
    }

    private func copyModelToAppSupport(from sourceURL: URL, config: ClassifierConfig) throws {
        let fileManager = FileManager.default

        if !fileManager.fileExists(atPath: modelsDirectory.path) {
            try fileManager.createDirectory(at: modelsDirectory, withIntermediateDirectories: true)
        }

        let destURL = modelsDirectory.appendingPathComponent("\(config.bundleName)_Updated.mlmodelc")
        if !fileManager.fileExists(atPath: destURL.path) {
            try fileManager.copyItem(at: sourceURL, to: destURL)
            logger.info("Copied model to App Support: \(config.name)")
        }
    }

    // MARK: - Tokenization

    /// 使用 swift-transformers 的 Tokenizer 对文本进行编码
    private func encode(_ text: String) -> (inputIds: [Int], attentionMask: [Int]) {
        guard let tokenizer = tokenizer else {
            logger.warning("Tokenizer not loaded, using fallback")
            return fallbackEncode(text)
        }

        // 使用 swift-transformers 编码
        let encoded = tokenizer.encode(text: text)
        var inputIds = encoded.map { Int($0) }

        // 截断
        if inputIds.count > maxLength {
            inputIds = Array(inputIds.prefix(maxLength - 1)) + [sepTokenId]
        }

        // 创建 attention mask
        var attentionMask = Array(repeating: 1, count: inputIds.count)

        // 填充到 maxLength
        let paddingCount = maxLength - inputIds.count
        if paddingCount > 0 {
            inputIds.append(contentsOf: Array(repeating: padTokenId, count: paddingCount))
            attentionMask.append(contentsOf: Array(repeating: 0, count: paddingCount))
        }

        return (inputIds, attentionMask)
    }

    /// 后备编码方法 (当 tokenizer 未加载时)
    private func fallbackEncode(_ text: String) -> (inputIds: [Int], attentionMask: [Int]) {
        let words = text.components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }

        var inputIds: [Int] = [clsTokenId]
        for _ in words.prefix(maxLength - 2) {
            inputIds.append(100) // UNK token
        }
        inputIds.append(sepTokenId)

        var attentionMask = Array(repeating: 1, count: inputIds.count)

        let paddingCount = maxLength - inputIds.count
        if paddingCount > 0 {
            inputIds.append(contentsOf: Array(repeating: padTokenId, count: paddingCount))
            attentionMask.append(contentsOf: Array(repeating: 0, count: paddingCount))
        }

        return (inputIds, attentionMask)
    }

    // MARK: - Inference

    /// 提取文本嵌入向量
    func extractEmbedding(from text: String) throws -> MLMultiArray {
        guard let model = featureExtractor else {
            throw AnalysisError.modelNotLoaded
        }

        let (inputIds, attentionMask) = encode(text)

        // 创建 MLMultiArray 输入
        let inputIdsArray = try createMultiArray(from: inputIds, shape: [1, maxLength])
        let attentionMaskArray = try createMultiArray(from: attentionMask, shape: [1, maxLength])

        // 创建输入特征
        let inputFeatures = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: inputIdsArray),
            "attention_mask": MLFeatureValue(multiArray: attentionMaskArray)
        ])

        // 运行推理
        let output = try model.prediction(from: inputFeatures)

        // 尝试常见的嵌入输出名称
        let possibleKeys = ["embedding", "pooler_output", "last_hidden_state", "embeddings", "output"]

        for key in possibleKeys {
            if let embedding = output.featureValue(for: key)?.multiArrayValue {
                return embedding
            }
        }

        // 回退：使用第一个可用的 MLMultiArray 输出
        for name in output.featureNames {
            if let embedding = output.featureValue(for: name)?.multiArrayValue {
                return embedding
            }
        }

        throw AnalysisError.modelNotLoaded
    }

    /// 分析目标文本，返回所有维度的分类结果
    func analyze(text: String) async throws -> InsightAnalysisResult {
        guard featureExtractor != nil else {
            throw AnalysisError.modelNotLoaded
        }

        let t0 = CFAbsoluteTimeGetCurrent()

        // 1. 提取嵌入 (共享)
        let embedding = try extractEmbedding(from: text)

        let t1 = CFAbsoluteTimeGetCurrent()
        let featureExtractionTime = (t1 - t0) * 1000

        // 2. 并行运行所有分类器
        var results: [String: DimensionResult] = [:]

        let embeddingInput = try MLDictionaryFeatureProvider(dictionary: [
            "embedding": MLFeatureValue(multiArray: embedding)
        ])

        await withTaskGroup(of: (String, DimensionResult)?.self) { group in
            for config in Self.classifierConfigs {
                guard let classifier = classifiers[config.name] else { continue }

                group.addTask {
                    do {
                        let result = try await self.runClassifier(
                            classifier: classifier,
                            config: config,
                            embeddingInput: embeddingInput
                        )
                        return (config.name, result)
                    } catch {
                        self.logger.error("\(config.name) inference failed: \(error.localizedDescription)")
                        return nil
                    }
                }
            }

            for await result in group {
                if let (name, dimensionResult) = result {
                    results[name] = dimensionResult
                }
            }
        }

        let t2 = CFAbsoluteTimeGetCurrent()
        let totalInferenceTime = (t2 - t0) * 1000

        logger.info(
            "inference_ms=\(String(format: "%.2f", totalInferenceTime)), fe_ms=\(String(format: "%.2f", featureExtractionTime)), classifiers=\(results.count)"
        )

        return InsightAnalysisResult(
            topic: results["topic"],
            sentiment: results["sentiment"],
            urgency: results["urgency"],
            timeFrame: results["timeFrame"],
            actionType: results["actionType"],
            difficulty: results["difficulty"],
            specificity: results["specificity"],
            inferenceTimeMs: totalInferenceTime,
            featureExtractionTimeMs: featureExtractionTime
        )
    }

    private func runClassifier(
        classifier: MLModel,
        config: ClassifierConfig,
        embeddingInput: MLDictionaryFeatureProvider
    ) async throws -> DimensionResult {
        let output = try await classifier.prediction(from: embeddingInput)

        // 获取概率输出
        guard let probs = output.featureValue(for: config.outputName)?.multiArrayValue else {
            throw AnalysisError.modelNotLoaded
        }

        // 获取标签
        let labels = getLabels(for: config.name)

        // 找到最大概率的索引
        let (maxIndex, maxProb) = argmax(probs)

        // 构建所有概率
        var allProbs: [String: Double] = [:]
        for i in 0 ..< probs.count {
            let label = labels.indices.contains(i) ? labels[i] : "unknown_\(i)"
            allProbs[label] = probs[i].doubleValue
        }

        let label = labels.indices.contains(maxIndex) ? labels[maxIndex] : "unknown"

        return DimensionResult(
            dimension: config.name,
            label: label,
            confidence: maxProb,
            allProbabilities: allProbs
        )
    }

    private func getLabels(for dimension: String) -> [String] {
        switch dimension {
        case "topic": return topicLabels
        case "sentiment": return sentimentLabels
        case "urgency": return urgencyLabels
        case "timeFrame": return timeframeLabels
        case "actionType": return actiontypeLabels
        case "difficulty": return difficultyLabels
        case "specificity": return specificityLabels
        default: return []
        }
    }

    // MARK: - Helper Methods

    private func createMultiArray(from array: [Int], shape: [Int]) throws -> MLMultiArray {
        let multiArray = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .int32)
        for (index, value) in array.enumerated() {
            multiArray[index] = NSNumber(value: value)
        }
        return multiArray
    }

    private func argmax(_ array: MLMultiArray) -> (index: Int, confidence: Double) {
        var maxIndex = 0
        var maxValue = -Double.infinity

        for i in 0 ..< array.count {
            let value = array[i].doubleValue
            if value > maxValue {
                maxValue = value
                maxIndex = i
            }
        }

        return (maxIndex, maxValue)
    }

    // MARK: - Model Info

    /// 获取模型版本信息
    func getModelVersions() -> [String: String] {
        var versions: [String: String] = [:]

        versions["featureExtractor"] = featureExtractor != nil ? "loaded" : "not_loaded"

        for config in Self.classifierConfigs {
            let updatedPath = modelsDirectory.appendingPathComponent("\(config.bundleName)_Updated.mlmodelc")
            let isUpdated = FileManager.default.fileExists(atPath: updatedPath.path)
            let isLoaded = classifiers[config.name] != nil

            if !isLoaded {
                versions[config.name] = "not_loaded"
            } else if isUpdated {
                versions[config.name] = "updated"
            } else {
                versions[config.name] = "bundled"
            }
        }

        return versions
    }

    /// 重置所有分类器到初始状态
    func resetAllClassifiers() throws {
        for config in Self.classifierConfigs {
            let updatedPath = modelsDirectory.appendingPathComponent("\(config.bundleName)_Updated.mlmodelc")
            if FileManager.default.fileExists(atPath: updatedPath.path) {
                try FileManager.default.removeItem(at: updatedPath)
                logger.info("Deleted updated model: \(config.name)")
            }
        }

        // 清空分类器并重新加载
        classifiers.removeAll()
        loadedClassifiers.removeAll()

        Task {
            await loadModels()
        }
    }

    /// 获取用于训练的嵌入向量
    func getEmbeddingForTraining(text: String) throws -> MLMultiArray {
        return try extractEmbedding(from: text)
    }

    /// 重新加载指定的分类器
    func reloadClassifier(name: String) throws {
        guard let config = Self.classifierConfigs.first(where: { $0.name == name }) else {
            throw AnalysisError.modelNotLoaded
        }

        let updatedPath = modelsDirectory.appendingPathComponent("\(config.bundleName)_Updated.mlmodelc")
        guard FileManager.default.fileExists(atPath: updatedPath.path) else {
            throw AnalysisError.modelNotLoaded
        }

        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = .cpuAndNeuralEngine
        classifiers[name] = try MLModel(contentsOf: updatedPath, configuration: mlConfig)
        logger.info("Reloaded \(name) classifier")
    }

    /// 重新加载所有分类器
    func reloadAllClassifiers() throws {
        for config in Self.classifierConfigs {
            try? reloadClassifier(name: config.name)
        }
        logger.info("All classifiers reloaded")
    }

    /// 训练完成后重新加载模型（异步版本，更新状态）
    func reloadModelsAfterTraining() async {
        statusMessage = "正在重新加载分类器..."
        loadedClassifiers.removeAll()

        for config in Self.classifierConfigs {
            do {
                try loadClassifier(config: config)
                if !loadedClassifiers.contains(config.name) {
                    loadedClassifiers.append(config.name)
                }
            } catch {
                logger.warning("Failed to reload \(config.name): \(error.localizedDescription)")
            }
        }

        statusMessage = "已加载 \(loadedClassifiers.count)/7 个分类器 (已更新)"
        let count = loadedClassifiers.count
        logger.info("Models reloaded after training: \(count) classifiers")
    }
}

// MARK: - InsightModelManager Wrapper

/// 安全的 InsightModelManager 包装器
@MainActor
final class InsightModelManagerWrapper: ObservableObject {
    private var manager: InsightModelManager?

    @Published var isInitialized: Bool = false
    @Published var statusMessage: String = "正在初始化..."
    @Published var loadedClassifiers: [String] = []
    @Published var initError: String?

    init() {
        manager = InsightModelManager()
    }

    func loadModels() async {
        guard let manager = manager else {
            statusMessage = initError ?? "模型管理器未初始化"
            return
        }

        await manager.loadModels()

        // 同步状态
        isInitialized = manager.isInitialized
        statusMessage = manager.statusMessage
        loadedClassifiers = manager.loadedClassifiers
    }

    func analyze(text: String) async throws -> InsightAnalysisResult {
        guard let manager = manager else {
            throw AnalysisError.modelNotLoaded
        }
        return try await manager.analyze(text: text)
    }

    func getModelVersions() -> [String: String] {
        return manager?.getModelVersions() ?? [:]
    }

    func resetAllClassifiers() throws {
        try manager?.resetAllClassifiers()
    }

    func getEmbeddingForTraining(text: String) throws -> MLMultiArray {
        guard let manager = manager else {
            throw AnalysisError.modelNotLoaded
        }
        return try manager.getEmbeddingForTraining(text: text)
    }

    /// 训练完成后重新加载模型
    func reloadModelsAfterTraining() async {
        guard let manager = manager else { return }

        await manager.reloadModelsAfterTraining()

        // 同步状态
        isInitialized = manager.isInitialized
        statusMessage = manager.statusMessage
        loadedClassifiers = manager.loadedClassifiers
    }
}
