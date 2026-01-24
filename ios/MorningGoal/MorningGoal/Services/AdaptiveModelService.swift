import CoreML
import Foundation
import OSLog

struct ModelMetadata {
    let inputShape: [Int]
    let maxSequenceLength: Int
    let supportedDataTypes: [MLMultiArrayDataType]

    static func detect(from model: MLModel) -> ModelMetadata? {
        guard let inputDescription = model.modelDescription.inputDescriptionsByName["input_ids"] else {
            return nil
        }

        guard let multiArrayConstraint = inputDescription.multiArrayConstraint else {
            return nil
        }

        let shape = multiArrayConstraint.shape.map { $0.intValue }
        let maxLength = shape.last ?? 128
        let dataTypes = multiArrayConstraint.dataType == .int32 ? [MLMultiArrayDataType.int32] : [MLMultiArrayDataType.int32, .double]

        return ModelMetadata(
            inputShape: shape,
            maxSequenceLength: maxLength,
            supportedDataTypes: dataTypes
        )
    }
}

@MainActor
final class AdaptiveModelService {
    private var model: MLModel
    private var currentModelURL: URL
    private let tokenizer: BertTokenizer
    private let metadata: ModelMetadata
    private let logger = Logger(subsystem: "com.morninggoal.app", category: "adaptive")

    init() throws {
        logger.log("🚀 [INIT] Starting AdaptiveModelService initialization...")

        let fileManager = FileManager.default
        guard let docURL = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first else {
            throw AnalysisError.modelNotLoaded
        }
        let updatedModelURL = docURL.appendingPathComponent("student_sequence_classification.mlmodelc")

        logger.log("📂 [INIT] Documents directory: \(docURL.path)")
        logger.log("🔍 [INIT] Checking for updated model at: \(updatedModelURL.path)")

        let bundle = Bundle.main
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine

        // 尝试加载已更新的模型
        if fileManager.fileExists(atPath: updatedModelURL.path) {
            logger.log("✅ [INIT] Found updated model, loading from Documents...")
            self.currentModelURL = updatedModelURL
            self.model = try MLModel(contentsOf: updatedModelURL, configuration: config)
            logger.log("✅ [INIT] Successfully loaded UPDATED model from: \(updatedModelURL.path)")
        } else {
            logger.log("📦 [INIT] No updated model found, loading bundled model...")
            // 加载内置模型
            let murl = bundle.url(forResource: "student_sequence_classification", withExtension: "mlpackage") ??
                bundle.url(forResource: "student_sequence_classification", withExtension: "mlmodelc")

            guard let url = murl else {
                logger.error("❌ [INIT] FATAL: Model file not found in bundle!")
                logger.log("🔍 [INIT] Searched for: student_sequence_classification.mlpackage or .mlmodelc")
                throw AnalysisError.modelNotLoaded
            }

            logger.log("📦 [INIT] Found bundled model at: \(url.path)")
            self.currentModelURL = url
            self.model = try MLModel(contentsOf: url, configuration: config)
            logger.log("✅ [INIT] Successfully loaded BUNDLED model from: \(url.path)")
        }

        logger.log("🔧 [INIT] Detecting model metadata...")
        guard let detectedMetadata = ModelMetadata.detect(from: model) else {
            logger.error("❌ [INIT] Failed to detect model metadata")
            throw AnalysisError.modelNotLoaded
        }

        self.metadata = detectedMetadata
        logger.log("✅ [INIT] Model metadata detected: shape=\(detectedMetadata.inputShape), maxLength=\(detectedMetadata.maxSequenceLength)")

        logger.log("📖 [INIT] Loading tokenizer vocabulary...")
        guard let vocabURL = bundle.url(forResource: "vocab", withExtension: "txt") else {
            logger.error("❌ [INIT] Vocabulary file (vocab.txt) not found in bundle")
            throw AnalysisError.modelNotLoaded
        }

        logger.log("📖 [INIT] Found vocab at: \(vocabURL.path)")
        self.tokenizer = try BertTokenizer(vocabURL: vocabURL)
        logger.log("✅ [INIT] Tokenizer initialized successfully")

        logger.log("🎉 [INIT] AdaptiveModelService initialization COMPLETE")
        logger
            .log(
                "📊 [INIT] Model config: maxLength=\(self.metadata.maxSequenceLength), shape=\(self.metadata.inputShape), computeUnits=cpuAndNeuralEngine"
            )
    }

    func analyzeGoal(_ text: String) async throws -> AnalysisResult {
        logger.log("🔮 [PREDICT] Starting goal analysis...")
        logger.log("📝 [PREDICT] Input text: '\(text.prefix(50))\(text.count > 50 ? "..." : "")'")

        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            logger.error("❌ [PREDICT] Empty input text")
            throw AnalysisError.invalidInput
        }

        let maxLength = metadata.maxSequenceLength
        logger.log("🔤 [PREDICT] Tokenizing with maxLength=\(maxLength)...")
        let enc = tokenizer.encode(text, maxLength: maxLength)

        logger.log("✅ [PREDICT] Tokenization complete: \(enc.tokens.count) tokens")
        logger.log("🔢 [PREDICT] Token IDs (first 10): \(Array(enc.ids.prefix(10)))")

        // 创建与模型期望维度完全匹配的MultiArray
        logger.log("🔧 [PREDICT] Creating MultiArrays with shape=\(self.metadata.inputShape)...")
        let ids = try createMultiArray(with: enc.ids, shape: metadata.inputShape)
        let mask = try createMultiArray(with: enc.mask, shape: metadata.inputShape)

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: ids),
            "attention_mask": MLFeatureValue(multiArray: mask)
        ])

        logger.log("⚡ [PREDICT] Running model inference...")
        let t0 = CFAbsoluteTimeGetCurrent()
        let out = try await model.prediction(from: input)
        let t1 = CFAbsoluteTimeGetCurrent()
        let inferenceTime = (t1 - t0) * 1000

        logger.log("✅ [PREDICT] Inference complete in \(String(format: "%.2f", inferenceTime))ms")

        let result = try parseResults(from: out)
        logger
            .log(
                "🎯 [PREDICT] Prediction result: category='\(result.category)' (conf=\(String(format: "%.3f", result.categoryConfidence))), sentiment='\(result.sentiment)' (score=\(String(format: "%.3f", result.sentimentScore)))"
            )

        return result
    }

    private func createMultiArray(with values: [Int32], shape: [Int]) throws -> MLMultiArray {
        let totalElements = shape.reduce(1, *)
        guard values.count <= totalElements else {
            throw AnalysisError.invalidInput
        }

        let multiArray = try MLMultiArray(shape: shape as [NSNumber], dataType: .int32)

        // 填充数据，确保维度匹配
        for (index, value) in values.enumerated() where index < totalElements {
            multiArray[index] = NSNumber(value: value)
        }

        // 填充剩余位置为pad token
        for i in values.count ..< totalElements {
            multiArray[i] = NSNumber(value: 0) // pad token
        }

        return multiArray
    }

    // 扩展到16个分类,匹配模型输出的 LABEL_0 到 LABEL_15
    // 基于常见的目标分类和产品定位
    private let categories = [
        "工作", // LABEL_0 - 工作相关目标
        "健康", // LABEL_1 - 健身、饮食、睡眠
        "家庭", // LABEL_2 - 家人、亲子
        "个人发展", // LABEL_3 - 自我提升、成长
        "财务", // LABEL_4 - 理财、储蓄
        "学习", // LABEL_5 - 学习、阅读、技能
        "社交", // LABEL_6 - 朋友、人际关系
        "休闲", // LABEL_7 - 娱乐、放松
        "创作", // LABEL_8 - 写作、艺术、创意
        "运动", // LABEL_9 - 体育、锻炼
        "旅行", // LABEL_10 - 旅游、探索
        "爱好", // LABEL_11 - 兴趣爱好
        "志愿", // LABEL_12 - 志愿服务、公益
        "精神", // LABEL_13 - 冥想、信仰
        "家务", // LABEL_14 - 家务、整理
        "其他" // LABEL_15 - 其他未分类
    ]
    private let sentiments = ["积极", "中性", "消极"]

    // 标签映射: LABEL_X -> 实际分类名称
    // 根据模型训练时的标签顺序映射
    private func mapLabel(_ label: String) -> String {
        // 提取标签索引
        if label.hasPrefix("LABEL_"),
           let indexStr = label.split(separator: "_").last,
           let index = Int(indexStr)
        {
            // 映射到预定义的分类
            if categories.indices.contains(index) {
                return categories[index]
            }
        }
        // 如果无法映射,返回原标签
        return label
    }

    private func parseResults(from output: MLFeatureProvider) throws -> AnalysisResult {
        if let probsResult = parseUsingProbs(output) { return probsResult }
        if let logitsResult = parseUsingLogits(output) { return logitsResult }
        if let labelResult = parseUsingClassLabel(output) { return labelResult }
        logger.error("❌ [PARSE] Cannot parse any output format")
        logger.log("🔍 [PARSE] Available features: \(output.featureNames.joined(separator: ", "))")
        throw AnalysisError.modelNotLoaded
    }

    private func applySoftmax(_ arr: MLMultiArray) -> [Double] {
        var maxv = -Double.infinity
        for i in 0 ..< arr.count {
            let value = arr[i].doubleValue
            if value > maxv { maxv = value }
        }

        var expValues: [Double] = []
        var sum = 0.0

        for i in 0 ..< arr.count {
            let expVal = exp(arr[i].doubleValue - maxv)
            expValues.append(expVal)
            sum += expVal
        }

        return expValues.map { $0 / sum }
    }

    private func softmax(_ logits: [Double]) -> [Double] {
        let maxLogit = logits.max() ?? 0.0
        let expValues = logits.map { exp($0 - maxLogit) }
        let sumExp = expValues.reduce(0, +)
        return expValues.map { $0 / sumExp }
    }

    private func parseUsingProbs(_ output: MLFeatureProvider) -> AnalysisResult? {
        guard let probsValue = output.featureValue(for: "classLabel_probs"),
              let probsDict = probsValue.dictionaryValue as? [String: Double] else { return nil }

        logger.log("✅ [PARSE] Found classLabel_probs (dictionary)")
        logger.log("📊 [PARSE] Raw logits: \(probsDict)")

        let labels = Array(probsDict.keys.sorted())
        let logits = labels.map { probsDict[$0] ?? 0.0 }
        let probs = softmax(logits)

        let labelProbs = zip(labels, probs).sorted { $0.1 > $1.1 }
        let mapped = labelProbs.map { (mapLabel($0.0), $0.1) }

        for (i, pair) in mapped.prefix(5).enumerated() {
            logger.log("   \(i + 1). \(labels[i]) (\(pair.0)): \(String(format: "%.3f", pair.1))")
        }

        guard let top = mapped.first else { return nil }
        let top5 = mapped.prefix(5).map { (name: $0.0, confidence: $0.1) }
        logger.log("📊 [PARSE] Returning Top-\(top5.count) categories for AI insights")

        return AnalysisResult(
            category: top.0,
            categoryConfidence: top.1,
            sentiment: "中性",
            sentimentScore: 0.5,
            topCategories: top5,
            topSentiments: [("中性", 0.5)]
        )
    }

    private func parseUsingLogits(_ output: MLFeatureProvider) -> AnalysisResult? {
        guard let logitsValue = output.featureValue(for: "var_331"),
              let logitsArray = logitsValue.multiArrayValue else { return nil }

        logger.log("✅ [PARSE] Found var_331 (MultiArray)")
        logger.log("📊 [PARSE] Logits shape: \(logitsArray.shape), count: \(logitsArray.count)")

        let probs = applySoftmax(logitsArray)
        logger.log("🎲 [PARSE] Probabilities (top 5): \(probs.prefix(5).map { String(format: "%.3f", $0) }.joined(separator: ", "))")

        guard let topCat = getTopK(probs, k: 1).first else { return nil }
        logger.log("🏆 [PARSE] Top category index: \(topCat.index), prob: \(String(format: "%.3f", topCat.probability))")

        let catLabel = categories.indices.contains(topCat.index) ? categories[topCat.index] : String(topCat.index)
        return AnalysisResult(
            category: catLabel,
            categoryConfidence: topCat.probability,
            sentiment: "中性",
            sentimentScore: 0.5
        )
    }

    private func parseUsingClassLabel(_ output: MLFeatureProvider) -> AnalysisResult? {
        guard let labelValue = output.featureValue(for: "classLabel") else { return nil }
        let label = labelValue.stringValue
        logger.log("✅ [PARSE] Found classLabel: \(label)")
        return AnalysisResult(
            category: label,
            categoryConfidence: 0.8,
            sentiment: "中性",
            sentimentScore: 0.5
        )
    }

    private func getTopK(_ probabilities: [Double], k: Int) -> [(index: Int, probability: Double)] {
        let indexedProbs = probabilities.enumerated().map { ($0.offset, $0.element) }
        let sorted = indexedProbs.sorted { $0.1 > $1.1 }
        return Array(sorted.prefix(k))
    }
}

// 扩展以支持协议
extension AdaptiveModelService: AnalysisService {
    func analyzeBatch(_ entries: [GoalEntry]) async throws -> [AnalysisResult] {
        var results: [AnalysisResult] = []
        for entry in entries {
            let result = try await analyzeGoal(entry.goalText)
            results.append(result)
        }
        return results
    }

    func updateModel(with corrections: [TrainingSample]) async throws {
        logger.log("🎓 [TRAIN] Starting on-device training...")
        logger.log("📊 [TRAIN] Training samples count: \(corrections.count)")

        for (idx, sample) in corrections.prefix(3).enumerated() {
            logger
                .log(
                    "📝 [TRAIN] Sample \(idx + 1): text='\(sample.text.prefix(30))...', category='\(sample.correctCategory)', sentiment='\(sample.correctSentiment)'"
                )
        }

        logger.log("🔧 [TRAIN] Preparing batch provider...")
        let trainingData = try prepareBatchProvider(from: corrections)
        let modelURL = self.currentModelURL

        logger.log("📂 [TRAIN] Source model: \(modelURL.path)")

        // 定义临时输出路径
        let fileManager = FileManager.default
        guard let docURL = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first else { throw AnalysisError.updateFailed }
        let updatedModelURL = docURL.appendingPathComponent("student_sequence_classification.mlmodelc")

        logger.log("💾 [TRAIN] Target model path: \(updatedModelURL.path)")
        logger.log("⚡ [TRAIN] Starting MLUpdateTask...")

        try await withCheckedThrowingContinuation { continuation in
            let task = try? MLUpdateTask(forModelAt: modelURL, trainingData: trainingData, configuration: nil) { context in
                if context.task.state == .completed {
                    self.logger.log("✅ [TRAIN] Training completed successfully")
                    do {
                        try context.model.write(to: updatedModelURL)
                        self.logger.log("💾 [TRAIN] Updated model saved to: \(updatedModelURL.path)")
                        continuation.resume()
                    } catch {
                        self.logger.error("❌ [TRAIN] Failed to save updated model: \(error.localizedDescription)")
                        continuation.resume(throwing: error)
                    }
                } else if context.task.state == .failed {
                    self.logger.error("❌ [TRAIN] Training failed: \(context.task.error?.localizedDescription ?? "unknown error")")
                    continuation.resume(throwing: context.task.error ?? AnalysisError.updateFailed)
                }
            }

            if let task = task {
                self.logger.log("▶️ [TRAIN] MLUpdateTask created and resumed")
                task.resume()
            } else {
                self.logger.error("❌ [TRAIN] Failed to create MLUpdateTask")
                continuation.resume(throwing: AnalysisError.updateFailed)
            }
        }

        // 重新加载模型
        logger.log("🔄 [TRAIN] Reloading updated model...")
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        self.model = try MLModel(contentsOf: updatedModelURL, configuration: config)
        self.currentModelURL = updatedModelURL
        logger.log("✅ [TRAIN] Model reloaded successfully, now using updated model")
        logger.log("🎉 [TRAIN] On-device training complete!")
    }

    private func prepareBatchProvider(from samples: [TrainingSample]) throws -> MLBatchProvider {
        var featureProviders: [MLFeatureProvider] = []

        for sample in samples {
            // Tokenize
            let enc = tokenizer.encode(sample.text, maxLength: metadata.maxSequenceLength)
            let ids = try createMultiArray(with: enc.ids, shape: metadata.inputShape)
            let mask = try createMultiArray(with: enc.mask, shape: metadata.inputShape)

            // Labels - 将分类名称映射为索引
            guard let catIndex = categories.firstIndex(of: sample.correctCategory) else {
                logger.error("❌ [TRAIN] Invalid category: '\(sample.correctCategory)' not found in categories list")
                logger.log("🔍 [TRAIN] Available categories: \(self.categories.joined(separator: ", "))")
                throw AnalysisError.invalidInput
            }

            guard let senIndex = sentiments.firstIndex(of: sample.correctSentiment) else {
                logger.error("❌ [TRAIN] Invalid sentiment: '\(sample.correctSentiment)' not found in sentiments list")
                logger.log("🔍 [TRAIN] Available sentiments: \(self.sentiments.joined(separator: ", "))")
                throw AnalysisError.invalidInput
            }

            logger.log("✅ [TRAIN] Mapped '\(sample.correctCategory)' -> index \(catIndex), '\(sample.correctSentiment)' -> index \(senIndex)")

            // Feature Dictionary
            // 假设模型训练输入名为 "category_true" 和 "sentiment_true"
            // 注意：这必须与mlpackage中的Training Input定义一致
            let features: [String: Any] = [
                "input_ids": ids,
                "attention_mask": mask,
                "category_true": NSNumber(value: catIndex),
                "sentiment_true": NSNumber(value: senIndex)
            ]

            if let provider = try? MLDictionaryFeatureProvider(dictionary: features) {
                featureProviders.append(provider)
            }
        }

        return MLArrayBatchProvider(array: featureProviders)
    }

    func getModelVersion() -> String {
        return "adaptive-v1.0"
    }
}
