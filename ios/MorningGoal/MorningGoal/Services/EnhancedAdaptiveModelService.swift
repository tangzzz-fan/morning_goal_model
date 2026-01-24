import CoreML
import Foundation
import OSLog

/// 增强的自适应模型服务
@MainActor
final class EnhancedAdaptiveModelService {
    private let model: MLModel
    private let tokenizer: EnhancedChineseBertTokenizer
    private let featureEngineer: IntelligentFeatureEngineeringService
    private let metadata: ModelMetadata
    private let logger = Logger(subsystem: "com.morninggoal.app", category: "enhanced-adaptive")

    // 优化参数
    private let confidenceThreshold: Double = 0.75
    private let temperature: Double = 0.8
    private let topK: Int = 5
    private let ensembleWeight: Double = 0.3

    // 类别和情感映射
    private let categories: [String]
    private let sentiments: [String]

    init() throws {
        let bundle = Bundle.main
        let murl = bundle.url(forResource: "GoalClassifier", withExtension: "mlpackage") ??
            bundle.url(forResource: "GoalClassifier", withExtension: "mlmodelc")

        guard let url = murl else {
            throw AnalysisError.modelNotLoaded
        }

        // 增强的模型配置
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        config.preferredMetalDevice = MTLCreateSystemDefaultDevice()

        self.model = try MLModel(contentsOf: url, configuration: config)

        guard let detectedMetadata = ModelMetadata.detect(from: model) else {
            throw AnalysisError.modelNotLoaded
        }

        self.metadata = detectedMetadata

        guard let vocabURL = bundle.url(forResource: "vocab", withExtension: "txt") else {
            throw AnalysisError.modelNotLoaded
        }

        self.tokenizer = try EnhancedChineseBertTokenizer(vocabURL: vocabURL)
        self.featureEngineer = IntelligentFeatureEngineeringService()

        // 设置类别和情感标签
        self.categories = ["工作", "健康", "家庭", "个人发展", "财务", "学习", "社交", "休闲"]
        self.sentiments = ["积极", "中性", "消极"]

        logger.log("initialized_enhanced_adaptive_model max_length=\(self.metadata.maxSequenceLength) temp=\(self.temperature) topK=\(self.topK)")
    }

    func analyzeGoal(_ text: String) async throws -> AnalysisResult {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw AnalysisError.invalidInput
        }

        let startTime = CFAbsoluteTimeGetCurrent()

        // 1. 多维度特征提取
        let features = featureEngineer.extractFeatures(from: text)
        logger.log("extracted_features basic_length=\(features.basicFeatures.length) sentiment_score=\(features.sentimentFeatures.sentimentScore)")

        // 2. 增强的tokenization
        let maxLength = metadata.maxSequenceLength
        let enhancedOutput = tokenizer.encode(text, maxLength: maxLength)
        logger
            .log(
                "enhanced_tokenization tokens=\(enhancedOutput.tokens.count) sentiment=\(enhancedOutput.sentimentScore) categories=\(enhancedOutput.categoryHints.keys.joined(separator: ","))"
            )

        // 3. 创建增强的模型输入
        let modelInput = try createEnhancedModelInput(
            tokenIds: enhancedOutput.ids,
            mask: enhancedOutput.mask,
            features: features,
            enhancedOutput: enhancedOutput
        )

        // 4. 模型推理
        let inferenceStart = CFAbsoluteTimeGetCurrent()
        let modelOutput = try await model.prediction(from: modelInput)
        let inferenceEnd = CFAbsoluteTimeGetCurrent()

        // 5. 多维度结果融合
        let fusedResult = try fuseMultiDimensionalResults(
            modelOutput: modelOutput,
            features: features,
            enhancedOutput: enhancedOutput
        )

        let endTime = CFAbsoluteTimeGetCurrent()
        logger
            .log(
                "enhanced_inference_completed total_time=\(String(format: "%.2f", (endTime - startTime) * 1000))ms model_time=\(String(format: "%.2f", (inferenceEnd - inferenceStart) * 1000))ms"
            )

        return fusedResult
    }

    private func createEnhancedModelInput(
        tokenIds: [Int32],
        mask: [Int32],
        features: IntelligentFeatureEngineeringService.TextFeatures,
        enhancedOutput: EnhancedTokenizedOutput
    ) throws -> MLDictionaryFeatureProvider {
        // 基础MultiArray
        let ids = try createMultiArray(with: tokenIds, shape: metadata.inputShape)
        let attentionMask = try createMultiArray(with: mask, shape: metadata.inputShape)

        // 增强特征MultiArray
        let featureArray = try createFeatureArray(from: features, enhancedOutput: enhancedOutput)

        var inputDict: [String: MLFeatureValue] = [
            "input_ids": MLFeatureValue(multiArray: ids),
            "attention_mask": MLFeatureValue(multiArray: attentionMask)
        ]

        // 如果模型支持增强特征
        if model.modelDescription.inputDescriptionsByName["feature_vector"] != nil {
            inputDict["feature_vector"] = MLFeatureValue(multiArray: featureArray)
        }

        return try MLDictionaryFeatureProvider(dictionary: inputDict)
    }

    private func createFeatureArray(
        from features: IntelligentFeatureEngineeringService.TextFeatures,
        enhancedOutput: EnhancedTokenizedOutput
    ) throws -> MLMultiArray {
        var featureVector: [Double] = []

        // 基础特征
        featureVector.append(Double(features.basicFeatures.length))
        featureVector.append(Double(features.basicFeatures.wordCount))
        featureVector.append(features.basicFeatures.avgWordLength)
        featureVector.append(Double(features.basicFeatures.punctuationCount))
        featureVector.append(features.basicFeatures.chineseCharRatio)

        // 情感特征
        featureVector.append(Double(features.sentimentFeatures.positiveWords))
        featureVector.append(Double(features.sentimentFeatures.negativeWords))
        featureVector.append(features.sentimentFeatures.sentimentScore)

        // 类别特征
        featureVector.append(Double(features.categoryFeatures.workKeywords))
        featureVector.append(Double(features.categoryFeatures.healthKeywords))
        featureVector.append(Double(features.categoryFeatures.familyKeywords))
        featureVector.append(Double(features.categoryFeatures.studyKeywords))
        featureVector.append(Double(features.categoryFeatures.financeKeywords))
        featureVector.append(Double(features.categoryFeatures.socialKeywords))
        featureVector.append(Double(features.categoryFeatures.leisureKeywords))
        featureVector.append(Double(features.categoryFeatures.personalKeywords))

        // 增强tokenizer特征
        featureVector.append(enhancedOutput.sentimentScore)
        featureVector.append(Double(enhancedOutput.keywords.count))

        // 填充到固定长度（128维）
        while featureVector.count < 128 {
            featureVector.append(0.0)
        }

        return try MLMultiArray(shape: [NSNumber(value: featureVector.count)], dataType: .double)
    }

    private func fuseMultiDimensionalResults(
        modelOutput: MLFeatureProvider,
        features: IntelligentFeatureEngineeringService.TextFeatures,
        enhancedOutput: EnhancedTokenizedOutput
    ) throws -> AnalysisResult {
        // 1. 模型输出解析
        guard let catValue = modelOutput.featureValue(for: "category_logits"),
              let senValue = modelOutput.featureValue(for: "sentiment_logits")
        else {
            throw AnalysisError.invalidInput
        }

        guard let catMultiArray = catValue.multiArrayValue,
              let senMultiArray = senValue.multiArrayValue
        else {
            throw AnalysisError.invalidInput
        }

        let catProbs = applyTemperatureSoftmax(catMultiArray, temperature: temperature)
        let senProbs = applyTemperatureSoftmax(senMultiArray, temperature: temperature)

        // 2. 特征工程预测
        let featureBasedPrediction = predictFromFeatures(features)

        // 3. 增强tokenizer预测
        let tokenizerBasedPrediction = predictFromTokenizer(enhancedOutput)

        // 4. 多维度融合
        let fusedCatProbs = fuseCategoryPredictions(
            modelProbs: catProbs,
            featureProbs: featureBasedPrediction.categoryProbs,
            tokenizerProbs: tokenizerBasedPrediction.categoryProbs
        )

        let fusedSenProbs = fuseSentimentPredictions(
            modelProbs: senProbs,
            featureProbs: featureBasedPrediction.sentimentProbs,
            tokenizerProbs: tokenizerBasedPrediction.sentimentProbs
        )

        // 5. 最终结果
        guard let topCat = getTopK(fusedCatProbs, k: 1).first,
              let topSen = getTopK(fusedSenProbs, k: 1).first
        else {
            throw AnalysisError.invalidInput
        }

        let catLabel = categories.indices.contains(topCat.index) ? categories[topCat.index] : String(topCat.index)
        let senLabel = sentiments.indices.contains(topSen.index) ? sentiments[topSen.index] : String(topSen.index)

        logger
            .log(
                "fused_prediction category=\(catLabel)(\(String(format: "%.3f", topCat.probability))) sentiment=\(senLabel)(\(String(format: "%.3f", topSen.probability)))"
            )

        return AnalysisResult(
            category: catLabel,
            categoryConfidence: topCat.probability,
            sentiment: senLabel,
            sentimentScore: topSen.probability
        )
    }

    private func predictFromFeatures(_ features: IntelligentFeatureEngineeringService
        .TextFeatures) -> (categoryProbs: [Double], sentimentProbs: [Double])
    {
        // 基于特征的简单预测
        var categoryProbs = Array(repeating: 0.0, count: categories.count)
        var sentimentProbs = Array(repeating: 0.0, count: sentiments.count)

        // 类别预测
        categoryProbs[0] = Double(features.categoryFeatures.workKeywords) * 0.1
        categoryProbs[1] = Double(features.categoryFeatures.healthKeywords) * 0.1
        categoryProbs[2] = Double(features.categoryFeatures.familyKeywords) * 0.1
        categoryProbs[3] = Double(features.categoryFeatures.personalKeywords) * 0.1
        categoryProbs[4] = Double(features.categoryFeatures.financeKeywords) * 0.1
        categoryProbs[5] = Double(features.categoryFeatures.studyKeywords) * 0.1
        categoryProbs[6] = Double(features.categoryFeatures.socialKeywords) * 0.1
        categoryProbs[7] = Double(features.categoryFeatures.leisureKeywords) * 0.1

        // 归一化
        let catSum = categoryProbs.reduce(0, +)
        if catSum > 0 {
            categoryProbs = categoryProbs.map { $0 / catSum }
        }

        // 情感预测
        if features.sentimentFeatures.sentimentScore > 0.2 {
            sentimentProbs[0] = 0.7 // 积极
            sentimentProbs[1] = 0.2 // 中性
            sentimentProbs[2] = 0.1 // 消极
        } else if features.sentimentFeatures.sentimentScore < -0.2 {
            sentimentProbs[0] = 0.1
            sentimentProbs[1] = 0.2
            sentimentProbs[2] = 0.7
        } else {
            sentimentProbs[0] = 0.2
            sentimentProbs[1] = 0.6
            sentimentProbs[2] = 0.2
        }

        return (categoryProbs, sentimentProbs)
    }

    private func predictFromTokenizer(_ enhancedOutput: EnhancedTokenizedOutput) -> (categoryProbs: [Double], sentimentProbs: [Double]) {
        var categoryProbs = Array(repeating: 0.0, count: categories.count)
        var sentimentProbs = Array(repeating: 0.0, count: sentiments.count)

        // 基于tokenizer的类别提示
        for (category, score) in enhancedOutput.categoryHints {
            if let index = categories.firstIndex(of: category) {
                categoryProbs[index] = score
            }
        }

        // 基于情感分数的情感预测
        if enhancedOutput.sentimentScore > 0.1 {
            sentimentProbs[0] = 0.6 // 积极
            sentimentProbs[1] = 0.3
            sentimentProbs[2] = 0.1
        } else if enhancedOutput.sentimentScore < -0.1 {
            sentimentProbs[0] = 0.1
            sentimentProbs[1] = 0.3
            sentimentProbs[2] = 0.6 // 消极
        } else {
            sentimentProbs[0] = 0.2
            sentimentProbs[1] = 0.6 // 中性
            sentimentProbs[2] = 0.2
        }

        return (categoryProbs, sentimentProbs)
    }

    private func fuseCategoryPredictions(
        modelProbs: [Double],
        featureProbs: [Double],
        tokenizerProbs: [Double]
    ) -> [Double] {
        var fusedProbs: [Double] = []

        for i in 0 ..< modelProbs.count {
            let modelWeight = 0.6
            let featureWeight = 0.25
            let tokenizerWeight = 0.15

            let tokenizerScore = i < tokenizerProbs.count ? tokenizerProbs[i] : 0.0

            let fused = modelProbs[i] * modelWeight +
                featureProbs[i] * featureWeight +
                tokenizerScore * tokenizerWeight

            fusedProbs.append(fused)
        }

        // 重新归一化
        let sum = fusedProbs.reduce(0, +)
        return sum > 0 ? fusedProbs.map { $0 / sum } : modelProbs
    }

    private func fuseSentimentPredictions(
        modelProbs: [Double],
        featureProbs: [Double],
        tokenizerProbs: [Double]
    ) -> [Double] {
        let modelWeight = 0.5
        let featureWeight = 0.3
        let tokenizerWeight = 0.2

        var fusedProbs: [Double] = []

        for i in 0 ..< modelProbs.count {
            let tokenizerProb = i < tokenizerProbs.count ? tokenizerProbs[i] : 0.0

            let fused = modelProbs[i] * modelWeight +
                featureProbs[i] * featureWeight +
                tokenizerProb * tokenizerWeight

            fusedProbs.append(fused)
        }

        // 重新归一化
        let sum = fusedProbs.reduce(0, +)
        return sum > 0 ? fusedProbs.map { $0 / sum } : modelProbs
    }

    private func applyTemperatureSoftmax(_ arr: MLMultiArray, temperature: Double) -> [Double] {
        var logits: [Double] = []
        for i in 0 ..< arr.count {
            logits.append(arr[i].doubleValue / temperature)
        }

        let maxLogit = logits.max() ?? 0
        var expValues: [Double] = []
        var sum = 0.0

        for logit in logits {
            let expVal = exp(logit - maxLogit)
            expValues.append(expVal)
            sum += expVal
        }

        return expValues.map { $0 / sum }
    }

    private func getTopK(_ probabilities: [Double], k: Int) -> [(index: Int, probability: Double)] {
        let indexedProbs = probabilities.enumerated().map { ($0.offset, $0.element) }
        let sorted = indexedProbs.sorted { $0.1 > $1.1 }
        return Array(sorted.prefix(k))
    }

    private func createMultiArray(with values: [Int32], shape: [Int]) throws -> MLMultiArray {
        let totalElements = shape.reduce(1, *)
        guard values.count <= totalElements else {
            throw AnalysisError.invalidInput
        }

        let multiArray = try MLMultiArray(shape: shape as [NSNumber], dataType: .int32)

        for (index, value) in values.enumerated() where index < totalElements {
            multiArray[index] = NSNumber(value: value)
        }

        // 填充剩余位置
        for i in values.count ..< totalElements {
            multiArray[i] = NSNumber(value: 0)
        }

        return multiArray
    }
}

// MARK: - 协议扩展

extension EnhancedAdaptiveModelService: AnalysisService {
    func analyzeBatch(_ entries: [GoalEntry]) async throws -> [AnalysisResult] {
        var results: [AnalysisResult] = []
        for entry in entries {
            let result = try await analyzeGoal(entry.goalText)
            results.append(result)
        }
        return results
    }

    func updateModel(with corrections: [TrainingSample]) async throws {
        logger.log("model_update_not_supported samples=\(corrections.count)")
        throw AnalysisError.updateFailed
    }

    func getModelVersion() -> String {
        return "enhanced-adaptive-v2.0"
    }
}
