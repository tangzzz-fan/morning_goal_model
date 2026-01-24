import CoreML
import Foundation
import Metal
import OSLog

@MainActor
final class CoreMLGoalAnalysisService: AnalysisService {
    private let model: MLModel
    private let tokenizer: BertTokenizer
    private let maxLength: Int
    private let categories: [String]
    private let sentiments: [String]
    private let logger = Logger(subsystem: "com.morninggoal.app", category: "ml")
    private let confidenceThreshold: Double = 0.7
    private let maxRetryAttempts: Int = 3

    init() throws {
        let bundle = Bundle.main
        // Explicitly load the compiled 8-bit optimized model
        guard let url = bundle.url(forResource: "GoalClassifier", withExtension: "mlmodelc") else {
            throw AnalysisError.modelNotLoaded
        }
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        config.preferredMetalDevice = MTLCreateSystemDefaultDevice()
        self.model = try MLModel(contentsOf: url, configuration: config)
        guard let vocabURL = bundle.url(forResource: "vocab", withExtension: "txt") else { throw AnalysisError.modelNotLoaded }
        self.tokenizer = try BertTokenizer(vocabURL: vocabURL)
        self.maxLength = 128
        if let viURL = bundle.url(forResource: "version_info", withExtension: "json"),
           let data = try? Data(contentsOf: viURL),
           let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let cats = obj["categories"] as? [String], let sens = obj["sentiments"] as? [String]
        {
            self.categories = cats
            self.sentiments = sens
        } else {
            self.categories = ["工作", "健康", "家庭", "个人发展", "财务", "学习", "社交", "休闲"]
            self.sentiments = ["积极", "中性", "消极"]
        }
    }

    func analyzeGoal(_ text: String) async throws -> AnalysisResult {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw AnalysisError.invalidInput
        }

        let enc = tokenizer.encode(text, maxLength: maxLength)
        logger.log("tokenized_text=\(enc.tokens.joined(separator: " | "))")

        let ids = try MLMultiArray(shape: [1, NSNumber(value: maxLength)], dataType: .int32)
        let mask = try MLMultiArray(shape: [1, NSNumber(value: maxLength)], dataType: .int32)

        for i in 0 ..< maxLength {
            ids[i] = NSNumber(value: enc.ids[i])
            mask[i] = NSNumber(value: enc.mask[i])
        }

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: ids),
            "attention_mask": MLFeatureValue(multiArray: mask)
        ])

        let t0 = CFAbsoluteTimeGetCurrent()
        let out = try await model.prediction(from: input)
        let t1 = CFAbsoluteTimeGetCurrent()

        logger.log("inference_ms=\(String(format: "%.2f", (t1 - t0) * 1000))")

        guard let cat = out.featureValue(for: "category_logits")?.multiArrayValue,
              let sen = out.featureValue(for: "sentiment_logits")?.multiArrayValue
        else {
            throw AnalysisError.modelNotLoaded
        }

        let catProbs = applySoftmax(cat)
        let senProbs = applySoftmax(sen)

        let topCatResults = getTopK(catProbs, k: 3)
        let topSenResults = getTopK(senProbs, k: 3)

        logger
            .log(
                "top_categories=\(topCatResults.map { "\(self.categories[$0.index]):\(String(format: "%.3f", $0.probability))" }.joined(separator: ", "))"
            )
        logger
            .log(
                "top_sentiments=\(topSenResults.map { "\(self.sentiments[$0.index]):\(String(format: "%.3f", $0.probability))" }.joined(separator: ", "))"
            )

        guard let bestCat = topCatResults.first,
              let bestSen = topSenResults.first
        else {
            throw AnalysisError.invalidInput
        }

        if bestCat.probability < confidenceThreshold || bestSen.probability < confidenceThreshold {
            logger.warning("low_confidence_detected cat=\(bestCat.probability) sen=\(bestSen.probability)")
        }

        let catLabel = categories.indices.contains(bestCat.index) ? categories[bestCat.index] : String(bestCat.index)
        let senLabel = sentiments.indices.contains(bestSen.index) ? sentiments[bestSen.index] : String(bestSen.index)

        return AnalysisResult(
            category: catLabel,
            categoryConfidence: bestCat.probability,
            sentiment: senLabel,
            sentimentScore: bestSen.probability
        )
    }

    func analyzeBatch(_ entries: [GoalEntry]) async throws -> [AnalysisResult] {
        var results: [AnalysisResult] = []
        for entry in entries {
            let result = try await analyzeGoal(entry.goalText)
            results.append(result)
        }
        return results
    }

    nonisolated func analyzeBatchIsolated(_ entries: [GoalEntry]) async throws -> [AnalysisResult] {
        return try await analyzeBatch(entries)
    }

    func updateModel(with corrections: [TrainingSample]) async throws { throw AnalysisError.updateFailed }
    func getModelVersion() -> String { return "mlprogram-v1" }
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

    private func getTopK(_ probabilities: [Double], k: Int) -> [(index: Int, probability: Double)] {
        let indexedProbs = probabilities.enumerated().map { ($0.offset, $0.element) }
        let sorted = indexedProbs.sorted { $0.1 > $1.1 }
        return Array(sorted.prefix(k))
    }

    private func argmax(_ arr: MLMultiArray) -> Int {
        var idx = 0
        var best = -Double.infinity
        for i in 0 ..< arr.count {
            let value = arr[i].doubleValue
            if value > best {
                best = value
                idx = i
            }
        }
        return idx
    }

    private func softmaxValue(_ arr: MLMultiArray, index: Int) -> Double {
        var maxv = -Double.infinity
        for i in 0 ..< arr.count {
            let value = arr[i].doubleValue
            if value > maxv { maxv = value }
        }

        var sum = 0.0
        for i in 0 ..< arr.count {
            sum += exp(arr[i].doubleValue - maxv)
        }

        return exp(arr[index].doubleValue - maxv) / sum
    }
}
