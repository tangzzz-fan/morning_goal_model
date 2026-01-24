import CoreML
import Foundation
import Metal
import OSLog

/// Updatable CoreML Goal Analysis Service
/// Supports on-device training via k-NN classifier with MLUpdateTask
@MainActor
final class UpdatableCoreMLGoalAnalysisService: AnalysisService {
    // MARK: - Properties

    /// Feature Extractor model (static, outputs embeddings)
    private var featureExtractor: MLModel

    /// k-NN Classifier model (updatable)
    private var knnClassifier: MLModel

    private let tokenizer: BertTokenizer
    private let maxLength: Int = 128
    let categories: [String]
    let sentiments: [String]

    private let logger = Logger(subsystem: "com.morninggoal.app", category: "updatable-ml")
    private let confidenceThreshold: Double = 0.7

    // MARK: - Model Paths

    /// Bundle path for original models
    private let bundleFEName = "FeatureExtractor"
    private let bundleKNNName = "KNNClassifier"

    /// Documents directory for updated models
    private var documentsDirectory: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    private var updatedKNNPath: URL {
        documentsDirectory.appendingPathComponent("UpdatedKNNClassifier.mlmodelc")
    }

    // MARK: - Initialization

    init() throws {
        let bundle = Bundle.main

        // 1. Load tokenizer
        guard let vocabURL = bundle.url(forResource: "vocab", withExtension: "txt") else {
            throw AnalysisError.modelNotLoaded
        }
        self.tokenizer = try BertTokenizer(vocabURL: vocabURL)

        // 2. Load Feature Extractor
        guard let feURL = bundle.url(forResource: bundleFEName, withExtension: "mlmodelc") else {
            throw AnalysisError.modelNotLoaded
        }
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        config.preferredMetalDevice = MTLCreateSystemDefaultDevice()
        self.featureExtractor = try MLModel(contentsOf: feURL, configuration: config)

        // 3. Load k-NN Classifier (prefer updated version if exists)
        // Compute path without using self (before all properties initialized)
        let documentsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let updatedPath = documentsDir.appendingPathComponent("UpdatedKNNClassifier.mlmodelc")

        if FileManager.default.fileExists(atPath: updatedPath.path) {
            self.knnClassifier = try MLModel(contentsOf: updatedPath, configuration: config)
        } else {
            guard let knnURL = bundle.url(forResource: bundleKNNName, withExtension: "mlmodelc") else {
                throw AnalysisError.modelNotLoaded
            }
            self.knnClassifier = try MLModel(contentsOf: knnURL, configuration: config)
        }

        // 4. Load metadata
        if let metaURL = bundle.url(forResource: "pipeline_metadata", withExtension: "json"),
           let data = try? Data(contentsOf: metaURL),
           let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let cats = obj["categories"] as? [String],
           let sens = obj["sentiments"] as? [String]
        {
            self.categories = cats
            self.sentiments = sens
        } else {
            self.categories = ["个人发展", "休闲", "健康", "学习", "家庭", "工作", "社交", "财务"]
            self.sentiments = ["中性", "消极", "积极"]
        }

        // Now we can use self safely
        if FileManager.default.fileExists(atPath: updatedPath.path) {
            logger.info("✅ Loaded updated k-NN model from Documents")
        } else {
            logger.info("✅ Loaded original k-NN model from Bundle")
        }
        logger.info("✅ Updatable ML Service initialized with \(self.categories.count) categories")
    }

    // MARK: - AnalysisService Protocol

    func analyzeGoal(_ text: String) async throws -> AnalysisResult {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw AnalysisError.invalidInput
        }

        let t0 = CFAbsoluteTimeGetCurrent()

        // 1. Tokenize
        let enc = tokenizer.encode(text, maxLength: maxLength)

        // 2. Prepare input tensors
        let ids = try MLMultiArray(shape: [1, NSNumber(value: maxLength)], dataType: .int32)
        let mask = try MLMultiArray(shape: [1, NSNumber(value: maxLength)], dataType: .int32)

        for i in 0 ..< maxLength {
            ids[i] = NSNumber(value: enc.ids[i])
            mask[i] = NSNumber(value: enc.mask[i])
        }

        // 3. Run Feature Extractor
        let feInput = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: ids),
            "attention_mask": MLFeatureValue(multiArray: mask)
        ])

        let feOutput = try await featureExtractor.prediction(from: feInput)

        guard let embedding = feOutput.featureValue(for: "embedding")?.multiArrayValue,
              let sentimentLogits = feOutput.featureValue(for: "sentiment_logits")?.multiArrayValue
        else {
            throw AnalysisError.modelNotLoaded
        }

        // 4. Run k-NN Classifier with embedding
        // Embedding is already shape [768] from Feature Extractor
        let knnInput = try MLDictionaryFeatureProvider(dictionary: [
            "embedding": MLFeatureValue(multiArray: embedding)
        ])

        let knnOutput = try await knnClassifier.prediction(from: knnInput)

        // 5. Extract results
        guard let category = knnOutput.featureValue(for: "category")?.stringValue else {
            throw AnalysisError.modelNotLoaded
        }

        // Get category probability (if available)
        var categoryConfidence = 0.8 // default
        if let probsDict = knnOutput.featureValue(for: "categoryProbs")?.dictionaryValue as? [String: Double] {
            categoryConfidence = probsDict[category] ?? 0.8
        }

        // 6. Process sentiment logits
        let senProbs = applySoftmax(sentimentLogits)
        let topSenResults = getTopK(senProbs, k: 3)

        guard let bestSen = topSenResults.first else {
            throw AnalysisError.invalidInput
        }

        let senLabel = sentiments.indices.contains(bestSen.index) ? sentiments[bestSen.index] : "中性"

        let t1 = CFAbsoluteTimeGetCurrent()
        logger.log("inference_ms=\(String(format: "%.2f", (t1 - t0) * 1000)), category=\(category), sentiment=\(senLabel)")

        return AnalysisResult(
            category: category,
            categoryConfidence: categoryConfidence,
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

    // MARK: - On-Device Training

    func updateModel(with corrections: [TrainingSample]) async throws {
        guard !corrections.isEmpty else {
            logger.warning("No corrections provided for update")
            return
        }

        logger.info("🔄 Starting on-device model update with \(corrections.count) samples...")

        // 1. Get the updatable model URL (from bundle for base, or documents if already updated)
        let baseModelURL: URL
        if let bundleURL = Bundle.main.url(forResource: bundleKNNName, withExtension: "mlmodelc") {
            baseModelURL = bundleURL
        } else {
            throw AnalysisError.modelNotLoaded
        }

        // 2. Generate embeddings for correction samples
        let trainingData = try await prepareTrainingData(corrections)

        // 3. Create update task
        let updateTask = try MLUpdateTask(
            forModelAt: baseModelURL,
            trainingData: trainingData,
            configuration: nil,
            completionHandler: { [weak self] context in
                self?.handleUpdateCompletion(context: context)
            }
        )

        // 4. Execute update
        updateTask.resume()

        logger.info("📤 Update task submitted")
    }

    private func handleUpdateCompletion(context: MLUpdateContext) {
        switch context.task.state {
        case .completed:
            logger.info("✅ Model update completed successfully")
            do {
                // Save updated model to Documents
                try context.model.write(to: updatedKNNPath)
                logger.info("💾 Updated model saved to: \(self.updatedKNNPath.path)")

                // Reload the model
                Task { @MainActor in
                    try? self.reloadKNNModel()
                }
            } catch {
                logger.error("❌ Failed to save updated model: \(error.localizedDescription)")
            }

        case .failed:
            logger.error("❌ Model update failed: \(context.task.error?.localizedDescription ?? "unknown")")

        default:
            break
        }
    }

    private func reloadKNNModel() throws {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        self.knnClassifier = try MLModel(contentsOf: updatedKNNPath, configuration: config)
        logger.info("🔄 Reloaded updated k-NN model")
    }

    private func prepareTrainingData(_ samples: [TrainingSample]) async throws -> MLBatchProvider {
        var featureProviders: [MLFeatureProvider] = []

        for sample in samples {
            // Tokenize
            let enc = tokenizer.encode(sample.text, maxLength: maxLength)

            // Create input tensors
            let ids = try MLMultiArray(shape: [1, NSNumber(value: maxLength)], dataType: .int32)
            let mask = try MLMultiArray(shape: [1, NSNumber(value: maxLength)], dataType: .int32)

            for i in 0 ..< maxLength {
                ids[i] = NSNumber(value: enc.ids[i])
                mask[i] = NSNumber(value: enc.mask[i])
            }

            // Get embedding from Feature Extractor
            let feInput = try MLDictionaryFeatureProvider(dictionary: [
                "input_ids": MLFeatureValue(multiArray: ids),
                "attention_mask": MLFeatureValue(multiArray: mask)
            ])

            let feOutput = try await featureExtractor.prediction(from: feInput)

            guard let embedding = feOutput.featureValue(for: "embedding")?.multiArrayValue else {
                continue
            }

            // Create training sample with embedding and label
            // Embedding is already shape [768] from Feature Extractor
            let trainingFeatures = try MLDictionaryFeatureProvider(dictionary: [
                "embedding": MLFeatureValue(multiArray: embedding),
                "category": MLFeatureValue(string: sample.correctCategory)
            ])

            featureProviders.append(trainingFeatures)
        }

        logger.info("📊 Prepared \(featureProviders.count) training samples")
        return MLArrayBatchProvider(array: featureProviders)
    }

    // MARK: - Model Management

    func getModelVersion() -> String {
        let isUpdated = FileManager.default.fileExists(atPath: updatedKNNPath.path)
        return isUpdated ? "updatable-knn-v1-updated" : "updatable-knn-v1"
    }

    func resetModel() throws {
        // Delete updated model to revert to bundle version
        if FileManager.default.fileExists(atPath: updatedKNNPath.path) {
            try FileManager.default.removeItem(at: updatedKNNPath)
            logger.info("🗑️ Deleted updated model, reverting to bundle version")

            // Reload from bundle
            guard let knnURL = Bundle.main.url(forResource: bundleKNNName, withExtension: "mlmodelc") else {
                throw AnalysisError.modelNotLoaded
            }
            let config = MLModelConfiguration()
            config.computeUnits = .cpuAndNeuralEngine
            self.knnClassifier = try MLModel(contentsOf: knnURL, configuration: config)
        }
    }

    // Note: flattenEmbedding is no longer needed as model outputs [768] directly
    // Kept for backward compatibility if needed
    private func flattenEmbedding(_ embedding: MLMultiArray) throws -> MLMultiArray {
        // If already 1D [768], return as-is
        if embedding.shape.count == 1 {
            return embedding
        }
        // Otherwise convert [1, 768] to [768]
        let flatArray = try MLMultiArray(shape: [768], dataType: .float32)
        for i in 0 ..< 768 {
            flatArray[i] = embedding[i]
        }
        return flatArray
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

    private func getTopK(_ probabilities: [Double], k: Int) -> [(index: Int, probability: Double)] {
        let indexedProbs = probabilities.enumerated().map { ($0.offset, $0.element) }
        let sorted = indexedProbs.sorted { $0.1 > $1.1 }
        return Array(sorted.prefix(k))
    }
}
