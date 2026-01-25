import CoreML
import Foundation
import OSLog

@MainActor
final class AdaptiveModelService {
    private let insightManager: InsightModelManager
    private let logger = Logger(subsystem: "com.morninggoal.app", category: "adaptive-service")

    init(bundle: Bundle = Bundle(for: AdaptiveModelService.self)) throws {
        logger.log("🚀 [INIT] Initializing AdaptiveModelService with InsightModelManager...")
        self.insightManager = InsightModelManager()
    }

    func analyzeGoal(_ text: String) async throws -> GoalAnalysisResult {
        logger.log("🔮 [PREDICT] Starting goal analysis: '\(text.prefix(20))...'")

        // Ensure models are loaded
        if !insightManager.isInitialized {
            logger.log("⏳ [PREDICT] Models not initialized, loading now...")
            await insightManager.loadModels()
        }

        // Use InsightModelManager for analysis
        let insightResult = try await insightManager.analyze(text: text)

        // Map Topic to Category
        let categoryLabel = insightResult.topic?.label ?? "其他"
        let categoryConf = insightResult.topic?.confidence ?? 0.0

        // Map Sentiment
        let sentimentLabel = insightResult.sentiment?.label ?? "中性"
        let sentimentConf = insightResult.sentiment?.confidence ?? 0.0

        // Map Top Categories
        let topCategories = insightResult.topic?.allProbabilities
            .map { ($0.key, $0.value) }
            .sorted { $0.1 > $1.1 }
            .prefix(5)
            .map { (name: $0.0, confidence: $0.1) } ?? []

        // Map Top Sentiments
        let topSentiments = insightResult.sentiment?.allProbabilities
            .map { ($0.key, $0.value) }
            .sorted { $0.1 > $1.1 }
            .map { (name: $0.0, score: $0.1) } ?? []

        logger.log("✅ [PREDICT] Result: \(categoryLabel) (\(categoryConf)), \(sentimentLabel) (\(sentimentConf))")

        return GoalAnalysisResult(
            from: insightResult,
            topCategories: Array(topCategories),
            topSentiments: Array(topSentiments)
        )
    }

    func getModelVersion() -> String {
        let versions = insightManager.getModelVersions()
        let loadedCount = versions.values.filter { $0 != "not_loaded" }.count
        return "adaptive-v2.0-fanout (\(loadedCount)/8 models loaded)"
    }
}

// MARK: - Protocol Conformance

extension AdaptiveModelService: AnalysisService {
    func analyzeBatch(_ entries: [GoalEntry]) async throws -> [GoalAnalysisResult] {
        var results: [GoalAnalysisResult] = []
        for entry in entries {
            let result = try await analyzeGoal(entry.goalText)
            results.append(result)
        }
        return results
    }

    func updateModel(with corrections: [TrainingSample]) async throws {
        logger.warning("⚠️ [TRAIN] On-device training is not yet implemented for the fan-out architecture.")
        // TODO: Implement training logic for individual classifiers using InsightModelManager
        throw AnalysisError.updateFailed
    }
}
