import CoreData
import CoreML
import Foundation
import OSLog

final class MultiModelEnsembleService: AnalysisService {
    private let models: [CoreMLGoalAnalysisService]
    private let optimizationEngine: ModelOptimizationEngine
    private let logger = Logger(subsystem: "com.morninggoal.app", category: "ensemble")

    init() throws {
        self.optimizationEngine = ModelOptimizationEngine(strategy: .accuracy)

        var modelServices: [CoreMLGoalAnalysisService] = []

        for i in 0 ..< 3 {
            do {
                let service = try CoreMLGoalAnalysisService()
                modelServices.append(service)
                logger.log("loaded_ensemble_model index=\(i)")
            } catch {
                logger.error("failed_to_load_ensemble_model index=\(i) error=\(error.localizedDescription)")
            }
        }

        guard !modelServices.isEmpty else {
            throw AnalysisError.modelNotLoaded
        }

        self.models = modelServices
    }

    func analyzeGoal(_ text: String) async throws -> GoalAnalysisResult {
        logger.log("starting_ensemble_analysis text_length=\(text.count)")

        var results: [GoalAnalysisResult] = []
        var errors: [Error] = []

        for (index, model) in models.enumerated() {
            do {
                let result = try await model.analyzeGoal(text)
                results.append(result)
                logger.log("model_\(index)_completed confidence=\(result.categoryConfidence)")
            } catch {
                errors.append(error)
                logger.error("model_\(index)_failed error=\(error.localizedDescription)")
            }
        }

        guard !results.isEmpty else {
            logger.error("all_models_failed errors=\(errors.count)")
            throw AnalysisError.modelNotLoaded
        }

        return ensembleResults(results)
    }

    func analyzeBatch(_ entries: [GoalEntry]) async throws -> [GoalAnalysisResult] {
        var batchResults: [GoalAnalysisResult] = []

        for entry in entries {
            let result = try await analyzeGoal(entry.goalText)
            batchResults.append(result)
        }

        return batchResults
    }

    nonisolated func analyzeBatchIsolated(_ entries: [GoalEntry]) async throws -> [GoalAnalysisResult] {
        return try await analyzeBatch(entries)
    }

    func updateModel(with corrections: [TrainingSample]) async throws {
        logger.log("ensemble_update_not_supported samples=\(corrections.count)")
        throw AnalysisError.updateFailed
    }

    func getModelVersion() -> String {
        return "ensemble-v1.0"
    }

    private func ensembleResults(_ results: [GoalAnalysisResult]) -> GoalAnalysisResult {
        let categoryVotes = aggregateVotes(results.map { ($0.category, $0.categoryConfidence) })
        let sentimentVotes = aggregateVotes(results.map { ($0.sentiment, $0.sentimentScore) })

        guard let bestCategory = categoryVotes.first,
              let bestSentiment = sentimentVotes.first
        else {
            return results.first ?? GoalAnalysisResult(
                category: "其他",
                categoryConfidence: 0.5,
                sentiment: "中性",
                sentimentScore: 0.5
            )
        }

        let avgCatConfidence = results.map { $0.categoryConfidence }.reduce(0, +) / Double(results.count)
        let avgSenConfidence = results.map { $0.sentimentScore }.reduce(0, +) / Double(results.count)

        logger.log("ensemble_decision category=\(bestCategory.key) confidence=\(bestCategory.value) models_used=\(results.count)")

        // For simplicity, we just use the first model's top categories if we can't merge them easily
        // Or we could merge them.
        let topCats = results.first?.topCategories
        let topSens = results.first?.topSentiments

        return GoalAnalysisResult(
            category: bestCategory.key,
            categoryConfidence: bestCategory.value, // This is voting score, not raw probability
            sentiment: bestSentiment.key,
            sentimentScore: bestSentiment.value,
            topCategories: topCats,
            topSentiments: topSens
        )
    }

    private func aggregateVotes(_ votes: [(key: String, confidence: Double)]) -> [String: Double] {
        var aggregated: [String: (sum: Double, count: Int)] = [:]

        for vote in votes {
            if let existing = aggregated[vote.key] {
                aggregated[vote.key] = (existing.sum + vote.confidence, existing.count + 1)
            } else {
                aggregated[vote.key] = (vote.confidence, 1)
            }
        }

        return aggregated.mapValues { $0.sum / Double($0.count) }
            .sorted { $0.value > $1.value }
            .reduce(into: [:]) { $0[$1.key] = $1.value }
    }
}
