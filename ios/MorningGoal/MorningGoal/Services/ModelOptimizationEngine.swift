import CoreML
import Foundation
import OSLog

struct ModelConfiguration {
    let maxSequenceLength: Int
    let confidenceThreshold: Double
    let topK: Int
    let temperature: Double
    let computeUnits: MLComputeUnits
    let enableQuantization: Bool
    let batchSize: Int

    static let `default` = ModelConfiguration(
        maxSequenceLength: 256,
        confidenceThreshold: 0.7,
        topK: 3,
        temperature: 1.0,
        computeUnits: .cpuAndNeuralEngine,
        enableQuantization: true,
        batchSize: 1
    )

    static let highQuality = ModelConfiguration(
        maxSequenceLength: 512,
        confidenceThreshold: 0.8,
        topK: 5,
        temperature: 0.8,
        computeUnits: .all,
        enableQuantization: false,
        batchSize: 1
    )

    static let fast = ModelConfiguration(
        maxSequenceLength: 128,
        confidenceThreshold: 0.6,
        topK: 1,
        temperature: 1.2,
        computeUnits: .cpuOnly,
        enableQuantization: true,
        batchSize: 4
    )
}

enum ModelOptimizationStrategy {
    case accuracy
    case balanced
    case speed

    var configuration: ModelConfiguration {
        switch self {
        case .accuracy:
            return .highQuality
        case .balanced:
            return .default
        case .speed:
            return .fast
        }
    }
}

final class ModelOptimizationEngine {
    private let logger = Logger(subsystem: "com.morninggoal.app", category: "optimization")
    private var currentConfig: ModelConfiguration

    init(strategy: ModelOptimizationStrategy = .balanced) {
        self.currentConfig = strategy.configuration
    }

    func optimizePrediction(_ probabilities: [Double], for task: String) -> [(index: Int, probability: Double)] {
        logger.log("optimizing_prediction task=\(task) temp=\(self.currentConfig.temperature)")

        var adjustedProbs = probabilities

        if currentConfig.temperature != 1.0 {
            adjustedProbs = applyTemperature(adjustedProbs, temperature: currentConfig.temperature)
        }

        let topResults = getTopK(adjustedProbs, k: currentConfig.topK)

        return topResults.filter { $0.probability >= currentConfig.confidenceThreshold }
    }

    func shouldRetry(_ confidence: Double) -> Bool {
        return confidence < currentConfig.confidenceThreshold
    }

    func getOptimalBatchSize() -> Int {
        return currentConfig.batchSize
    }

    func updateConfiguration(_ newConfig: ModelConfiguration) {
        logger.log("updating_configuration maxLength=\(newConfig.maxSequenceLength) threshold=\(newConfig.confidenceThreshold)")
        self.currentConfig = newConfig
    }

    private func applyTemperature(_ probabilities: [Double], temperature: Double) -> [Double] {
        let scaledLogits = probabilities.map { log(max($0, 1e-8)) / temperature }
        let maxLogit = scaledLogits.max() ?? 0
        let expValues = scaledLogits.map { exp($0 - maxLogit) }
        let sum = expValues.reduce(0, +)
        return expValues.map { $0 / sum }
    }

    private func getTopK(_ probabilities: [Double], k: Int) -> [(index: Int, probability: Double)] {
        let indexedProbs = probabilities.enumerated().map { ($0.offset, $0.element) }
        let sorted = indexedProbs.sorted { $0.1 > $1.1 }
        return Array(sorted.prefix(k))
    }
}
