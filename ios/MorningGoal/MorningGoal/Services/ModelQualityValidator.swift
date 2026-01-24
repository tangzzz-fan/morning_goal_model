import CoreData
import Foundation
import MachO
import OSLog

final class ModelQualityValidator {
    private let logger = Logger(subsystem: "com.morninggoal.app", category: "validation")

    struct ValidationResult {
        let accuracy: Double
        let precision: Double
        let recall: Double
        let f1Score: Double
        let confidenceDistribution: [Double]
        let inferenceTime: Double
        let memoryUsage: Double
    }

    struct QualityMetrics {
        let categoryAccuracy: Double
        let sentimentAccuracy: Double
        let avgConfidence: Double
        let lowConfidenceRate: Double
        let avgInferenceTime: Double
        let memoryEfficiency: Double
    }

    func validateModel(_ service: AnalysisService, with testData: [TestSample]) async -> ValidationResult {
        logger.log("starting_model_validation samples=\(testData.count)")

        var predictions: [(predicted: AnalysisResult, actual: TestSample)] = []
        var inferenceTimes: [Double] = []
        var confidences: [Double] = []

        for (index, sample) in testData.enumerated() {
            let t0 = CFAbsoluteTimeGetCurrent()
            let prediction = try? await service.analyzeGoal(sample.text)
            let t1 = CFAbsoluteTimeGetCurrent()

            if let pred = prediction {
                predictions.append((pred, sample))
                confidences.append(pred.categoryConfidence)
                inferenceTimes.append((t1 - t0) * 1000)
            }

            if index % 5 == 0 {
                logger.log("validation_progress=\(index)/\(testData.count)")
            }
        }

        let accuracy = calculateAccuracy(predictions)
        let precision = calculatePrecision(predictions)
        let recall = calculateRecall(predictions)
        let f1Score = 2 * (precision * recall) / (precision + recall)

        let avgInferenceTime = inferenceTimes.isEmpty ? 0 : inferenceTimes.reduce(0, +) / Double(inferenceTimes.count)
        let memoryUsage = memoryMB()

        logger.log("validation_complete accuracy=\(String(format: "%.3f", accuracy)) f1=\(String(format: "%.3f", f1Score))")

        return ValidationResult(
            accuracy: accuracy,
            precision: precision,
            recall: recall,
            f1Score: f1Score,
            confidenceDistribution: confidences,
            inferenceTime: avgInferenceTime,
            memoryUsage: memoryUsage
        )
    }

    func assessQuality(_ metrics: QualityMetrics) -> String {
        var issues: [String] = []
        var score = 100.0

        if metrics.categoryAccuracy < 0.8 {
            issues.append("低分类准确率 (\(String(format: "%.1f", metrics.categoryAccuracy * 100))%)")
            score -= 20
        }

        if metrics.sentimentAccuracy < 0.8 {
            issues.append("低情感分析准确率 (\(String(format: "%.1f", metrics.sentimentAccuracy * 100))%)")
            score -= 15
        }

        if metrics.avgConfidence < 0.7 {
            issues.append("低平均置信度 (\(String(format: "%.1f", metrics.avgConfidence * 100))%)")
            score -= 10
        }

        if metrics.lowConfidenceRate > 0.3 {
            issues.append("高不确定性率 (\(String(format: "%.1f", metrics.lowConfidenceRate * 100))%)")
            score -= 15
        }

        if metrics.avgInferenceTime > 1000 {
            issues.append("推理时间过长 (\(String(format: "%.0f", metrics.avgInferenceTime))ms)")
            score -= 10
        }

        let assessment: String
        if score >= 90 {
            assessment = "优秀"
        } else if score >= 80 {
            assessment = "良好"
        } else if score >= 70 {
            assessment = "一般"
        } else {
            assessment = "需改进"
        }

        logger.log("quality_assessment score=\(String(format: "%.1f", score)) grade=\(assessment) issues=\(issues.count)")

        return "\(assessment) (得分: \(String(format: "%.1f", score)))"
    }

    private func calculateAccuracy(_ predictions: [(predicted: AnalysisResult, actual: TestSample)]) -> Double {
        guard !predictions.isEmpty else { return 0 }

        let correct = predictions.filter { prediction in
            prediction.predicted.category == prediction.actual.expectedCategory &&
                prediction.predicted.sentiment == prediction.actual.expectedSentiment
        }

        return Double(correct.count) / Double(predictions.count)
    }

    private func calculatePrecision(_ predictions: [(predicted: AnalysisResult, actual: TestSample)]) -> Double {
        guard !predictions.isEmpty else { return 0 }

        let categories = Set(predictions.map { $0.predicted.category })
        var precisions: [Double] = []

        for category in categories {
            let categoryPredictions = predictions.filter { $0.predicted.category == category }
            let truePositives = categoryPredictions.filter { $0.actual.expectedCategory == category }.count
            let falsePositives = categoryPredictions.filter { $0.actual.expectedCategory != category }.count

            let precision = Double(truePositives) / Double(max(truePositives + falsePositives, 1))
            precisions.append(precision)
        }

        return precisions.isEmpty ? 0 : precisions.reduce(0, +) / Double(precisions.count)
    }

    private func calculateRecall(_ predictions: [(predicted: AnalysisResult, actual: TestSample)]) -> Double {
        guard !predictions.isEmpty else { return 0 }

        let categories = Set(predictions.map { $0.actual.expectedCategory })
        var recalls: [Double] = []

        for category in categories {
            let categoryActual = predictions.filter { $0.actual.expectedCategory == category }
            let truePositives = categoryActual.filter { $0.predicted.category == category }.count
            let falseNegatives = categoryActual.filter { $0.predicted.category != category }.count

            let recall = Double(truePositives) / Double(max(truePositives + falseNegatives, 1))
            recalls.append(recall)
        }

        return recalls.isEmpty ? 0 : recalls.reduce(0, +) / Double(recalls.count)
    }
}

private func getMemoryUsage() -> Double {
    var taskInfo = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4

    let kerr: kern_return_t = withUnsafeMutablePointer(to: &taskInfo) {
        $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
        }
    }

    if kerr == KERN_SUCCESS {
        return Double(taskInfo.resident_size) / 1024.0 / 1024.0
    } else {
        return 0.0
    }
}

struct TestSample {
    let text: String
    let expectedCategory: String
    let expectedSentiment: String
    let difficulty: Difficulty = .medium

    enum Difficulty: String {
        case easy = "简单"
        case medium = "中等"
        case hard = "困难"
    }
}
