import Combine
import CoreData
import Foundation
import OSLog
import SwiftUI

class EnhancedModelQualityViewModel: ObservableObject {
    @Published var overallQuality: String = "分析中..."
    @Published var overallScore: Double = 0.0
    @Published var categoryAccuracy: Double = 0.0
    @Published var sentimentAccuracy: Double = 0.0
    @Published var avgConfidence: Double = 0.0
    @Published var avgInferenceTime: Double = 0.0
    @Published var memoryUsage: Double = 0.0
    @Published var lowConfidenceRate: Double = 0.0
    @Published var improvementPotential: Double = 0.0

    @Published var optimizationStrategies: [OptimizationStrategy] = []
    @Published var benchmarkResults: [BenchmarkResult] = []
    @Published var featureAnalysis: [FeatureAnalysis] = []

    @Published var isBenchmarking: Bool = false
    @Published var isGeneratingStrategies: Bool = false
    @Published var isApplyingSettings: Bool = false
    @Published var benchmarkProgress: String = ""

    @Published var selectedStrategy: ModelOptimizationStrategy = .balanced
    @Published var confidenceThreshold: Double = 0.75
    @Published var temperature: Double = 0.8

    private let logger = Logger(subsystem: "com.morninggoal.app", category: "enhanced-dashboard")
    private let analyzer = ModelQualityAnalyzer()
    private let optimizationEngine = EnhancedModelOptimizationEngine()

    private var context: NSManagedObjectContext?

    private struct TestSample {
        let text: String
        let expectedCategory: String
        let expectedSentiment: String
    }

    var qualityColor: Color {
        switch overallScore {
        case 90 ... 100: return .green
        case 80 ..< 90: return .blue
        case 70 ..< 80: return .orange
        default: return .red
        }
    }

    func refreshData(context: NSManagedObjectContext) {
        logger.log("refreshing_enhanced_dashboard_data")
        self.context = context

        Task {
            await runEnhancedBenchmark()
        }
    }

    func runEnhancedBenchmark() {
        logger.log("starting_enhanced_benchmark")
        isBenchmarking = true
        benchmarkProgress = "正在初始化模型..."

        Task {
            do {
                // 使用增强的模型进行测试
                let enhancedService = try EnhancedAdaptiveModelService()

                await MainActor.run {
                    self.benchmarkProgress = "正在运行质量评估..."
                }

                // 运行增强的基准测试
                let enhancedResults = await runEnhancedModelBenchmark(service: enhancedService)

                await MainActor.run {
                    self.benchmarkResults = enhancedResults
                    self.updateMetrics(from: enhancedResults)
                    self.benchmarkProgress = "正在分析特征..."
                }

                // 生成特征分析
                let featureAnalysis = await generateFeatureAnalysis(results: enhancedResults)

                await MainActor.run {
                    self.featureAnalysis = featureAnalysis
                    self.benchmarkProgress = "正在生成优化策略..."
                }

                // 生成优化策略
                generateOptimizationStrategies()

                await MainActor.run {
                    self.isBenchmarking = false
                    self.benchmarkProgress = ""
                    self.logger.log("enhanced_benchmark_completed score=\(self.overallScore)")
                }

            } catch {
                await MainActor.run {
                    self.isBenchmarking = false
                    self.overallQuality = "评估失败"
                    self.logger.error("enhanced_benchmark_failed error=\(error.localizedDescription)")
                }
            }
        }
    }

    private func runEnhancedModelBenchmark(service: EnhancedAdaptiveModelService) async -> [BenchmarkResult] {
        let testSamples = createEnhancedTestSamples()
        var results: [BenchmarkResult] = []

        for (index, sample) in testSamples.enumerated() {
            await MainActor.run {
                self.benchmarkProgress = "测试样本 \(index + 1)/\(testSamples.count)..."
            }

            let startTime = CFAbsoluteTimeGetCurrent()
            let result = try? await service.analyzeGoal(sample.text)
            let endTime = CFAbsoluteTimeGetCurrent()

            let inferenceTime = (endTime - startTime) * 1000

            if let result = result {
                let isCorrect = result.category == sample.expectedCategory && result.sentiment == sample.expectedSentiment

                results.append(BenchmarkResult(
                    testName: "样本\(index + 1): \(String(sample.text.prefix(10)))...",
                    accuracy: isCorrect ? 1.0 : 0.0,
                    confidence: (result.categoryConfidence + result.sentimentScore) / 2,
                    inferenceTime: inferenceTime,
                    category: result.category,
                    sentiment: result.sentiment,
                    expectedCategory: sample.expectedCategory,
                    expectedSentiment: sample.expectedSentiment
                ))

                logger
                    .log(
                        "sample_\(index): text='\(sample.text)' predicted=(\(result.category),\(result.sentiment)) expected=(\(sample.expectedCategory),\(sample.expectedSentiment)) correct=\(isCorrect)"
                    )
            }
        }

        return results
    }

    private func updateMetrics(from results: [BenchmarkResult]) {
        guard !results.isEmpty else { return }

        let correctResults = results.filter { $0.accuracy > 0.5 }
        let overallAccuracy = Double(correctResults.count) / Double(results.count)

        let categoryCorrect = results.filter { $0.category == $0.expectedCategory }.count
        let sentimentCorrect = results.filter { $0.sentiment == $0.expectedSentiment }.count

        categoryAccuracy = Double(categoryCorrect) / Double(results.count)
        sentimentAccuracy = Double(sentimentCorrect) / Double(results.count)
        avgConfidence = results.map { $0.confidence }.reduce(0, +) / Double(results.count)
        avgInferenceTime = results.map { $0.inferenceTime }.reduce(0, +) / Double(results.count)
        lowConfidenceRate = Double(results.filter { $0.confidence < 0.7 }.count) / Double(results.count)
        memoryUsage = getMemoryUsage()

        // 计算综合得分
        let metrics = QualityMetrics(
            categoryAccuracy: categoryAccuracy,
            sentimentAccuracy: sentimentAccuracy,
            avgConfidence: avgConfidence,
            lowConfidenceRate: lowConfidenceRate,
            avgInferenceTime: avgInferenceTime,
            memoryEfficiency: 1.0 - (memoryUsage / 200.0) // 假设200MB为满分
        )

        overallScore = calculateOverallScore(metrics: metrics)
        let analysis = analyzer.analyzeQualityScore(overallScore, metrics: metrics)
        overallQuality = analysis.priorityActions.first ?? "评估完成"

        // 分析改进潜力
        improvementPotential = analysis.improvementPotential
    }

    private func generateFeatureAnalysis(results: [BenchmarkResult]) async -> [FeatureAnalysis] {
        var analysis: [FeatureAnalysis] = []

        // 分析不同特征的重要性
        let categoryDistribution = Dictionary(grouping: results, by: { $0.expectedCategory })
        let sentimentDistribution = Dictionary(grouping: results, by: { $0.expectedSentiment })

        // 类别分析
        for (category, categoryResults) in categoryDistribution {
            let accuracy = Double(categoryResults.filter { $0.category == $0.expectedCategory }.count) / Double(categoryResults.count)
            analysis.append(FeatureAnalysis(
                featureName: "\(category)分类",
                importance: accuracy,
                description: "\(category)类别的识别准确率为\(String(format: "%.1f", accuracy * 100))%"
            ))
        }

        // 情感分析
        for (sentiment, sentimentResults) in sentimentDistribution {
            let accuracy = Double(sentimentResults.filter { $0.sentiment == $0.expectedSentiment }.count) / Double(sentimentResults.count)
            analysis.append(FeatureAnalysis(
                featureName: "\(sentiment)情感",
                importance: accuracy,
                description: "\(sentiment)情感的识别准确率为\(String(format: "%.1f", accuracy * 100))%"
            ))
        }

        // 性能分析
        let avgTime = results.map { $0.inferenceTime }.reduce(0, +) / Double(results.count)
        let timeImportance = max(0, 1.0 - (avgTime / 1000.0)) // 1000ms为基准
        analysis.append(FeatureAnalysis(
            featureName: "推理速度",
            importance: timeImportance,
            description: "平均推理时间\(String(format: "%.0f", avgTime))ms"
        ))

        // 置信度分析
        let avgConf = results.map { $0.confidence }.reduce(0, +) / Double(results.count)
        analysis.append(FeatureAnalysis(
            featureName: "置信度",
            importance: avgConf,
            description: "平均置信度\(String(format: "%.1f", avgConf * 100))%"
        ))

        return analysis.sorted { $0.importance > $1.importance }
    }

    func generateOptimizationStrategies() {
        logger.log("generating_optimization_strategies")
        isGeneratingStrategies = true

        Task {
            let metrics = QualityMetrics(
                categoryAccuracy: categoryAccuracy,
                sentimentAccuracy: sentimentAccuracy,
                avgConfidence: avgConfidence,
                lowConfidenceRate: lowConfidenceRate,
                avgInferenceTime: avgInferenceTime,
                memoryEfficiency: 1.0 - (memoryUsage / 200.0)
            )

            let strategies = await optimizationEngine.generateOptimizationStrategies(
                for: overallScore,
                metrics: metrics
            )

            await MainActor.run {
                self.optimizationStrategies = strategies.map { strategy in
                    OptimizationStrategy(
                        name: strategy.name,
                        description: strategy.description,
                        expectedImprovement: strategy.expectedImprovement,
                        priority: "高",
                        isRecommended: true
                    )
                }
                self.isGeneratingStrategies = false
            }
        }
    }

    func applyOptimizationSettings() {
        logger
            .log(
                "applying_optimization_settings strategy=\(String(describing: self.selectedStrategy)) threshold=\(self.confidenceThreshold) temp=\(self.temperature)"
            )
        isApplyingSettings = true

        // 这里可以实现应用设置的逻辑
        Task {
            // 模拟设置应用过程
            try? await Task.sleep(nanoseconds: 1_000_000_000)

            await MainActor.run {
                self.isApplyingSettings = false
                self.logger.log("optimization_settings_applied")

                // 重新运行基准测试
                self.runEnhancedBenchmark()
            }
        }
    }

    private func calculateOverallScore(metrics: QualityMetrics) -> Double {
        let categoryWeight = 0.3
        let sentimentWeight = 0.3
        let confidenceWeight = 0.2
        let performanceWeight = 0.15
        let memoryWeight = 0.05

        let score = (metrics.categoryAccuracy * categoryWeight +
            metrics.sentimentAccuracy * sentimentWeight +
            metrics.avgConfidence * confidenceWeight +
            (1.0 - min(metrics.avgInferenceTime / 1000.0, 1.0)) * performanceWeight +
            metrics.memoryEfficiency * memoryWeight) * 100.0

        return min(max(score, 0.0), 100.0)
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

    private func createEnhancedTestSamples() -> [TestSample] {
        return [
            TestSample(text: "今天要完成项目开发工作，任务很紧急，压力很大", expectedCategory: "工作", expectedSentiment: "消极"),
            TestSample(text: "成功完成了重要项目，感觉很有成就感，得到领导认可", expectedCategory: "工作", expectedSentiment: "积极"),
            TestSample(text: "正常处理日常工作事务，按部就班完成任务", expectedCategory: "工作", expectedSentiment: "中性"),

            TestSample(text: "晚上要去健身房锻炼身体，保持健康活力，感觉很棒", expectedCategory: "健康", expectedSentiment: "积极"),
            TestSample(text: "最近身体状况不太好，需要看医生治疗，有些担心", expectedCategory: "健康", expectedSentiment: "消极"),
            TestSample(text: "每天坚持跑步锻炼，身体状态保持得不错", expectedCategory: "健康", expectedSentiment: "中性"),

            TestSample(text: "周末要和家人一起出游，期待美好时光，家庭很幸福", expectedCategory: "家庭", expectedSentiment: "积极"),
            TestSample(text: "孩子最近学习状态不好，让人担心焦虑，需要关注", expectedCategory: "家庭", expectedSentiment: "消极"),
            TestSample(text: "陪父母吃饭聊天，家庭时光很温馨平淡", expectedCategory: "家庭", expectedSentiment: "中性"),

            TestSample(text: "需要学习新的编程技能，提升专业能力，迎接挑战", expectedCategory: "学习", expectedSentiment: "中性"),
            TestSample(text: "学习进度很慢，感觉有些沮丧，怀疑自己能力", expectedCategory: "学习", expectedSentiment: "消极"),
            TestSample(text: "通过努力学习获得了认证，非常开心，付出有回报", expectedCategory: "学习", expectedSentiment: "积极"),

            TestSample(text: "这个月要控制消费支出，做好预算管理，理性消费", expectedCategory: "财务", expectedSentiment: "中性"),
            TestSample(text: "投资亏损了很多钱，心情很糟糕，财务压力很大", expectedCategory: "财务", expectedSentiment: "消极"),
            TestSample(text: "理财收益不错，财务状况改善很多，感到满意", expectedCategory: "财务", expectedSentiment: "积极"),

            TestSample(text: "和朋友聚餐很开心，友谊很珍贵，感受到温暖", expectedCategory: "社交", expectedSentiment: "积极"),
            TestSample(text: "社交活动让我感到很累，想独处，人际关系复杂", expectedCategory: "社交", expectedSentiment: "消极"),
            TestSample(text: "参加同事聚会，交流工作心得，关系融洽", expectedCategory: "社交", expectedSentiment: "中性"),

            TestSample(text: "要看电影放松一下，享受休闲时光，生活很美好", expectedCategory: "休闲", expectedSentiment: "积极"),
            TestSample(text: "娱乐活动很无聊，浪费时间，感到空虚", expectedCategory: "休闲", expectedSentiment: "消极"),
            TestSample(text: "在家听音乐看书，平静地休息，内心宁静", expectedCategory: "休闲", expectedSentiment: "中性"),

            TestSample(text: "需要提升个人能力和技能，实现成长，追求进步", expectedCategory: "个人发展", expectedSentiment: "中性"),
            TestSample(text: "个人发展遇到瓶颈，感到很迷茫，失去方向", expectedCategory: "个人发展", expectedSentiment: "消极"),
            TestSample(text: "通过努力实现了目标，个人能力提升，感到自豪", expectedCategory: "个人发展", expectedSentiment: "积极")
        ]
    }
}

// MARK: - 数据模型

struct OptimizationStrategy {
    let name: String
    let description: String
    let expectedImprovement: Double
    let priority: String
    let isRecommended: Bool
}

struct BenchmarkResult {
    let testName: String
    let accuracy: Double
    let confidence: Double
    let inferenceTime: Double
    let category: String
    let sentiment: String
    let expectedCategory: String
    let expectedSentiment: String
}

struct FeatureAnalysis {
    let featureName: String
    let importance: Double
    let description: String
}
