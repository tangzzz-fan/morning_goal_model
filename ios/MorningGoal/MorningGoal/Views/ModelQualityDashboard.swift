import Combine
import CoreData
import OSLog
import SwiftUI

struct ModelQualityDashboard: View {
    @Environment(\.managedObjectContext) private var context
    @StateObject private var viewModel = ModelQualityViewModel()

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    qualityOverviewCard
                    performanceMetricsCard
                    accuracyBreakdownCard
                    optimizationControlsCard
                    recentPredictionsCard
                }
                .padding()
            }
            .navigationTitle("模型质量监控")
            .onAppear {
                viewModel.refreshData(context: context)
            }
        }
    }

    private var qualityOverviewCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("模型状态")
                    .font(.headline)
                Spacer()
                Text(viewModel.overallQuality)
                    .font(.subheadline)
                    .foregroundColor(viewModel.qualityColor)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(viewModel.qualityColor.opacity(0.2))
                    .cornerRadius(8)
            }

            HStack(spacing: 20) {
                VStack(alignment: .leading) {
                    Text("分类准确率")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text("\(String(format: "%.1f", viewModel.categoryAccuracy * 100))%")
                        .font(.title2)
                        .fontWeight(.bold)
                }

                VStack(alignment: .leading) {
                    Text("情感准确率")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text("\(String(format: "%.1f", viewModel.sentimentAccuracy * 100))%")
                        .font(.title2)
                        .fontWeight(.bold)
                }

                VStack(alignment: .leading) {
                    Text("平均置信度")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text("\(String(format: "%.1f", viewModel.avgConfidence * 100))%")
                        .font(.title2)
                        .fontWeight(.bold)
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 2)
    }

    private var performanceMetricsCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("性能指标")
                .font(.headline)

            HStack(spacing: 20) {
                VStack(alignment: .leading) {
                    Text("平均推理时间")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text("\(String(format: "%.0f", viewModel.avgInferenceTime))ms")
                        .font(.title3)
                        .fontWeight(.semibold)
                }

                VStack(alignment: .leading) {
                    Text("内存使用")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text("\(String(format: "%.1f", viewModel.memoryUsage))MB")
                        .font(.title3)
                        .fontWeight(.semibold)
                }

                VStack(alignment: .leading) {
                    Text("低置信度率")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text("\(String(format: "%.1f", viewModel.lowConfidenceRate * 100))%")
                        .font(.title3)
                        .fontWeight(.semibold)
                        .foregroundColor(viewModel.lowConfidenceRate > 0.3 ? .orange : .primary)
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 2)
    }

    private var accuracyBreakdownCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("分类准确率详情")
                .font(.headline)

            ForEach(viewModel.categoryBreakdown, id: \.category) { item in
                HStack {
                    Text(item.category)
                        .font(.subheadline)
                    Spacer()
                    Text("\(String(format: "%.1f", item.accuracy * 100))%")
                        .font(.subheadline)
                        .fontWeight(.semibold)
                    ProgressView(value: item.accuracy)
                        .frame(width: 60)
                        .tint(item.accuracy > 0.8 ? .green : item.accuracy > 0.6 ? .orange : .red)
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 2)
    }

    private var optimizationControlsCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("优化控制")
                .font(.headline)

            HStack {
                Text("优化策略")
                    .font(.subheadline)
                Spacer()
                Picker("策略", selection: $viewModel.optimizationStrategy) {
                    Text("准确率优先").tag(ModelOptimizationStrategy.accuracy)
                    Text("平衡").tag(ModelOptimizationStrategy.balanced)
                    Text("速度优先").tag(ModelOptimizationStrategy.speed)
                }
                .pickerStyle(SegmentedPickerStyle())
                .frame(width: 200)
            }

            Button(action: {
                viewModel.runBenchmark()
            }, label: {
                HStack {
                    Image(systemName: "arrow.clockwise")
                    Text("重新基准测试")
                }
                .frame(maxWidth: .infinity)
            })
            .buttonStyle(.borderedProminent)
            .disabled(viewModel.isBenchmarking)

            if viewModel.isBenchmarking {
                ProgressView()
                    .progressViewStyle(CircularProgressViewStyle())
                    .frame(maxWidth: .infinity)
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 2)
    }

    private var recentPredictionsCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("最近预测")
                    .font(.headline)
                Spacer()
                Button("清除") {
                    viewModel.clearRecentPredictions()
                }
                .font(.subheadline)
            }

            if viewModel.recentPredictions.isEmpty {
                Text("暂无预测记录")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding()
            } else {
                ForEach(viewModel.recentPredictions.prefix(5), id: \.text) { prediction in
                    VStack(alignment: .leading, spacing: 4) {
                        Text(prediction.text)
                            .font(.subheadline)
                            .lineLimit(2)
                        HStack {
                            Label("\(prediction.category)", systemImage: "tag")
                                .font(.caption)
                            Spacer()
                            Label("\(prediction.sentiment)", systemImage: "face.smiling")
                                .font(.caption)
                            Text("\(String(format: "%.1f", prediction.confidence * 100))%")
                                .font(.caption)
                                .fontWeight(.semibold)
                                .foregroundColor(prediction.confidence > 0.7 ? .green : .orange)
                        }
                    }
                    .padding(.vertical, 4)
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(12)
        .shadow(radius: 2)
    }
}

class ModelQualityViewModel: ObservableObject {
    @Published var overallQuality: String = "评估中..."
    @Published var categoryAccuracy: Double = 0
    @Published var sentimentAccuracy: Double = 0
    @Published var avgConfidence: Double = 0
    @Published var avgInferenceTime: Double = 0
    @Published var memoryUsage: Double = 0
    @Published var lowConfidenceRate: Double = 0
    @Published var categoryBreakdown: [CategoryAccuracy] = []
    @Published var recentPredictions: [PredictionRecord] = []
    @Published var isBenchmarking: Bool = false
    @Published var optimizationStrategy: ModelOptimizationStrategy = .balanced

    private let logger = Logger(subsystem: "com.morninggoal.app", category: "dashboard")
    private let validator = ModelQualityValidator()
    private var context: NSManagedObjectContext?

    var qualityColor: Color {
        switch overallQuality {
        case "优秀": return .green
        case "良好": return .blue
        case "一般": return .orange
        default: return .red
        }
    }

    func refreshData(context: NSManagedObjectContext) {
        logger.log("refreshing_dashboard_data")
        self.context = context

        Task {
            await runBenchmark()
        }
    }

    func runBenchmark() {
        logger.log("starting_benchmark")
        isBenchmarking = true

        guard let context = context else {
            logger.error("no_context_available")
            isBenchmarking = false
            return
        }

        Task {
            do {
                let benchSummary = await ModelBench.run(in: context)

                await MainActor.run {
                    self.avgInferenceTime = benchSummary.avg
                    self.categoryAccuracy = benchSummary.catAcc
                    self.sentimentAccuracy = benchSummary.senAcc
                    self.memoryUsage = benchSummary.memMB

                    let metrics = ModelQualityValidator.QualityMetrics(
                        categoryAccuracy: benchSummary.catAcc,
                        sentimentAccuracy: benchSummary.senAcc,
                        avgConfidence: (benchSummary.catAcc + benchSummary.senAcc) / 2,
                        lowConfidenceRate: 0.1,
                        avgInferenceTime: benchSummary.avg,
                        memoryEfficiency: 1.0
                    )

                    self.overallQuality = validator.assessQuality(metrics)
                    self.isBenchmarking = false

                    logger.log("benchmark_complete quality=\(self.overallQuality)")
                }
            }
        }
    }

    func clearRecentPredictions() {
        recentPredictions.removeAll()
        logger.log("cleared_recent_predictions")
    }

    func addPrediction(_ text: String, category: String, sentiment: String, confidence: Double) {
        let record = PredictionRecord(
            text: text,
            category: category,
            sentiment: sentiment,
            confidence: confidence,
            timestamp: Date()
        )

        recentPredictions.insert(record, at: 0)

        if recentPredictions.count > 20 {
            recentPredictions.removeLast()
        }
    }
}

struct CategoryAccuracy {
    let category: String
    let accuracy: Double
}

struct PredictionRecord: Identifiable {
    let id = UUID()
    let text: String
    let category: String
    let sentiment: String
    let confidence: Double
    let timestamp: Date
}

struct ModelQualityDashboard_Previews: PreviewProvider {
    static var previews: some View {
        ModelQualityDashboard()
    }
}
