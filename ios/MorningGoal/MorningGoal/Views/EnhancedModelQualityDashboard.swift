import Combine
import CoreData
import OSLog
import SwiftUI

struct EnhancedModelQualityDashboard: View {
    @Environment(\.managedObjectContext) private var context
    @StateObject private var viewModel = EnhancedModelQualityViewModel()

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    enhancedQualityOverviewCard
                    detailedMetricsCard
                    improvementStrategiesCard
                    advancedBenchmarkCard
                    featureAnalysisCard
                    optimizationControlsCard
                }
                .padding()
            }
            .navigationTitle("增强模型质量分析")
            .onAppear {
                viewModel.refreshData(context: context)
            }
        }
    }

    private var enhancedQualityOverviewCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                VStack(alignment: .leading) {
                    Text("综合质量评估")
                        .font(.headline)
                        .foregroundColor(.primary)

                    Text(viewModel.overallQuality)
                        .font(.largeTitle)
                        .fontWeight(.bold)
                        .foregroundColor(viewModel.qualityColor)
                }

                Spacer()

                CircularProgressView(
                    progress: viewModel.overallScore / 100.0,
                    color: viewModel.qualityColor,
                    size: 80
                )
            }

            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("改进潜力")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                    Spacer()
                    Text("\(String(format: "%.1f", viewModel.improvementPotential))%")
                        .font(.subheadline)
                        .fontWeight(.semibold)
                }

                HStack {
                    Text("置信度")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                    Spacer()
                    Text("\(String(format: "%.1f", viewModel.avgConfidence * 100))%")
                        .font(.subheadline)
                        .fontWeight(.semibold)
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
        .shadow(color: .black.opacity(0.1), radius: 4, x: 0, y: 2)
    }

    private var detailedMetricsCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("详细性能指标")
                .font(.headline)

            LazyVGrid(columns: [
                GridItem(.flexible()),
                GridItem(.flexible())
            ], spacing: 12) {
                MetricView(
                    title: "分类准确率",
                    value: "\(String(format: "%.1f", viewModel.categoryAccuracy * 100))%",
                    color: viewModel.categoryAccuracy > 0.8 ? .green : viewModel.categoryAccuracy > 0.6 ? .orange : .red,
                    icon: "tag"
                )

                MetricView(
                    title: "情感准确率",
                    value: "\(String(format: "%.1f", viewModel.sentimentAccuracy * 100))%",
                    color: viewModel.sentimentAccuracy > 0.8 ? .green : viewModel.sentimentAccuracy > 0.6 ? .orange : .red,
                    icon: "face.smiling"
                )

                MetricView(
                    title: "推理时间",
                    value: "\(String(format: "%.0f", viewModel.avgInferenceTime))ms",
                    color: viewModel.avgInferenceTime < 500 ? .green : viewModel.avgInferenceTime < 1000 ? .orange : .red,
                    icon: "clock"
                )

                MetricView(
                    title: "内存使用",
                    value: "\(String(format: "%.1f", viewModel.memoryUsage))MB",
                    color: viewModel.memoryUsage < 100 ? .green : viewModel.memoryUsage < 200 ? .orange : .red,
                    icon: "memorychip"
                )
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
        .shadow(color: .black.opacity(0.1), radius: 4, x: 0, y: 2)
    }

    private var improvementStrategiesCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Text("优化策略建议")
                    .font(.headline)

                Spacer()

                if viewModel.isGeneratingStrategies {
                    ProgressView()
                        .scaleEffect(0.8)
                } else {
                    Button("重新生成") {
                        viewModel.generateOptimizationStrategies()
                    }
                    .font(.subheadline)
                }
            }

            if viewModel.optimizationStrategies.isEmpty {
                VStack(spacing: 8) {
                    Image(systemName: "lightbulb")
                        .font(.largeTitle)
                        .foregroundColor(.orange)

                    Text("正在生成优化策略...")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                .frame(maxWidth: .infinity)
                .padding()
            } else {
                ForEach(viewModel.optimizationStrategies, id: \.name) { strategy in
                    StrategyRow(strategy: strategy)
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
        .shadow(color: .black.opacity(0.1), radius: 4, x: 0, y: 2)
    }

    private var advancedBenchmarkCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Text("高级基准测试")
                    .font(.headline)

                Spacer()

                Button(viewModel.isBenchmarking ? "测试中..." : "开始测试") {
                    viewModel.runEnhancedBenchmark()
                }
                .disabled(viewModel.isBenchmarking)
                .buttonStyle(.borderedProminent)
            }

            if viewModel.isBenchmarking {
                VStack(spacing: 12) {
                    ProgressView()
                        .scaleEffect(1.2)

                    Text("正在执行增强基准测试...")
                        .font(.subheadline)
                        .foregroundColor(.secondary)

                    if !viewModel.benchmarkProgress.isEmpty {
                        Text(viewModel.benchmarkProgress)
                            .font(.caption)
                            .foregroundColor(.blue)
                    }
                }
                .frame(maxWidth: .infinity)
                .padding()
            } else if !viewModel.benchmarkResults.isEmpty {
                ForEach(viewModel.benchmarkResults, id: \.testName) { result in
                    BenchmarkResultRow(result: result)
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
        .shadow(color: .black.opacity(0.1), radius: 4, x: 0, y: 2)
    }

    private var featureAnalysisCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("特征分析")
                .font(.headline)

            if !viewModel.featureAnalysis.isEmpty {
                ForEach(viewModel.featureAnalysis, id: \.featureName) { analysis in
                    FeatureAnalysisRow(analysis: analysis)
                }
            } else {
                Text("运行基准测试查看特征分析")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                    .frame(maxWidth: .infinity)
                    .padding()
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
        .shadow(color: .black.opacity(0.1), radius: 4, x: 0, y: 2)
    }

    private var optimizationControlsCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("优化控制")
                .font(.headline)

            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Text("模型策略")
                        .font(.subheadline)
                    Spacer()
                    Picker("策略", selection: $viewModel.selectedStrategy) {
                        Text("准确率优先").tag(ModelOptimizationStrategy.accuracy)
                        Text("平衡").tag(ModelOptimizationStrategy.balanced)
                        Text("速度优先").tag(ModelOptimizationStrategy.speed)
                    }
                    .pickerStyle(SegmentedPickerStyle())
                    .frame(width: 200)
                }

                HStack {
                    Text("置信度阈值")
                        .font(.subheadline)
                    Spacer()
                    Slider(value: $viewModel.confidenceThreshold, in: 0.5 ... 0.95, step: 0.05)
                        .frame(width: 150)
                    Text("\(String(format: "%.2f", viewModel.confidenceThreshold))")
                        .font(.caption)
                        .frame(width: 30)
                }

                HStack {
                    Text("温度参数")
                        .font(.subheadline)
                    Spacer()
                    Slider(value: $viewModel.temperature, in: 0.5 ... 1.5, step: 0.1)
                        .frame(width: 150)
                    Text("\(String(format: "%.1f", viewModel.temperature))")
                        .font(.caption)
                        .frame(width: 30)
                }
            }

            Button(action: {
                viewModel.applyOptimizationSettings()
            }, label: {
                HStack {
                    Image(systemName: "checkmark.circle")
                    Text("应用设置")
                }
                .frame(maxWidth: .infinity)
            })
            .buttonStyle(.borderedProminent)
            .disabled(viewModel.isApplyingSettings)
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(16)
        .shadow(color: .black.opacity(0.1), radius: 4, x: 0, y: 2)
    }
}

// MARK: - 子视图组件

struct CircularProgressView: View {
    let progress: Double
    let color: Color
    let size: CGFloat

    var body: some View {
        ZStack {
            Circle()
                .stroke(lineWidth: 8)
                .opacity(0.3)
                .foregroundColor(color)

            Circle()
                .trim(from: 0.0, to: CGFloat(min(progress, 1.0)))
                .stroke(style: StrokeStyle(lineWidth: 8, lineCap: .round, lineJoin: .round))
                .foregroundColor(color)
                .rotationEffect(Angle(degrees: 270.0))
                .animation(.linear, value: progress)

            Text("\(Int(progress * 100))")
                .font(.caption)
                .fontWeight(.bold)
                .foregroundColor(color)
        }
        .frame(width: size, height: size)
    }
}

struct MetricView: View {
    let title: String
    let value: String
    let color: Color
    let icon: String

    var body: some View {
        VStack(spacing: 8) {
            HStack {
                Image(systemName: icon)
                    .foregroundColor(color)
                    .font(.title3)
                Spacer()
            }

            Text(value)
                .font(.title2)
                .fontWeight(.bold)
                .foregroundColor(color)

            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(color.opacity(0.1))
        .cornerRadius(12)
    }
}

struct StrategyRow: View {
    let strategy: OptimizationStrategy

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(strategy.name)
                    .font(.subheadline)
                    .fontWeight(.semibold)

                Spacer()

                Text("+\(String(format: "%.1f", strategy.expectedImprovement))%")
                    .font(.caption)
                    .fontWeight(.bold)
                    .foregroundColor(.green)
            }

            Text(strategy.description)
                .font(.caption)
                .foregroundColor(.secondary)

            HStack {
                Text("优先级: \(strategy.priority)")
                    .font(.caption2)
                    .foregroundColor(.blue)

                Spacer()

                if strategy.isRecommended {
                    Label("推荐", systemImage: "star.fill")
                        .font(.caption2)
                        .foregroundColor(.orange)
                }
            }
        }
        .padding()
        .background(Color(.secondarySystemBackground))
        .cornerRadius(10)
    }
}

struct BenchmarkResultRow: View {
    let result: BenchmarkResult

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(result.testName)
                    .font(.subheadline)
                    .fontWeight(.semibold)

                Spacer()

                Text("\(String(format: "%.1f", result.accuracy * 100))%")
                    .font(.caption)
                    .fontWeight(.bold)
                    .foregroundColor(result.accuracy > 0.8 ? .green : result.accuracy > 0.6 ? .orange : .red)
            }

            HStack {
                Label("\(String(format: "%.0f", result.inferenceTime))ms", systemImage: "clock")
                    .font(.caption)
                    .foregroundColor(.secondary)

                Spacer()

                Label("\(String(format: "%.1f", result.confidence * 100))%", systemImage: "checkmark.shield")
                    .font(.caption)
                    .foregroundColor(.blue)
            }
        }
        .padding()
        .background(Color(.secondarySystemBackground))
        .cornerRadius(10)
    }
}

struct FeatureAnalysisRow: View {
    let analysis: FeatureAnalysis

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(analysis.featureName)
                    .font(.subheadline)
                    .fontWeight(.semibold)

                Spacer()

                Text("\(String(format: "%.3f", analysis.importance))")
                    .font(.caption)
                    .fontWeight(.bold)
                    .foregroundColor(.purple)
            }

            ProgressView(value: analysis.importance)
                .progressViewStyle(LinearProgressViewStyle())
                .tint(analysis.importance > 0.7 ? .green : analysis.importance > 0.4 ? .orange : .red)

            Text(analysis.description)
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(Color(.secondarySystemBackground))
        .cornerRadius(10)
    }
}

struct EnhancedModelQualityDashboard_Previews: PreviewProvider {
    static var previews: some View {
        EnhancedModelQualityDashboard()
    }
}
