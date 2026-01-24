import CoreData
import SwiftUI

/// 模型质量测试视图
/// 提供交互式界面测试 student_sequence_classification.mlpackage 的性能和准确率
struct ModelQualityTestView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var testResult: BenchSummary?
    @State private var isTesting = false
    @State private var testLog: [String] = []

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 24) {
                    // 标题区域
                    headerSection

                    // 测试控制
                    controlSection

                    // 测试结果
                    if let result = testResult {
                        resultSection(result)
                    }

                    // 测试日志
                    if !testLog.isEmpty {
                        logSection
                    }
                }
                .padding()
            }
            .navigationTitle("模型质量测试")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("关闭") {
                        dismiss()
                    }
                }
            }
        }
    }

    // MARK: - UI Components

    private var headerSection: some View {
        VStack(spacing: 8) {
            Image(systemName: "cpu.fill")
                .font(.system(size: 48))
                .foregroundColor(.blue)

            Text("MobileBERT 质量测试")
                .font(.title2)
                .fontWeight(.bold)

            Text("测试模型在 24 个标准样本上的表现")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding()
    }

    private var controlSection: some View {
        VStack(spacing: 16) {
            if isTesting {
                ProgressView("正在测试模型...")
                    .progressViewStyle(.circular)
            } else {
                Button(action: runTest) {
                    Label("开始测试", systemImage: "play.circle.fill")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(12)
                }
            }

            if testResult != nil {
                Button(action: clearResults) {
                    Label("清除结果", systemImage: "trash")
                        .font(.subheadline)
                        .foregroundColor(.red)
                }
            }
        }
    }

    private func resultSection(_ result: BenchSummary) -> some View {
        VStack(spacing: 16) {
            Text("测试结果")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)

            // 性能指标
            MetricCard(
                title: "性能指标",
                icon: "speedometer",
                color: .blue,
                metrics: [
                    MetricItem(label: "平均推理时间", value: String(format: "%.2f", result.avg) + "ms", isGood: result.avg < 100),
                    MetricItem(label: "P90 延迟", value: String(format: "%.2f", result.p90) + "ms", isGood: result.p90 < 150),
                    MetricItem(label: "内存使用", value: String(format: "%.2f", result.memMB) + "MB", isGood: result.memMB < 100)
                ]
            )

            // 准确率指标
            MetricCard(
                title: "准确率指标",
                icon: "checkmark.circle",
                color: .green,
                metrics: [
                    MetricItem(label: "分类准确率", value: String(format: "%.1f", result.catAcc * 100) + "%", isGood: result.catAcc >= 0.8),
                    MetricItem(label: "情感准确率", value: String(format: "%.1f", result.senAcc * 100) + "%", isGood: result.senAcc >= 0.75)
                ]
            )

            // 综合评估
            assessmentCard(result)
        }
    }

    private func assessmentCard(_ result: BenchSummary) -> some View {
        let score = calculateScore(result)
        let grade = gradeForScore(score)

        return VStack(spacing: 12) {
            HStack {
                Image(systemName: "star.fill")
                    .foregroundColor(.yellow)
                Text("综合评估")
                    .font(.headline)
                Spacer()
            }

            HStack {
                Text(grade)
                    .font(.title)
                    .fontWeight(.bold)
                    .foregroundColor(colorForGrade(grade))

                Spacer()

                Text("得分: \(String(format: "%.0f", score))")
                    .font(.title3)
                    .foregroundColor(.secondary)
            }

            if score < 80 {
                Text("建议: 检查模型训练数据质量或考虑重新训练")
                    .font(.caption)
                    .foregroundColor(.orange)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding()
        .background(Color.gray.opacity(0.1))
        .cornerRadius(12)
    }

    private var logSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("测试日志")
                .font(.headline)

            VStack(alignment: .leading, spacing: 4) {
                ForEach(testLog, id: \.self) { log in
                    Text(log)
                        .font(.system(.caption, design: .monospaced))
                        .foregroundColor(.secondary)
                }
            }
            .padding()
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.black.opacity(0.05))
            .cornerRadius(8)
        }
    }

    // MARK: - Actions

    private func runTest() {
        isTesting = true
        testLog = []
        addLog("开始模型质量测试...")

        Task {
            let context = DataController.shared.container.viewContext
            addLog("加载测试数据...")

            let result = await ModelBench.run(in: context)

            await MainActor.run {
                testResult = result
                isTesting = false
                addLog("测试完成!")
                addLog("分类准确率: \(String(format: "%.1f%%", result.catAcc * 100))")
                addLog("情感准确率: \(String(format: "%.1f%%", result.senAcc * 100))")
                addLog("平均推理时间: \(String(format: "%.2fms", result.avg))")
            }
        }
    }

    private func clearResults() {
        testResult = nil
        testLog = []
    }

    private func addLog(_ message: String) {
        let timestamp = DateFormatter.localizedString(from: Date(), dateStyle: .none, timeStyle: .medium)
        testLog.append("[\(timestamp)] \(message)")
    }

    // MARK: - Helpers

    private func calculateScore(_ result: BenchSummary) -> Double {
        var score = 100.0

        if result.catAcc < 0.8 { score -= 20 }
        if result.senAcc < 0.75 { score -= 15 }
        if result.avg > 100 { score -= 10 }
        if result.p90 > 150 { score -= 10 }
        if result.memMB > 100 { score -= 5 }

        return max(score, 0)
    }

    private func gradeForScore(_ score: Double) -> String {
        if score >= 90 { return "优秀" }
        if score >= 80 { return "良好" }
        if score >= 70 { return "一般" }
        return "需改进"
    }

    private func colorForGrade(_ grade: String) -> Color {
        switch grade {
        case "优秀": return .green
        case "良好": return .blue
        case "一般": return .orange
        default: return .red
        }
    }
}

// MARK: - Supporting Views

struct MetricItem {
    let label: String
    let value: String
    let isGood: Bool
}

struct MetricCard: View {
    let title: String
    let icon: String
    let color: Color
    let metrics: [MetricItem]

    var body: some View {
        VStack(spacing: 12) {
            HStack {
                Image(systemName: icon)
                    .foregroundColor(color)
                Text(title)
                    .font(.headline)
                Spacer()
            }

            VStack(spacing: 8) {
                ForEach(metrics, id: \.label) { metric in
                    HStack {
                        Text(metric.label)
                            .foregroundColor(.secondary)
                        Spacer()
                        Text(metric.value)
                            .fontWeight(.semibold)
                        Image(systemName: metric.isGood ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                            .foregroundColor(metric.isGood ? .green : .orange)
                            .font(.caption)
                    }
                }
            }
        }
        .padding()
        .background(Color.gray.opacity(0.1))
        .cornerRadius(12)
    }
}

// MARK: - Preview

#Preview {
    ModelQualityTestView()
}
