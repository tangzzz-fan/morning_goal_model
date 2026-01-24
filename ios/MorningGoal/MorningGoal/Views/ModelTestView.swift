import CoreData
import Darwin
import OSLog
import SwiftUI

struct TestMetrics {
    var avg: Double
    var p90: Double
    var catAcc: Double
    var senAcc: Double
    var memMB: Double
}

struct ModelTestView: View {
    @Environment(\.managedObjectContext) private var viewContext
    @FetchRequest(entity: GoalEntry.entity(), sortDescriptors: []) private var entries: FetchedResults<GoalEntry>
    @State private var selectedIndex: Int?
    @State private var metrics: TestMetrics?
    @State private var lastResult: AnalysisResult?
    @State private var isRunning: Bool = false
    private let logger = Logger(subsystem: "com.morninggoal.app", category: "ml-bench")
    var body: some View {
        VStack {
            HStack {
                Picker("记录", selection: Binding(get: { selectedIndex ?? 0 }, set: { selectedIndex = $0 })) {
                    ForEach(Array(entries.enumerated()), id: \.offset) { pair in
                        Text(pair.element.goalText).lineLimit(1).tag(pair.offset)
                    }
                }
                Button("单条测试") { runSingle() }.disabled(isRunning || entries.isEmpty)
                Button("批量测试") { runBatch() }.disabled(isRunning || entries.isEmpty)
            }
            .padding()
            if let result = lastResult {
                VStack(alignment: .leading) {
                    Text("类别: \(result.category)  置信度: \(String(format: "%.2f", result.categoryConfidence))")
                    Text("情感: \(result.sentiment)  置信度: \(String(format: "%.2f", result.sentimentScore))")
                }.padding()
            }
            if let summary = metrics {
                VStack(alignment: .leading) {
                    Text("平均耗时(ms): \(String(format: "%.2f", summary.avg))")
                    Text("90分位(ms): \(String(format: "%.2f", summary.p90))")
                    Text("分类准确率: \(String(format: "%.2f", summary.catAcc))")
                    Text("情感准确率: \(String(format: "%.2f", summary.senAcc))")
                    Text("内存(MB): \(String(format: "%.2f", summary.memMB))")
                }.padding()
            }
            List { ForEach(entries) { entry in Text(entry.goalText) } }
        }.navigationTitle("模型测试")
    }

    private func ruleExpected(_ text: String) async -> AnalysisResult {
        do {
            return try await RuleBasedAnalysisService().analyzeGoal(text)
        } catch {
            return AnalysisResult(
                category: "个人发展",
                categoryConfidence: 0.5,
                sentiment: "中性",
                sentimentScore: 0.0
            )
        }
    }

    private func coremlService() -> AnalysisService {
        (try? CoreMLGoalAnalysisService()) ?? RuleBasedAnalysisService()
    }

    private func runSingle() {
        guard let idx = selectedIndex, entries.indices.contains(idx) else { return }
        Task {
            isRunning = true
            defer { isRunning = false }
            let entry = entries[idx]
            let service = coremlService()
            let result = try? await service.analyzeGoal(entry.goalText)
            lastResult = result
        }
    }

    private func runBatch() {
        Task {
            isRunning = true
            defer { isRunning = false }
            let service = coremlService()
            var times: [Double] = []
            var correctCat = 0
            var correctSen = 0
            for entry in entries {
                let start = CFAbsoluteTimeGetCurrent()
                let result = try? await service.analyzeGoal(entry.goalText)
                let end = CFAbsoluteTimeGetCurrent()
                times.append((end - start) * 1000)
                if let result = result {
                    let expected = await ruleExpected(entry.goalText)
                    if result.category == expected.category { correctCat += 1 }
                    if result.sentiment == expected.sentiment { correctSen += 1 }
                    lastResult = result
                }
            }
            times.sort()
            let avg = times.reduce(0, +) / Double(times.count)
            let p90 = times[Int(Double(times.count) * 0.9)]
            let memory = memoryMB()
            let summary = TestMetrics(
                avg: avg,
                p90: p90,
                catAcc: Double(correctCat) / Double(max(entries.count, 1)),
                senAcc: Double(correctSen) / Double(max(entries.count, 1)),
                memMB: memory
            )
            metrics = summary
            logger.log("bench avg=\(String(format: "%.2f", avg)) p90=\(String(format: "%.2f", p90)) memMB=\(String(format: "%.2f", memory))")
        }
    }
}

func memoryMB() -> Double {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
    let kerr = withUnsafeMutablePointer(to: &info) { ptr -> kern_return_t in
        ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
        }
    }
    if kerr == KERN_SUCCESS { return Double(info.resident_size) / 1024.0 / 1024.0 } else { return 0 }
}
