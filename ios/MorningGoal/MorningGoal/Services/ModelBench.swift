import CoreData
import Foundation
import MachO
import OSLog

struct BenchSummary {
    let avg: Double
    let p90: Double
    let catAcc: Double
    let senAcc: Double
    let memMB: Double
}

enum ModelBench {
    static func run(in context: NSManagedObjectContext) async -> BenchSummary {
        let logger = Logger(subsystem: "com.morninggoal.app", category: "ml-bench")

        // 使用自适应模型服务
        let services: [AnalysisService] = [
            (try? AdaptiveModelService()) as AnalysisService?,
            (try? CoreMLGoalAnalysisService()) as AnalysisService?
        ].compactMap { $0 }

        guard let service = services.first else {
            logger.error("no_models_available")
            return BenchSummary(avg: 0, p90: 0, catAcc: 0, senAcc: 0, memMB: 0)
        }

        // 创建测试数据以确保有评估数据
        let testSamples = createTestSamples()
        logger.log("starting_benchmark_with_samples count=\(testSamples.count)")

        var times: [Double] = []
        var correctCat = 0
        var correctSen = 0
        var lowConfidenceCount = 0
        var totalTokens = 0

        for (index, sample) in testSamples.enumerated() {
            let t0 = CFAbsoluteTimeGetCurrent()
            let result = try? await service.analyzeGoal(sample.text)
            let t1 = CFAbsoluteTimeGetCurrent()

            let inferenceTime = (t1 - t0) * 1000
            times.append(inferenceTime)

            if let result = result {
                if result.categoryConfidence < 0.7 || result.sentimentScore < 0.7 {
                    lowConfidenceCount += 1
                }

                totalTokens += min(sample.text.count, 256)

                // 与期望结果比较
                if result.category == sample.expectedCategory { correctCat += 1 }
                if result.sentiment == sample.expectedSentiment { correctSen += 1 }

                logger
                    .log(
                        "sample_\(index): text='\(sample.text)' predicted=(\(result.category),\(result.sentiment)) expected=(\(sample.expectedCategory),\(sample.expectedSentiment))"
                    )
            }

            if index % 5 == 0 {
                logger.log("bench_progress=\(index)/\(testSamples.count)")
            }
        }

        times.sort()
        let avg = times.isEmpty ? 0 : times.reduce(0, +) / Double(times.count)
        let p90 = times.isEmpty ? 0 : times[Int(Double(times.count) * 0.9)]
        let p50 = times.isEmpty ? 0 : times[Int(Double(times.count) * 0.5)]
        let mem = getMemoryUsage()

        let catAcc = Double(correctCat) / Double(max(testSamples.count, 1))
        let senAcc = Double(correctSen) / Double(max(testSamples.count, 1))
        let lowConfRate = Double(lowConfidenceCount) / Double(max(testSamples.count, 1))

        let summary = BenchSummary(avg: avg, p90: p90, catAcc: catAcc, senAcc: senAcc, memMB: mem)

        logger.log("bench_complete avg=\(String(format: "%.2f", avg)) p50=\(String(format: "%.2f", p50)) p90=\(String(format: "%.2f", p90))")
        logger
            .log(
                "bench_accuracy cat_acc=\(String(format: "%.3f", catAcc)) sen_acc=\(String(format: "%.3f", senAcc)) low_conf=\(String(format: "%.3f", lowConfRate))"
            )
        logger.log("bench_resources mem_mb=\(String(format: "%.2f", mem)) total_tokens=\(totalTokens)")

        return summary
    }

    private static func createTestSamples() -> [TestSample] {
        let items: [TestSample] = [
            TestSample(text: "今天要完成项目开发工作，任务很紧急", expectedCategory: "工作", expectedSentiment: "中性"),
            TestSample(text: "工作压力很大，让我感到非常焦虑", expectedCategory: "工作", expectedSentiment: "消极"),
            TestSample(text: "成功完成了重要项目，感觉很有成就感", expectedCategory: "工作", expectedSentiment: "积极"),

            TestSample(text: "晚上要去健身房锻炼身体，保持健康", expectedCategory: "健康", expectedSentiment: "积极"),
            TestSample(text: "最近身体状况不太好，需要看医生", expectedCategory: "健康", expectedSentiment: "消极"),
            TestSample(text: "每天坚持跑步锻炼，身体状态不错", expectedCategory: "健康", expectedSentiment: "中性"),

            TestSample(text: "周末要和家人一起出游，期待美好时光", expectedCategory: "家庭", expectedSentiment: "积极"),
            TestSample(text: "孩子最近学习状态不好，让人担心", expectedCategory: "家庭", expectedSentiment: "消极"),
            TestSample(text: "陪父母吃饭聊天，家庭时光很温馨", expectedCategory: "家庭", expectedSentiment: "中性"),

            TestSample(text: "需要学习新的编程技能，提升专业能力", expectedCategory: "学习", expectedSentiment: "中性"),
            TestSample(text: "学习进度很慢，感觉有些沮丧", expectedCategory: "学习", expectedSentiment: "消极"),
            TestSample(text: "通过努力学习获得了认证，非常开心", expectedCategory: "学习", expectedSentiment: "积极"),

            TestSample(text: "这个月要控制消费支出，做好预算管理", expectedCategory: "财务", expectedSentiment: "中性"),
            TestSample(text: "投资亏损了很多钱，心情很糟糕", expectedCategory: "财务", expectedSentiment: "消极"),
            TestSample(text: "理财收益不错，财务状况改善很多", expectedCategory: "财务", expectedSentiment: "积极"),

            TestSample(text: "和朋友聚餐很开心，友谊很珍贵", expectedCategory: "社交", expectedSentiment: "积极"),
            TestSample(text: "社交活动让我感到很累，想独处", expectedCategory: "社交", expectedSentiment: "消极"),
            TestSample(text: "参加同事聚会，交流工作心得", expectedCategory: "社交", expectedSentiment: "中性"),

            TestSample(text: "要看电影放松一下，享受休闲时光", expectedCategory: "休闲", expectedSentiment: "积极"),
            TestSample(text: "娱乐活动很无聊，浪费时间", expectedCategory: "休闲", expectedSentiment: "消极"),
            TestSample(text: "在家听音乐看书，平静地休息", expectedCategory: "休闲", expectedSentiment: "中性"),

            TestSample(text: "需要提升个人能力和技能，实现成长", expectedCategory: "个人发展", expectedSentiment: "中性"),
            TestSample(text: "个人发展遇到瓶颈，感到很迷茫", expectedCategory: "个人发展", expectedSentiment: "消极"),
            TestSample(text: "通过努力实现了目标，个人能力提升", expectedCategory: "个人发展", expectedSentiment: "积极")
        ]
        return items
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
