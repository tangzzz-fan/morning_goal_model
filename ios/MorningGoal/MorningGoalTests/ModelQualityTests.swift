import XCTest
import CoreData
@testable import MorningGoal

/// 模型质量单元测试
/// 可以通过 Xcode Test Navigator 或命令行 `xcodebuild test` 运行
final class ModelQualityTests: XCTestCase {
    
    var testContext: NSManagedObjectContext?
    
    override func setUpWithError() throws {
        try super.setUpWithError()
        // 使用内存数据库进行测试
        let container = InMemoryCoreData.container()
        testContext = container.viewContext
    }
    
    override func tearDownWithError() throws {
        testContext = nil
        try super.tearDownWithError()
    }
    
    // MARK: - 模型加载测试
    
    @MainActor
    func testModelLoading() throws {
        // 测试模型是否能成功加载
        let service = try AdaptiveModelService()
        XCTAssertNotNil(service, "模型服务应该成功初始化")
    }
    
    @MainActor
    func testModelVersion() throws {
        let service = try AdaptiveModelService()
        let version = service.getModelVersion()
        XCTAssertFalse(version.isEmpty, "模型版本不应为空")
        print("📦 模型版本: \(version)")
    }
    
    // MARK: - 推理性能测试
    
    @MainActor
    func testInferencePerformance() async throws {
        let service = try AdaptiveModelService()
        let testText = "今天要完成项目开发工作"
        
        // 测量推理时间
        measure {
            let expectation = XCTestExpectation(description: "推理完成")
            
            Task {
                _ = try? await service.analyzeGoal(testText)
                expectation.fulfill()
            }
            
            wait(for: [expectation], timeout: 5.0)
        }
    }
    
    @MainActor
    func testInferenceSpeed() async throws {
        let service = try AdaptiveModelService()
        let testText = "晚上去健身房锻炼身体"
        
        let startTime = CFAbsoluteTimeGetCurrent()
        _ = try await service.analyzeGoal(testText)
        let endTime = CFAbsoluteTimeGetCurrent()
        
        let inferenceTime = (endTime - startTime) * 1000 // ms
        
        print("⏱️ 推理时间: \(String(format: "%.2f", inferenceTime))ms")
        XCTAssertLessThan(inferenceTime, 200, "推理时间应小于 200ms")
    }
    
    // MARK: - 分类准确率测试
    
    @MainActor
    func testCategoryClassification() async throws {
        let service = try AdaptiveModelService()
        
        let testCases: [(text: String, expectedCategory: String)] = [
            ("今天要完成项目开发工作", "工作"),
            ("晚上去健身房锻炼身体", "健康"),
            ("周末和家人一起出游", "家庭"),
            ("需要学习新的编程技能", "学习"),
            ("这个月要控制消费支出", "财务"),
            ("和朋友聚餐很开心", "社交"),
            ("看电影放松一下", "休闲"),
            ("需要提升个人能力", "个人发展")
        ]
        
        var correctCount = 0
        
        for testCase in testCases {
            let result = try await service.analyzeGoal(testCase.text)
            if result.category == testCase.expectedCategory {
                correctCount += 1
            }
            print("📝 '\(testCase.text)' -> 预测:\(result.category) (期望:\(testCase.expectedCategory)) ✓")
        }
        
        let accuracy = Double(correctCount) / Double(testCases.count)
        print("🎯 分类准确率: \(String(format: "%.1f%%", accuracy * 100))")
        
        XCTAssertGreaterThanOrEqual(accuracy, 0.75, "分类准确率应至少达到 75%")
    }
    
    // MARK: - 情感分析测试
    
    @MainActor
    func testSentimentAnalysis() async throws {
        let service = try AdaptiveModelService()
        
        let testCases: [(text: String, expectedSentiment: String)] = [
            ("成功完成了重要项目，感觉很有成就感", "积极"),
            ("工作压力很大，让我感到非常焦虑", "消极"),
            ("今天要完成项目开发工作", "中性")
        ]
        
        var correctCount = 0
        
        for testCase in testCases {
            let result = try await service.analyzeGoal(testCase.text)
            if result.sentiment == testCase.expectedSentiment {
                correctCount += 1
            }
            print("💭 '\(testCase.text)' -> 预测:\(result.sentiment) (期望:\(testCase.expectedSentiment))")
        }
        
        let accuracy = Double(correctCount) / Double(testCases.count)
        print("🎯 情感准确率: \(String(format: "%.1f%%", accuracy * 100))")
        
        XCTAssertGreaterThanOrEqual(accuracy, 0.6, "情感准确率应至少达到 60%")
    }
    
    // MARK: - 置信度测试
    
    @MainActor
    func testConfidenceScores() async throws {
        let service = try AdaptiveModelService()
        let testText = "今天要完成项目开发工作"
        
        let result = try await service.analyzeGoal(testText)
        
        print("🔍 分类置信度: \(String(format: "%.3f", result.categoryConfidence))")
        print("🔍 情感置信度: \(String(format: "%.3f", result.sentimentScore))")
        
        XCTAssertGreaterThan(result.categoryConfidence, 0, "分类置信度应大于 0")
        XCTAssertLessThanOrEqual(result.categoryConfidence, 1, "分类置信度应小于等于 1")
    }
    
    // MARK: - 边界情况测试
    
    @MainActor
    func testEmptyInput() async throws {
        let service = try AdaptiveModelService()
        
        do {
            _ = try await service.analyzeGoal("")
            XCTFail("空输入应该抛出错误")
        } catch {
            // 预期会抛出错误
            XCTAssertTrue(true, "正确处理了空输入")
        }
    }
    
    @MainActor
    func testLongInput() async throws {
        let service = try AdaptiveModelService()
        let longText = String(repeating: "今天要完成项目开发工作，", count: 50)
        
        let result = try await service.analyzeGoal(longText)
        XCTAssertNotNil(result, "应该能处理长文本输入")
    }
    
    @MainActor
    func testSpecialCharacters() async throws {
        let service = try AdaptiveModelService()
        let specialText = "今天要完成项目！@#$%^&*()_+"
        
        let result = try await service.analyzeGoal(specialText)
        XCTAssertNotNil(result, "应该能处理特殊字符")
    }
    
    // MARK: - 批量测试
    
    @MainActor
    func testBatchProcessing() async throws {
        let service = try AdaptiveModelService()
        
        let entries = createMockEntries(count: 10)
        let results = try await service.analyzeBatch(entries)
        
        XCTAssertEqual(results.count, entries.count, "批量处理应返回相同数量的结果")
        print("📊 批量处理: \(results.count) 个样本")
    }
    
    // MARK: - 综合基准测试
    
    @MainActor
    func testComprehensiveBenchmark() async throws {
        guard let context = testContext else { return }
        let summary = await ModelBench.run(in: context)
        
        print("\n📈 综合测试结果:")
        print("   平均推理时间: \(String(format: "%.2f", summary.avg))ms")
        print("   P90 延迟: \(String(format: "%.2f", summary.p90))ms")
        print("   分类准确率: \(String(format: "%.1f%%", summary.catAcc * 100))")
        print("   情感准确率: \(String(format: "%.1f%%", summary.senAcc * 100))")
        print("   内存使用: \(String(format: "%.2f", summary.memMB))MB")
        
        // 性能断言
        XCTAssertLessThan(summary.avg, 200, "平均推理时间应小于 200ms")
        XCTAssertLessThan(summary.p90, 300, "P90 延迟应小于 300ms")
        
        // 准确率断言
        XCTAssertGreaterThanOrEqual(summary.catAcc, 0.7, "分类准确率应至少 70%")
        XCTAssertGreaterThanOrEqual(summary.senAcc, 0.6, "情感准确率应至少 60%")
        
        // 资源使用断言
        XCTAssertLessThan(summary.memMB, 150, "内存使用应小于 150MB")
    }
    
    // MARK: - Helper Methods
    
    private func createMockEntries(count: Int) -> [GoalEntry] {
        var entries: [GoalEntry] = []
        let sampleTexts = [
            "完成项目开发",
            "去健身房锻炼",
            "和家人吃饭",
            "学习新技能",
            "整理财务",
            "朋友聚会",
            "看电影放松",
            "阅读书籍"
        ]
        
        for i in 0..<count {
            guard let context = testContext else { break }
            let entry = GoalEntry(context: context)
            entry.goalText = sampleTexts[i % sampleTexts.count]
            entry.dateString = "2024-01-\(String(format: "%02d", i + 1))"
            entries.append(entry)
        }
        
        return entries
    }
}
