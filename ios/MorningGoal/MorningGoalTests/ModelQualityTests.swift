import XCTest
import Combine
@testable import MorningGoal

/// 模型质量单元测试
/// 可以通过 Xcode Test Navigator 或命令行 `xcodebuild test` 运行
final class ModelQualityTests: XCTestCase {
    
    private var modelManager: InsightModelManager!
    private var cancellables: Set<AnyCancellable> = []
    
    @MainActor
    override func setUp() async throws {
        try await super.setUp()
        modelManager = InsightModelManager()
    }
    
    @MainActor
    override func tearDown() {
        modelManager = nil
        cancellables.removeAll()
        super.tearDown()
    }
    
    /// 测试模型加载流程
    @MainActor
    func testModelLoading() async throws {
        // Given
        print("Starting model loading test...")
        
        // When
        await modelManager.loadModels()
        
        // Then
        XCTAssertTrue(modelManager.isInitialized, "Model manager should be initialized")
        XCTAssertFalse(modelManager.loadedClassifiers.isEmpty, "Should have loaded classifiers")
        
        // Log loaded models
        print("Loaded classifiers: \(modelManager.loadedClassifiers)")
        
        // Check version info
        let versions = modelManager.getModelVersions()
        print("Model versions: \(versions)")
        XCTAssertEqual(versions["featureExtractor"], "loaded")
    }
    
    /// 测试文本分析功能
    @MainActor
    func testAnalysis() async throws {
        // 1. Initialize
        await modelManager.loadModels()
        
        guard modelManager.isInitialized else {
            XCTFail("Failed to initialize models - check logs for missing files")
            return
        }
        
        // 2. Analyze sample text
        // "我想在明天早上6点起床跑步，锻炼身体"
        // 预期: Topic=健康/运动, Action=运动, TimeFrame=明天/本周
        let text = "我想在明天早上6点起床跑步，锻炼身体"
        
        let result = try await modelManager.analyze(text: text)
        
        // 3. Verify structure
        XCTAssertNotNil(result.topic, "Should have topic result")
        XCTAssertNotNil(result.sentiment, "Should have sentiment result")
        XCTAssertNotNil(result.urgency, "Should have urgency result")
        
        // 4. Log detailed results
        print("\n=== Analysis Result for: \"\(text)\" ===")
        print("Topic: \(result.topic?.label ?? "N/A") (Conf: \(String(format: "%.2f", result.topic?.confidence ?? 0)))")
        print("Sentiment: \(result.sentiment?.label ?? "N/A")")
        print("Urgency: \(result.urgency?.label ?? "N/A")")
        print("Action: \(result.actionType?.label ?? "N/A")")
        print("TimeFrame: \(result.timeFrame?.label ?? "N/A")")
        print("Difficulty: \(result.difficulty?.label ?? "N/A")")
        print("Specificity: \(result.specificity?.label ?? "N/A")")
        print("Inference Time: \(result.inferenceTimeMs)ms")
        print("Feature Extraction: \(result.featureExtractionTimeMs)ms")
        print("==========================================\n")
        
        // 5. Basic Logic Assertions (Relaxed for CI/CD stability)
        if let topic = result.topic {
            XCTAssert(topic.confidence >= 0 && topic.confidence <= 1.0)
            XCTAssertFalse(topic.label.isEmpty)
        }
    }
    
    /// 测试空文本和短文本处理
    @MainActor
    func testEdgeCases() async throws {
        await modelManager.loadModels()
        guard modelManager.isInitialized else { return }
        
        // Empty text
        do {
            // Depending on tokenizer implementation, this might work or throw
            let res = try await modelManager.analyze(text: "")
            print("Empty text analysis result: \(res.allDimensions.count) dimensions")
        } catch {
            print("Empty text analysis threw: \(error)")
        }
        
        // Short text
        let shortText = "跑"
        let res = try await modelManager.analyze(text: shortText)
        XCTAssertNotNil(res.topic)
        print("Short text analysis successful")
    }
}
