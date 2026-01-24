# iOS 项目 - 5个新分类器集成实施计划

> 基于 `User_Insight_System_Design.md` 设计文档
> 更新时间: 2026-01-24

---

## 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户输入文本                              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│           BertFeatureExtractor.mlpackage (静态, 共享)            │
│                       → 512维 Embedding                         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
    ┌────────┬────────┬────┴────┬────────┬────────┬────────┐
    ▼        ▼        ▼         ▼        ▼        ▼        ▼
┌───────┐┌───────┐┌───────┐┌───────┐┌───────┐┌───────┐┌───────┐
│Topic  ││Senti- ││Urgency││Time-  ││Action-││Diffi- ││Speci- │
│(16cls)││ment   ││(3cls) ││Frame  ││Type   ││culty  ││ficity │
│  ✏️   ││(3cls) ││  ✏️   ││(4cls) ││(6cls) ││(3cls) ││(3cls) │
│       ││  ✏️   ││       ││  ✏️   ││  ✏️   ││  ✏️   ││  ✏️   │
└───┬───┘└───┬───┘└───┬───┘└───┬───┘└───┬───┘└───┬───┘└───┬───┘
    │        │        │        │        │        │        │
    └────────┴────────┴────┬───┴────────┴────────┴────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   InsightAnalysisResult                         │
│  (topic, sentiment, urgency, timeFrame, actionType,             │
│   difficulty, specificity + 各维度置信度)                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                       GoalEntry (Core Data)                     │
│                    存储所有分类结果 + 用户纠正                    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                       InsightEngine                             │
│              生成多维度洞察 (模式、平衡、趋势等)                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: 数据模型扩展

### 1.1 扩展 AnalysisResult

**文件**: `Services/CoreMLAnalysisService.swift`

```swift
// 扩展后的 AnalysisResult
struct AnalysisResult {
    // 原有字段
    let category: String
    let categoryConfidence: Double
    let sentiment: String
    let sentimentScore: Double
    
    // 新增字段 - 5个洞察维度
    let urgency: String?
    let urgencyConfidence: Double?
    let timeFrame: String?
    let timeFrameConfidence: Double?
    let actionType: String?
    let actionTypeConfidence: Double?
    let difficulty: String?
    let difficultyConfidence: Double?
    let specificity: String?
    let specificityConfidence: Double?
    
    // Top-K 结果 (可选)
    let topCategories: [(name: String, confidence: Double)]?
    let topSentiments: [(name: String, score: Double)]?
    
    // 向后兼容初始化器
    init(
        category: String,
        categoryConfidence: Double,
        sentiment: String,
        sentimentScore: Double,
        topCategories: [(name: String, confidence: Double)]? = nil,
        topSentiments: [(name: String, score: Double)]? = nil
    ) {
        self.category = category
        self.categoryConfidence = categoryConfidence
        self.sentiment = sentiment
        self.sentimentScore = sentimentScore
        self.urgency = nil
        self.urgencyConfidence = nil
        self.timeFrame = nil
        self.timeFrameConfidence = nil
        self.actionType = nil
        self.actionTypeConfidence = nil
        self.difficulty = nil
        self.difficultyConfidence = nil
        self.specificity = nil
        self.specificityConfidence = nil
        self.topCategories = topCategories
        self.topSentiments = topSentiments
    }
    
    // 完整初始化器
    init(
        category: String,
        categoryConfidence: Double,
        sentiment: String,
        sentimentScore: Double,
        urgency: String?,
        urgencyConfidence: Double?,
        timeFrame: String?,
        timeFrameConfidence: Double?,
        actionType: String?,
        actionTypeConfidence: Double?,
        difficulty: String?,
        difficultyConfidence: Double?,
        specificity: String?,
        specificityConfidence: Double?,
        topCategories: [(name: String, confidence: Double)]? = nil,
        topSentiments: [(name: String, score: Double)]? = nil
    ) {
        self.category = category
        self.categoryConfidence = categoryConfidence
        self.sentiment = sentiment
        self.sentimentScore = sentimentScore
        self.urgency = urgency
        self.urgencyConfidence = urgencyConfidence
        self.timeFrame = timeFrame
        self.timeFrameConfidence = timeFrameConfidence
        self.actionType = actionType
        self.actionTypeConfidence = actionTypeConfidence
        self.difficulty = difficulty
        self.difficultyConfidence = difficultyConfidence
        self.specificity = specificity
        self.specificityConfidence = specificityConfidence
        self.topCategories = topCategories
        self.topSentiments = topSentiments
    }
}
```

### 1.2 创建分类标签常量

**新建文件**: `Models/InsightClassifierLabels.swift`

```swift
import Foundation

/// 洞察分类器标签定义
enum InsightLabels {
    
    // MARK: - Urgency (紧急度)
    enum Urgency: String, CaseIterable {
        case low = "low"
        case medium = "medium"
        case high = "high"
        
        var displayName: String {
            switch self {
            case .low: return "低"
            case .medium: return "中"
            case .high: return "高"
            }
        }
        
        var index: Int {
            switch self {
            case .low: return 0
            case .medium: return 1
            case .high: return 2
            }
        }
    }
    
    // MARK: - TimeFrame (时间范围)
    enum TimeFrame: String, CaseIterable {
        case today = "today"
        case thisWeek = "this_week"
        case thisMonth = "this_month"
        case longTerm = "long_term"
        
        var displayName: String {
            switch self {
            case .today: return "今天"
            case .thisWeek: return "本周"
            case .thisMonth: return "本月"
            case .longTerm: return "长期"
            }
        }
        
        var index: Int {
            switch self {
            case .today: return 0
            case .thisWeek: return 1
            case .thisMonth: return 2
            case .longTerm: return 3
            }
        }
    }
    
    // MARK: - ActionType (行动类型)
    enum ActionType: String, CaseIterable {
        case learning = "learning"
        case exercise = "exercise"
        case work = "work"
        case lifestyle = "lifestyle"
        case social = "social"
        case creative = "creative"
        
        var displayName: String {
            switch self {
            case .learning: return "学习"
            case .exercise: return "运动"
            case .work: return "工作"
            case .lifestyle: return "生活"
            case .social: return "社交"
            case .creative: return "创意"
            }
        }
        
        var index: Int {
            switch self {
            case .learning: return 0
            case .exercise: return 1
            case .work: return 2
            case .lifestyle: return 3
            case .social: return 4
            case .creative: return 5
            }
        }
    }
    
    // MARK: - Difficulty (难度)
    enum Difficulty: String, CaseIterable {
        case easy = "easy"
        case moderate = "moderate"
        case hard = "hard"
        
        var displayName: String {
            switch self {
            case .easy: return "简单"
            case .moderate: return "中等"
            case .hard: return "困难"
            }
        }
        
        var index: Int {
            switch self {
            case .easy: return 0
            case .moderate: return 1
            case .hard: return 2
            }
        }
    }
    
    // MARK: - Specificity (具体程度)
    enum Specificity: String, CaseIterable {
        case vague = "vague"
        case moderate = "moderate"
        case specific = "specific"
        
        var displayName: String {
            switch self {
            case .vague: return "模糊"
            case .moderate: return "一般"
            case .specific: return "具体"
            }
        }
        
        var index: Int {
            switch self {
            case .vague: return 0
            case .moderate: return 1
            case .specific: return 2
            }
        }
    }
}
```

### 1.3 扩展 GoalEntry Core Data 实体

**更新文件**: `Models/GoalEntry.swift`

```swift
@objc(GoalEntry)
final class GoalEntry: NSManagedObject, Identifiable {
    // ... 原有字段 ...
    
    // 原有分析结果
    @NSManaged var category: String?
    @NSManaged var categoryConfidence: Double
    @NSManaged var sentiment: String?
    @NSManaged var sentimentScore: Double
    
    // 新增: 5个洞察维度
    @NSManaged var urgency: String?
    @NSManaged var urgencyConfidence: Double
    @NSManaged var timeFrame: String?
    @NSManaged var timeFrameConfidence: Double
    @NSManaged var actionType: String?
    @NSManaged var actionTypeConfidence: Double
    @NSManaged var difficulty: String?
    @NSManaged var difficultyConfidence: Double
    @NSManaged var specificity: String?
    @NSManaged var specificityConfidence: Double
    
    // 新增: 用户纠正字段
    @NSManaged var urgencyUserCorrected: String?
    @NSManaged var timeFrameUserCorrected: String?
    @NSManaged var actionTypeUserCorrected: String?
    @NSManaged var difficultyUserCorrected: String?
    @NSManaged var specificityUserCorrected: String?
    
    // 便捷访问器
    var effectiveUrgency: String? { urgencyUserCorrected ?? urgency }
    var effectiveTimeFrame: String? { timeFrameUserCorrected ?? timeFrame }
    var effectiveActionType: String? { actionTypeUserCorrected ?? actionType }
    var effectiveDifficulty: String? { difficultyUserCorrected ?? difficulty }
    var effectiveSpecificity: String? { specificityUserCorrected ?? specificity }
}
```

**更新文件**: `Services/Storage/CoreDataModelBuilder.swift`

添加新属性到 GoalEntry 实体定义。

---

## Phase 2: 多分类器服务实现

### 2.1 InsightModelManager

**新建文件**: `Services/InsightModelManager.swift`

```swift
import CoreML
import Foundation
import OSLog

/// 分类器配置
struct ClassifierConfig {
    let name: String
    let numClasses: Int
    let labels: [String]
    let bundleName: String
    let outputName: String
}

/// 洞察模型管理器 - 管理所有7个分类器
@MainActor
final class InsightModelManager {
    
    // MARK: - Properties
    
    private let logger = Logger(subsystem: "com.morninggoal.app", category: "insight-model")
    
    /// 共享特征提取器 (静态)
    private var featureExtractor: MLModel?
    
    /// 分类器字典
    private var classifiers: [String: MLModel] = [:]
    
    /// Tokenizer
    private let tokenizer: BertTokenizer
    private let maxLength: Int = 128
    
    /// 分类器配置
    static let classifierConfigs: [ClassifierConfig] = [
        ClassifierConfig(
            name: "urgency",
            numClasses: 3,
            labels: ["low", "medium", "high"],
            bundleName: "UrgencyClassifier_Updatable",
            outputName: "urgency_probs"
        ),
        ClassifierConfig(
            name: "timeFrame",
            numClasses: 4,
            labels: ["today", "this_week", "this_month", "long_term"],
            bundleName: "TimeframeClassifier_Updatable",
            outputName: "timeFrame_probs"
        ),
        ClassifierConfig(
            name: "actionType",
            numClasses: 6,
            labels: ["learning", "exercise", "work", "lifestyle", "social", "creative"],
            bundleName: "ActiontypeClassifier_Updatable",
            outputName: "actionType_probs"
        ),
        ClassifierConfig(
            name: "difficulty",
            numClasses: 3,
            labels: ["easy", "moderate", "hard"],
            bundleName: "DifficultyClassifier_Updatable",
            outputName: "difficulty_probs"
        ),
        ClassifierConfig(
            name: "specificity",
            numClasses: 3,
            labels: ["vague", "moderate", "specific"],
            bundleName: "SpecificityClassifier_Updatable",
            outputName: "specificity_probs"
        ),
    ]
    
    /// Documents 目录
    private var documentsDirectory: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }
    
    // MARK: - Initialization
    
    init() throws {
        let bundle = Bundle.main
        
        // 1. 加载 tokenizer
        guard let vocabURL = bundle.url(forResource: "vocab", withExtension: "txt") else {
            throw AnalysisError.modelNotLoaded
        }
        self.tokenizer = try BertTokenizer(vocabURL: vocabURL)
        
        // 2. 加载特征提取器
        try loadFeatureExtractor()
        
        // 3. 加载所有分类器
        for config in Self.classifierConfigs {
            try loadClassifier(config: config)
        }
        
        logger.info("✅ InsightModelManager initialized with \(self.classifiers.count) classifiers")
    }
    
    // MARK: - Model Loading
    
    private func loadFeatureExtractor() throws {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        
        // 优先从 coreml 子目录加载
        if let url = Bundle.main.url(forResource: "BertFeatureExtractor", withExtension: "mlpackage", subdirectory: "coreml") {
            self.featureExtractor = try MLModel(contentsOf: url, configuration: config)
            logger.info("✅ Loaded BertFeatureExtractor from coreml/")
        } else if let url = Bundle.main.url(forResource: "BertFeatureExtractor", withExtension: "mlpackage") {
            self.featureExtractor = try MLModel(contentsOf: url, configuration: config)
            logger.info("✅ Loaded BertFeatureExtractor from Bundle root")
        } else {
            throw AnalysisError.modelNotLoaded
        }
    }
    
    private func loadClassifier(config: ClassifierConfig) throws {
        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = .cpuAndNeuralEngine
        
        // 检查是否有已更新的模型
        let updatedPath = documentsDirectory.appendingPathComponent("\(config.bundleName).mlmodelc")
        
        if FileManager.default.fileExists(atPath: updatedPath.path) {
            classifiers[config.name] = try MLModel(contentsOf: updatedPath, configuration: mlConfig)
            logger.info("✅ Loaded updated \(config.name) classifier from Documents")
        } else {
            // 从 Bundle 加载
            if let url = Bundle.main.url(forResource: config.bundleName, withExtension: "mlpackage", subdirectory: "coreml") {
                classifiers[config.name] = try MLModel(contentsOf: url, configuration: mlConfig)
                logger.info("✅ Loaded \(config.name) classifier from coreml/")
            } else if let url = Bundle.main.url(forResource: config.bundleName, withExtension: "mlpackage") {
                classifiers[config.name] = try MLModel(contentsOf: url, configuration: mlConfig)
                logger.info("✅ Loaded \(config.name) classifier from Bundle root")
            } else {
                logger.warning("⚠️ \(config.name) classifier not found, skipping")
            }
        }
    }
    
    // MARK: - Inference
    
    /// 分析目标文本，返回所有维度的分类结果
    func analyze(text: String) async throws -> [String: (label: String, confidence: Double)] {
        guard let featureExtractor = featureExtractor else {
            throw AnalysisError.modelNotLoaded
        }
        
        let t0 = CFAbsoluteTimeGetCurrent()
        
        // 1. Tokenize
        let enc = tokenizer.encode(text, maxLength: maxLength)
        
        // 2. 创建输入张量
        let ids = try MLMultiArray(shape: [1, NSNumber(value: maxLength)], dataType: .int32)
        let mask = try MLMultiArray(shape: [1, NSNumber(value: maxLength)], dataType: .int32)
        
        for i in 0..<maxLength {
            ids[i] = NSNumber(value: enc.ids[i])
            mask[i] = NSNumber(value: enc.mask[i])
        }
        
        // 3. 提取特征 (只做一次)
        let feInput = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: ids),
            "attention_mask": MLFeatureValue(multiArray: mask)
        ])
        
        let feOutput = try await featureExtractor.prediction(from: feInput)
        
        guard let embedding = feOutput.featureValue(for: "embedding")?.multiArrayValue else {
            throw AnalysisError.modelNotLoaded
        }
        
        // 4. 并行运行所有分类器
        var results: [String: (label: String, confidence: Double)] = [:]
        
        await withTaskGroup(of: (String, String, Double)?.self) { group in
            for config in Self.classifierConfigs {
                guard let classifier = classifiers[config.name] else { continue }
                
                group.addTask {
                    do {
                        let input = try MLDictionaryFeatureProvider(dictionary: [
                            "embedding": MLFeatureValue(multiArray: embedding)
                        ])
                        
                        let output = try await classifier.prediction(from: input)
                        
                        guard let probs = output.featureValue(for: config.outputName)?.multiArrayValue else {
                            return nil
                        }
                        
                        // 找到最大概率的索引
                        var maxIdx = 0
                        var maxProb = 0.0
                        for i in 0..<probs.count {
                            let prob = probs[i].doubleValue
                            if prob > maxProb {
                                maxProb = prob
                                maxIdx = i
                            }
                        }
                        
                        let label = config.labels.indices.contains(maxIdx) ? config.labels[maxIdx] : "unknown"
                        return (config.name, label, maxProb)
                    } catch {
                        return nil
                    }
                }
            }
            
            for await result in group {
                if let (name, label, confidence) = result {
                    results[name] = (label, confidence)
                }
            }
        }
        
        let t1 = CFAbsoluteTimeGetCurrent()
        logger.info("inference_ms=\(String(format: "%.2f", (t1 - t0) * 1000)), classifiers=\(results.count)")
        
        return results
    }
    
    // MARK: - Model Update
    
    /// 更新指定维度的分类器
    func updateClassifier(
        dimension: String,
        embedding: MLMultiArray,
        correctLabel: String
    ) async throws {
        guard let config = Self.classifierConfigs.first(where: { $0.name == dimension }) else {
            throw AnalysisError.invalidInput
        }
        
        guard let bundleURL = Bundle.main.url(forResource: config.bundleName, withExtension: "mlpackage", subdirectory: "coreml") ??
              Bundle.main.url(forResource: config.bundleName, withExtension: "mlpackage") else {
            throw AnalysisError.modelNotLoaded
        }
        
        // 准备训练数据
        let labelIndex = config.labels.firstIndex(of: correctLabel) ?? 0
        
        // 创建 one-hot 标签
        let labelArray = try MLMultiArray(shape: [NSNumber(value: config.numClasses)], dataType: .float32)
        for i in 0..<config.numClasses {
            labelArray[i] = i == labelIndex ? 1.0 : 0.0
        }
        
        let trainingFeatures = try MLDictionaryFeatureProvider(dictionary: [
            "embedding": MLFeatureValue(multiArray: embedding),
            "\(config.name)_probs_true": MLFeatureValue(multiArray: labelArray)
        ])
        
        let batchProvider = MLArrayBatchProvider(array: [trainingFeatures])
        
        // 创建更新任务
        let updatedPath = documentsDirectory.appendingPathComponent("\(config.bundleName).mlmodelc")
        
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            do {
                let updateTask = try MLUpdateTask(
                    forModelAt: bundleURL,
                    trainingData: batchProvider,
                    configuration: nil
                ) { context in
                    switch context.task.state {
                    case .completed:
                        do {
                            try context.model.write(to: updatedPath)
                            self.logger.info("✅ Updated \(dimension) classifier saved")
                            continuation.resume()
                        } catch {
                            continuation.resume(throwing: error)
                        }
                    case .failed:
                        continuation.resume(throwing: context.task.error ?? AnalysisError.updateFailed)
                    default:
                        break
                    }
                }
                
                updateTask.resume()
            } catch {
                continuation.resume(throwing: error)
            }
        }
        
        // 重新加载更新后的模型
        try loadClassifier(config: config)
    }
    
    // MARK: - Version Info
    
    func getModelVersions() -> [String: String] {
        var versions: [String: String] = [:]
        
        for config in Self.classifierConfigs {
            let updatedPath = documentsDirectory.appendingPathComponent("\(config.bundleName).mlmodelc")
            let isUpdated = FileManager.default.fileExists(atPath: updatedPath.path)
            versions[config.name] = isUpdated ? "updated" : "bundled"
        }
        
        return versions
    }
}
```

---

## Phase 3: 洞察引擎扩展

### 3.1 扩展 Insight 类型

**更新文件**: `Services/InsightEngine.swift`

```swift
enum Insight: Equatable {
    // 原有类型
    case categoryDistribution(topCategory: String, percentage: Int)
    case streakBooster(category: String, improvement: Int)
    case consistencyPattern(category: String, days: Int)
    case sentimentTrend(direction: String)
    
    // 新增类型
    case urgencyPattern(highCount: Int, mediumCount: Int, lowCount: Int)
    case timeFrameDistribution(today: Int, thisWeek: Int, thisMonth: Int, longTerm: Int)
    case actionTypeBalance(dominant: String, percentage: Int, suggestion: String)
    case difficultyTrend(direction: String, avgDifficulty: String)
    case specificityImprovement(improvement: Int, recentAvg: String)
    case achievabilityWarning(reason: String, suggestion: String)
    
    var id: String {
        switch self {
        case .categoryDistribution: return "category_dist"
        case .streakBooster: return "streak_boost"
        case .consistencyPattern: return "consistency"
        case .sentimentTrend: return "sentiment"
        case .urgencyPattern: return "urgency_pattern"
        case .timeFrameDistribution: return "timeframe_dist"
        case .actionTypeBalance: return "actiontype_balance"
        case .difficultyTrend: return "difficulty_trend"
        case .specificityImprovement: return "specificity_improvement"
        case .achievabilityWarning: return "achievability_warning"
        }
    }
    
    var priority: Int {
        switch self {
        case .achievabilityWarning: return 5
        case .actionTypeBalance: return 4
        case .urgencyPattern: return 3
        case .streakBooster: return 3
        case .sentimentTrend: return 2
        case .difficultyTrend: return 2
        case .specificityImprovement: return 2
        case .categoryDistribution: return 1
        case .consistencyPattern: return 1
        case .timeFrameDistribution: return 1
        }
    }
}
```

### 3.2 新增分析器

```swift
// MARK: - 紧急度分析

extension InsightEngine {
    func analyzeUrgencyPattern(_ entries: [GoalEntry]) -> Insight? {
        let urgencyCounts = Dictionary(grouping: entries) { $0.effectiveUrgency ?? "low" }
        
        let high = urgencyCounts["high"]?.count ?? 0
        let medium = urgencyCounts["medium"]?.count ?? 0
        let low = urgencyCounts["low"]?.count ?? 0
        
        guard entries.count >= 7 else { return nil }
        
        let highRatio = Double(high) / Double(entries.count)
        
        if highRatio > 0.5 {
            return .urgencyPattern(highCount: high, mediumCount: medium, lowCount: low)
        }
        
        return nil
    }
    
    func analyzeActionTypeBalance(_ entries: [GoalEntry]) -> Insight? {
        let typeCounts = Dictionary(grouping: entries) { $0.effectiveActionType ?? "lifestyle" }
        let total = entries.count
        
        guard total >= 7, let dominant = typeCounts.max(by: { $0.value.count < $1.value.count }) else {
            return nil
        }
        
        let percentage = Int(Double(dominant.value.count) / Double(total) * 100)
        
        if percentage > 60 {
            let suggestion = suggestComplementary(dominant.key)
            return .actionTypeBalance(dominant: dominant.key, percentage: percentage, suggestion: suggestion)
        }
        
        return nil
    }
    
    private func suggestComplementary(_ type: String) -> String {
        switch type {
        case "work": return "运动或社交"
        case "exercise": return "学习或创意"
        case "learning": return "社交或生活"
        case "lifestyle": return "运动或学习"
        case "social": return "学习或工作"
        case "creative": return "运动或社交"
        default: return "其他类型"
        }
    }
}
```

---

## Phase 4: 将分类器添加到 Xcode 项目

### 4.1 添加模型文件到项目

1. 在 Xcode 中，右键点击 `MorningGoal` 文件夹
2. 选择 "Add Files to MorningGoal..."
3. 选择 `coreml/` 目录下的所有 `.mlpackage` 文件
4. 确保勾选 "Copy items if needed" 和 "Create folder references"
5. 确保 Target Membership 选中 `MorningGoal`

### 4.2 验证模型文件已添加

检查 `project.pbxproj` 中是否包含:
- `BertFeatureExtractor.mlpackage`
- `UrgencyClassifier_Updatable.mlpackage`
- `TimeframeClassifier_Updatable.mlpackage`
- `ActiontypeClassifier_Updatable.mlpackage`
- `DifficultyClassifier_Updatable.mlpackage`
- `SpecificityClassifier_Updatable.mlpackage`
- `SentimentClassifier_Updatable.mlpackage`

---

## 性能优化建议

### 并行推理
使用 `TaskGroup` 并行执行 5 个分类器，预期总推理时间 < 50ms

### 内存优化
- 特征提取器只加载一次
- 分类器按需加载
- 使用 `autoreleasepool` 控制临时对象

### 电池优化
- 批量处理洞察计算
- 避免在后台频繁推理

---

## 测试检查清单

- [ ] 所有模型文件正确添加到 Xcode 项目
- [ ] InsightModelManager 初始化成功
- [ ] 5 个分类器全部加载成功
- [ ] 并行推理延迟 < 50ms
- [ ] 端侧更新功能正常
- [ ] GoalEntry 新字段持久化正常
- [ ] InsightEngine 生成新类型洞察
