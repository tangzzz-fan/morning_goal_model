import CoreData
import CoreML
import Foundation

// MARK: - 分析结果模型

struct GoalAnalysisResult {
    // MARK: - 原有维度

    let category: String // Topic (主题)
    let categoryConfidence: Double
    let sentiment: String // Sentiment (情感)
    let sentimentScore: Double

    // MARK: - 新增维度 (5个)

    let urgency: String? // Urgency (紧急度)
    let urgencyConfidence: Double
    let timeFrame: String? // TimeFrame (时间范围)
    let timeFrameConfidence: Double
    let actionType: String? // ActionType (行动类型)
    let actionTypeConfidence: Double
    let difficulty: String? // Difficulty (难度)
    let difficultyConfidence: Double
    let specificity: String? // Specificity (具体程度)
    let specificityConfidence: Double

    // MARK: - 可选数据

    let embedding: [Float]? // 768维向量，用于相似度搜索

    // Top-K 结果 (用于 v2.0 AI 洞察)
    // 例如: [("工作", 0.456), ("学习", 0.234), ("个人发展", 0.156)]
    let topCategories: [(name: String, confidence: Double)]?
    let topSentiments: [(name: String, score: Double)]?

    // 便捷初始化器 (保持向后兼容)
    init(
        category: String,
        categoryConfidence: Double,
        sentiment: String,
        sentimentScore: Double,
        urgency: String? = nil,
        urgencyConfidence: Double = 0.0,
        timeFrame: String? = nil,
        timeFrameConfidence: Double = 0.0,
        actionType: String? = nil,
        actionTypeConfidence: Double = 0.0,
        difficulty: String? = nil,
        difficultyConfidence: Double = 0.0,
        specificity: String? = nil,
        specificityConfidence: Double = 0.0,
        embedding: [Float]? = nil,
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
        self.embedding = embedding
        self.topCategories = topCategories
        self.topSentiments = topSentiments
    }

    /// 从 InsightAnalysisResult 转换
    init(
        from insightResult: InsightAnalysisResult,
        topCategories: [(name: String, confidence: Double)]? = nil,
        topSentiments: [(name: String, score: Double)]? = nil
    ) {
        self.category = insightResult.topic?.label ?? "unknown"
        self.categoryConfidence = insightResult.topic?.confidence ?? 0.0
        self.sentiment = insightResult.sentiment?.label ?? "中性"
        self.sentimentScore = insightResult.sentiment?.confidence ?? 0.0
        self.urgency = insightResult.urgency?.label
        self.urgencyConfidence = insightResult.urgency?.confidence ?? 0.0
        self.timeFrame = insightResult.timeFrame?.label
        self.timeFrameConfidence = insightResult.timeFrame?.confidence ?? 0.0
        self.actionType = insightResult.actionType?.label
        self.actionTypeConfidence = insightResult.actionType?.confidence ?? 0.0
        self.difficulty = insightResult.difficulty?.label
        self.difficultyConfidence = insightResult.difficulty?.confidence ?? 0.0
        self.specificity = insightResult.specificity?.label
        self.specificityConfidence = insightResult.specificity?.confidence ?? 0.0
        self.embedding = nil // 可以通过 InsightModelManager 获取
        self.topCategories = topCategories
        self.topSentiments = topSentiments
    }
}

struct TrainingSample {
    let text: String
    let correctCategory: String
    let correctSentiment: String
}

// MARK: - 分析服务协议

protocol AnalysisService {
    func analyzeGoal(_ text: String) async throws -> GoalAnalysisResult
    func analyzeBatch(_ entries: [GoalEntry]) async throws -> [GoalAnalysisResult]
    func updateModel(with corrections: [TrainingSample]) async throws
    func getModelVersion() -> String
}

// MARK: - 错误类型

enum AnalysisError: Error, LocalizedError {
    case modelNotLoaded
    case invalidInput
    case updateFailed
    case networkUnavailable

    var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "模型未加载成功，请重启应用"
        case .invalidInput:
            return "输入文本格式无效"
        case .updateFailed:
            return "模型更新失败，将在下次尝试"
        case .networkUnavailable:
            return "网络不可用"
        }
    }
}

// MARK: - 临时规则引擎实现

/// 临时的规则引擎实现，用于在CoreML模型准备好之前提供基础功能
@MainActor
class RuleBasedAnalysisService: AnalysisService {
    // 关键词字典（基于Python原型）
    private let categoryKeywords: [String: [String]] = [
        "工作": [
            "工作",
            "项目",
            "客户",
            "会议",
            "代码",
            "开发",
            "系统",
            "功能",
            "演示",
            "团队",
            "技术",
            "性能",
            "优化",
            "专业",
            "邮件",
            "文档",
            "报告",
            "任务",
            "加班",
            "线上",
            "问题",
            "截止",
            "积压",
            "头脑风暴",
            "创新"
        ],
        "健康": [
            "健康",
            "跑步",
            "锻炼",
            "运动",
            "饮食",
            "瑜伽",
            "健身",
            "睡眠",
            "水",
            "早睡",
            "早起",
            "休息",
            "早餐",
            "午休",
            "暴饮暴食",
            "咖啡",
            "公里",
            "蔬菜",
            "水果",
            "充足"
        ],
        "家庭": [
            "家庭",
            "孩子",
            "家人",
            "晚餐",
            "父母",
            "郊游",
            "陪伴",
            "亲子",
            "家务",
            "房间",
            "另一半",
            "电影",
            "买菜",
            "做饭"
        ],
        "个人发展": [
            "阅读",
            "书籍",
            "学习",
            "成长",
            "发展",
            "写作",
            "博客",
            "反思",
            "收获",
            "总结",
            "笔记",
            "规划",
            "目标",
            "提升"
        ],
        "财务": [
            "财务",
            "开支",
            "预算",
            "投资",
            "理财",
            "账单",
            "银行",
            "账户",
            "余额",
            "支付",
            "金融",
            "资产",
            "报表"
        ],
        "学习": [
            "课程",
            "章节",
            "算法",
            "编程",
            "研究",
            "练习",
            "工具",
            "技能",
            "分享会",
            "复习",
            "Python",
            "机器学习",
            "在线",
            "深入"
        ],
        "社交": [
            "朋友",
            "社交",
            "约饭",
            "聚餐",
            "活动",
            "认识",
            "同事",
            "团建",
            "社区",
            "帮助",
            "叙旧",
            "组织",
            "消息"
        ],
        "休闲": [
            "休闲",
            "电影",
            "音乐",
            "放松",
            "画画",
            "手工",
            "游戏",
            "咖啡",
            "咖啡厅",
            "照片",
            "媒体",
            "安静",
            "享受",
            "什么都不做"
        ]
    ]

    private let positiveKeywords = [
        "完成", "成功", "好", "棒", "优化", "提升", "增进", "坚持",
        "充足", "新", "深入", "准备", "规划", "帮助", "享受", "喜欢"
    ]

    private let negativeKeywords = [
        "紧急", "问题", "加班", "赶", "积压", "不要", "暴饮", "控制",
        "减少", "不必要", "中断", "压力"
    ]

    // MARK: - 推理接口

    func analyzeGoal(_ text: String) async throws -> GoalAnalysisResult {
        guard !text.isEmpty else {
            throw AnalysisError.invalidInput
        }

        // 主题分类
        let category = classifyCategory(text)

        // 情感分析
        let sentimentResult = analyzeSentiment(text)

        return GoalAnalysisResult(
            category: category.name,
            categoryConfidence: category.confidence,
            sentiment: sentimentResult.sentiment,
            sentimentScore: sentimentResult.score
        )
    }

    func analyzeBatch(_ entries: [GoalEntry]) async throws -> [GoalAnalysisResult] {
        var results: [GoalAnalysisResult] = []

        for entry in entries {
            let result = try await analyzeGoal(entry.goalText)
            results.append(result)
        }

        return results
    }

    // MARK: - 设备端训练接口（规则引擎不支持）

    func updateModel(with corrections: [TrainingSample]) async throws {
        // 规则引擎不支持动态学习
        // 这个方法在使用CoreML实现时会被实际实现
        print("⚠️ 规则引擎不支持模型更新，请使用CoreML版本")
    }

    func getModelVersion() -> String {
        return "rule-based-v1.0"
    }

    // MARK: - 分类逻辑

    private func classifyCategory(_ text: String) -> (name: String, confidence: Double) {
        var scores: [String: Int] = [:]

        // 计算每个类别的关键词匹配分数
        for (category, keywords) in categoryKeywords {
            let score = keywords.filter { text.contains($0) }.count
            scores[category] = score
        }

        // 找到最高分
        guard let best = scores.max(by: { $0.value < $1.value }), best.value > 0 else {
            return (name: "个人发展", confidence: 0.5) // 默认类别
        }

        // 计算置信度（基于匹配的关键词数量）
        let totalMatches = scores.values.reduce(0, +)
        let confidence = Double(best.value) / Double(totalMatches + 1)

        return (name: best.key, confidence: min(confidence, 0.95))
    }

    private func analyzeSentiment(_ text: String) -> (sentiment: String, score: Double) {
        let positiveCount = positiveKeywords.filter { text.contains($0) }.count
        let negativeCount = negativeKeywords.filter { text.contains($0) }.count

        let sentiment: String
        let score: Double

        if negativeCount > positiveCount {
            sentiment = "消极"
            score = -Double(negativeCount) / Double(positiveCount + negativeCount + 1)
        } else if positiveCount > negativeCount {
            sentiment = "积极"
            score = Double(positiveCount) / Double(positiveCount + negativeCount + 1)
        } else {
            sentiment = "中性"
            score = 0.0
        }

        return (sentiment, score)
    }
}

// MARK: - CoreML实现（待模型文件导入后启用）

/*
 /// CoreML版本的分析服务
 /// 需要先将.mlpackage文件导入项目才能编译
 @MainActor
 class CoreMLAnalysisService: AnalysisService {
     private var categoryModel: GoalCategoryClassifier?
     private var sentimentModel: GoalSentimentAnalyzer?

     init() {
         do {
             let config = MLModelConfiguration()
             config.computeUnits = .cpuAndGPU

             categoryModel = try GoalCategoryClassifier(configuration: config)
             sentimentModel = try GoalSentimentAnalyzer(configuration: config)

             print("✅ CoreML模型加载成功")
         } catch {
             print("❌ CoreML模型加载失败: \(error)")
         }
     }

     func analyzeGoal(_ text: String) async throws -> AnalysisResult {
         guard let categoryModel = categoryModel,
               let sentimentModel = sentimentModel else {
             throw AnalysisError.modelNotLoaded
         }

         // 主题分类
         let categoryInput = GoalCategoryClassifierInput(text: text)
         let categoryOutput = try categoryModel.prediction(input: categoryInput)

         // 情感分析
         let sentimentInput = GoalSentimentAnalyzerInput(text: text)
         let sentimentOutput = try sentimentModel.prediction(input: sentimentInput)

         // 提取置信度
         let categoryConfidence = categoryOutput.labelProbability[categoryOutput.label] ?? 0.0
         let sentimentConfidence = sentimentOutput.labelProbability[sentimentOutput.label] ?? 0.0

         return AnalysisResult(
             category: categoryOutput.label,
             categoryConfidence: categoryConfidence,
             sentiment: sentimentOutput.label,
             sentimentScore: sentimentConfidence
         )
     }

     func analyzeBatch(_ entries: [GoalEntry]) async throws -> [AnalysisResult] {
         var results: [AnalysisResult] = []

         for entry in entries {
            let result = try await analyzeGoal(entry.goalText)
             results.append(result)
         }

         return results
     }

     func updateModel(with corrections: [TrainingSample]) async throws {
         guard let categoryModel = categoryModel else {
             throw AnalysisError.modelNotLoaded
         }

         print("🔄 开始设备端模型更新...")

         // 准备训练数据
         let trainingData = try prepareTrainingBatch(corrections)

         // 获取模型URL
         guard let modelURL = Bundle.main.url(
             forResource: "GoalCategoryClassifier",
             withExtension: "mlmodelc"
         ) else {
             throw AnalysisError.modelNotLoaded
         }

         // 临时更新模型URL
         let tempURL = FileManager.default.temporaryDirectory
             .appendingPathComponent("updated_model.mlmodelc")

         // 创建更新任务
         let updateTask = try MLUpdateTask(
             forModelAt: modelURL,
             trainingData: trainingData,
             configuration: nil,
             completionHandler: { context in
                 switch context.task.state {
                 case .completed:
                     print("✅ 模型更新完成")
                     do {
                         try context.model.write(to: tempURL)
                         // TODO: 替换当前模型
                     } catch {
                         print("❌ 保存更新模型失败: \(error)")
                     }

                 case .failed:
                     print("❌ 模型更新失败: \(context.task.error?.localizedDescription ?? "")")

                 default:
                     break
                 }
             }
         )

         updateTask.resume()
     }

     private func prepareTrainingBatch(_ samples: [TrainingSample]) throws -> MLBatchProvider {
         let featureProviders = samples.compactMap { sample -> MLFeatureProvider? in
             do {
                 let features: [String: Any] = [
                     "text": sample.text,
                     "label": sample.correctCategory
                 ]
                 return try MLDictionaryFeatureProvider(dictionary: features)
             } catch {
                 print("❌ 特征提取失败: \(error)")
                 return nil
             }
         }

         return MLArrayBatchProvider(array: featureProviders)
     }

     func getModelVersion() -> String {
         return categoryModel?.model.modelDescription.metadata[.versionString] as? String ?? "unknown"
     }
 }
 */

// MARK: - 全局单例（便于使用）

extension AnalysisService where Self == RuleBasedAnalysisService {
    @MainActor
    static var shared: AnalysisService {
        return RuleBasedAnalysisService()
    }
}
