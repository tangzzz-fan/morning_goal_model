# iOS 项目 - 5个新分类器集成任务清单

> 基于 `User_Insight_System_Design.md` 设计文档
> 更新时间: 2026-01-24

---

## 项目现状分析

### 已有的模型架构

| 文件 | 功能 | 状态 |
|-----|------|------|
| `CoreMLAnalysisService.swift` | AnalysisResult/AnalysisService 协议定义 | ✅ 已有 |
| `UpdatableCoreMLGoalAnalysisService.swift` | 可更新的 k-NN 分类器服务 | ✅ 已有 |
| `AdaptiveModelService.swift` | 自适应模型服务 | ✅ 已有 |
| `EnhancedAdaptiveModelService.swift` | 增强版自适应服务 | ✅ 已有 |
| `InsightEngine.swift` | 洞察生成引擎 | ✅ 已有 (需扩展) |

### 已复制的新模型文件

```
ios/MorningGoal/MorningGoal/coreml/
├── BertFeatureExtractor.mlpackage       # 共享特征提取器
├── SentimentClassifier_Updatable.mlpackage  # 情感分类 (已有)
├── ActiontypeClassifier_Updatable.mlpackage # 行动类型 [新增]
├── DifficultyClassifier_Updatable.mlpackage # 难度 [新增]
├── SpecificityClassifier_Updatable.mlpackage # 具体程度 [新增]
├── TimeframeClassifier_Updatable.mlpackage  # 时间范围 [新增]
└── UrgencyClassifier_Updatable.mlpackage    # 紧急度 [新增]
```

### 当前 AnalysisResult 结构

```swift
struct AnalysisResult {
    let category: String           // Topic 分类
    let categoryConfidence: Double
    let sentiment: String          // Sentiment 分类
    let sentimentScore: Double
    // 缺少: urgency, timeFrame, actionType, difficulty, specificity
}
```

---

## Phase 1: 数据模型扩展

### 1.1 扩展 AnalysisResult 结构
- [ ] 添加 `urgency: String` 和 `urgencyConfidence: Double`
- [ ] 添加 `timeFrame: String` 和 `timeFrameConfidence: Double`
- [ ] 添加 `actionType: String` 和 `actionTypeConfidence: Double`
- [ ] 添加 `difficulty: String` 和 `difficultyConfidence: Double`
- [ ] 添加 `specificity: String` 和 `specificityConfidence: Double`
- [ ] 保持向后兼容的初始化器

### 1.2 扩展 GoalEntry Core Data 实体
- [ ] 添加 `urgency: String?` 和 `urgencyConfidence: Double`
- [ ] 添加 `timeFrame: String?` 和 `timeFrameConfidence: Double`
- [ ] 添加 `actionType: String?` 和 `actionTypeConfidence: Double`
- [ ] 添加 `difficulty: String?` 和 `difficultyConfidence: Double`
- [ ] 添加 `specificity: String?` 和 `specificityConfidence: Double`
- [ ] 添加用户纠正字段 (用于端侧训练)
- [ ] 更新 Core Data Model Builder

### 1.3 定义分类标签常量
- [ ] 创建 `InsightClassifierLabels.swift` 定义所有标签
  - Urgency: ["low", "medium", "high"]
  - TimeFrame: ["today", "this_week", "this_month", "long_term"]
  - ActionType: ["learning", "exercise", "work", "lifestyle", "social", "creative"]
  - Difficulty: ["easy", "moderate", "hard"]
  - Specificity: ["vague", "moderate", "specific"]

---

## Phase 2: 多分类器服务实现

### 2.1 创建 InsightModelManager 服务
- [ ] 新建 `InsightModelManager.swift`
- [ ] 加载 BertFeatureExtractor (共享特征提取器)
- [ ] 加载 7 个可更新分类器
- [ ] 实现 `analyze(text:) -> InsightAnalysisResult`
- [ ] 实现并行推理 (DispatchGroup / TaskGroup)

### 2.2 分类器加载逻辑
- [ ] 从 Bundle 加载原始模型
- [ ] 从 Documents 加载已更新模型 (如存在)
- [ ] 实现 fallback 逻辑

### 2.3 端侧更新功能
- [ ] 实现 `updateClassifier(dimension:embedding:correctLabel:)`
- [ ] 为每个分类器独立保存更新后的模型
- [ ] 实现模型版本管理

### 2.4 扩展 TrainingSample 结构
- [ ] 添加 `correctUrgency: String?`
- [ ] 添加 `correctTimeFrame: String?`
- [ ] 添加 `correctActionType: String?`
- [ ] 添加 `correctDifficulty: String?`
- [ ] 添加 `correctSpecificity: String?`

---

## Phase 3: 洞察引擎扩展

### 3.1 扩展 Insight 类型
- [ ] 添加新的洞察类型枚举
  - `urgencyPattern(high: Int, medium: Int, low: Int)`
  - `timeFrameDistribution(today: Int, thisWeek: Int, thisMonth: Int, longTerm: Int)`
  - `actionTypeBalance(dominant: String, percentage: Int)`
  - `difficultyTrend(direction: String)`
  - `specificityImprovement(improvement: Int)`

### 3.2 实现新的分析器
- [ ] `UrgencyAnalyzer` - 紧急度模式分析
- [ ] `TimeFrameAnalyzer` - 时间范围分布分析
- [ ] `ActionTypeAnalyzer` - 行动类型平衡分析
- [ ] `DifficultyAnalyzer` - 难度趋势分析
- [ ] `SpecificityAnalyzer` - 具体程度改进分析

### 3.3 更新 InsightEngine
- [ ] 集成新的分析器
- [ ] 更新 `generateInsights()` 方法
- [ ] 实现洞察优先级排序

---

## Phase 4: UI 更新

### 4.1 更新 InsightCardView
- [ ] 支持新的洞察类型显示
- [ ] 添加对应的图标和颜色

### 4.2 更新 TodayInputView
- [ ] 显示所有 7 个分类维度结果
- [ ] 支持用户纠正每个维度

### 4.3 创建 InsightDashboardView (可选)
- [ ] 显示综合洞察面板
- [ ] 趋势图表 (Swift Charts)
- [ ] 分布饼图

### 4.4 更新 ModelDebugView
- [ ] 显示 7 个分类器的状态
- [ ] 显示每个分类器的版本信息

---

## Phase 5: 测试与验证

### 5.1 单元测试
- [ ] InsightModelManager 测试
- [ ] 各分析器测试
- [ ] Core Data 扩展测试

### 5.2 性能测试
- [ ] 7 个分类器并行推理延迟 (目标 < 50ms)
- [ ] 内存占用测试

### 5.3 端侧更新测试
- [ ] 每个分类器的更新功能测试
- [ ] 模型持久化测试

---

## 依赖关系

```
Phase 1 (数据模型) 
    ↓
Phase 2 (分类器服务) ←── 需要先完成 Phase 1
    ↓
Phase 3 (洞察引擎) ←── 需要先完成 Phase 2
    ↓
Phase 4 (UI 更新) ←── 需要先完成 Phase 3
    ↓
Phase 5 (测试验证) ←── 贯穿所有阶段
```

---

## 优先级排序

| 任务 | 优先级 | 预计工时 |
|-----|--------|---------|
| 1.1 扩展 AnalysisResult | P0 | 0.5 天 |
| 1.2 扩展 GoalEntry | P0 | 1 天 |
| 1.3 定义标签常量 | P0 | 0.5 天 |
| 2.1 InsightModelManager | P0 | 2 天 |
| 2.2 分类器加载 | P0 | 1 天 |
| 2.3 端侧更新 | P1 | 1 天 |
| 3.1 扩展 Insight 类型 | P1 | 0.5 天 |
| 3.2 新分析器 | P1 | 2 天 |
| 4.1-4.4 UI 更新 | P2 | 2 天 |
| 5.1-5.3 测试 | P1 | 2 天 |
| **总计** | | **约 12 天** |
