# 用户数据洞察系统 - 剩余任务清单

> 基于 `User_Insight_System_Design.md` 设计文档
> 更新时间: 2026-01-25

---

## 进度总览

| 阶段 | 内容 | 状态 | 交付物 |
|-----|------|------|--------|
| **Phase 1** | 数据准备 & 标注 | ✅ 已完成 | 标注数据集 (7个维度) |
| **Phase 2** | 训练分类器 | ✅ 已完成 | 7个 PyTorch 分类器 |
| **Phase 3** | CoreML 导出 | ✅ 已完成 | 7个 .mlpackage 文件 |
| **Phase 3.5** | iOS 模型集成 & 调试 | ✅ 已完成 | InsightModelManager, 调试界面 |
| **Phase 4.1-4.2** | Core Data 模型扩展 | ✅ 已完成 | GoalEntry 7维度字段, AnalysisResult 扩展 |
| **Phase 4.3** | GoalDataAggregator | ✅ 已完成 | 聚合查询服务 |
| **Phase 5** | 洞察引擎 | ✅ 已完成 | InsightEngine 扩展, 模板系统, 单元测试 |
| **Phase 6** | UI 开发 | ⏳ 进行中 | 统计Tab, 洞察卡片, 趋势图表 |
| **Phase 7** | 测试 & 优化 | ⏳ 进行中 | 单元测试, 性能报告 |

---

## 已完成工作详情

### Phase 1-3: 模型训练与导出 ✅

- [x] 7 个分类维度的数据准备与标注
- [x] 训练 7 个分类器（Topic, Sentiment, Urgency, TimeFrame, ActionType, Difficulty, Specificity）
- [x] 导出为 CoreML 可更新模型 (.mlpackage)

### Phase 3.5: iOS 模型集成 & 调试界面 ✅

这是根据实际开发需求新增的阶段，用于验证模型在 iOS 端的正确性。

#### InsightModelManager.swift - 核心模型管理服务 ✅
- [x] 加载 BertFeatureExtractor 和 7 个分类器
- [x] 集成 `swift-transformers` 库进行 BERT 分词
- [x] 实现 `analyze(text:)` 返回所有维度分类结果
- [x] 实现并行推理 (TaskGroup)
- [x] 实现 `reloadModelsAfterTraining()` - 训练后重新加载更新的模型

#### InsightUpdateManager.swift - 设备端训练管理 ✅
- [x] 实现 `InsightTrainingSample` 样本结构
- [x] 实现 `addTrainingSample()` - 添加训练样本
- [x] 实现 `updateAllModels()` - 批量更新所有分类器
- [x] 实现 `updateModel()` - 单个模型训练 (MLUpdateTask)
- [x] 模型路径一致性修复 (`{Name}Classifier_Updatable_Updated.mlmodelc`)

#### InsightClassifierLabels.swift - 标签定义 ✅
- [x] 定义 7 个维度的标签枚举
- [x] 提供 displayName、icon 等辅助方法

#### InsightModelDebugTab.swift - 调试界面 ✅
- [x] 模型加载状态显示 (x/7 分类器)
- [x] 文本分析测试
- [x] 7 维度分类结果展示
- [x] 用户纠正功能 (Picker 选择正确标签)
- [x] 训练功能 (添加样本 → 批量训练)
- [x] 并发保护 (训练时禁用分析按钮)

#### Insight_Model_Walkthrough.md - 技术文档 ✅
- [x] 系统架构说明
- [x] 模型详情
- [x] 设备端训练流程
- [x] 常见问题排查

---

## Phase 4: iOS 数据层 (进行中)

### 4.1 Core Data 模型扩展 ✅

当前项目已有 `GoalEntry` 实体，已扩展以支持 7 维度分类结果。

- [x] 扩展 `GoalEntry.swift` 添加新字段
  ```
  新增字段:
  - urgency: String?
  - urgencyConfidence: Double
  - timeFrame: String?
  - timeFrameConfidence: Double
  - actionType: String?
  - actionTypeConfidence: Double
  - difficulty: String?
  - difficultyConfidence: Double
  - specificity: String?
  - specificityConfidence: Double
  - embedding: Data? (可选，用于相似度搜索)
  - 用户纠正字段: urgencyUserCorrected, timeFrameUserCorrected, etc.
  ```

- [x] 更新 `CoreDataModelBuilder.swift` 添加新属性定义

- [x] 添加 `effective*` 访问器（优先返回用户纠正值）

- [x] 添加 `hasUserCorrections` 便捷属性

### 4.2 AnalysisResult 结构扩展 ✅

- [x] 修改 `CoreMLAnalysisService.swift` 中的 `AnalysisResult`
  - [x] 添加 5 个新维度的分类结果字段
  - [x] 添加 embedding 字段
  - [x] 添加 `init(from: InsightAnalysisResult)` 转换初始化器
  - [x] 保持向后兼容（新字段有默认值）

### 4.3 GoalDataAggregator 服务 ✅

- [x] 创建 `GoalDataAggregator.swift`
- [x] 实现 `getAggregatedStats(from:to:)` - 获取时间范围内统计
- [x] 实现 `getTopicDistribution(period:)` - 获取主题分布
- [x] 实现 `getSentimentTrend(days:)` - 获取情感趋势
- [x] 实现 `getCompletionRate(groupBy:period:)` - 获取完成率
- [x] 实现 `getSimilarGoals(embedding:limit:)` - 相似目标搜索 (使用 Accelerate 框架加速余弦相似度计算)
- [x] 创建 `InsightStatsTab.swift` - 统计展示界面
- [x] 集成到 RootView Tab 2 (替换原 "测试" Tab)

### 4.4 数据聚合定时任务 (可选)

- [ ] 使用 BackgroundTasks API 实现定时聚合
- [ ] 每日计算日统计
- [ ] 每周计算周统计

---

## Phase 5: 洞察生成引擎 ✅

### 5.1 洞察类型定义

- [x] 扩展 `Insight` 枚举，覆盖 balance / volumeTrend / weekdayPattern / encouragement / comparison / recurringGoal / achievability

### 5.2 分析器实现 (已集成在 InsightEngine)

- [x] `analyzeWeekdayPattern(entries:)` - 识别周期性模式
- [x] `findRecurringGoals(entries:)` - 识别高频目标
- [x] `analyzeBalance(entries:)` - 目标类型平衡建议
- [x] `analyzeSentimentTrend(entries:)` - 情感趋势
- [x] `analyzeVolumeTrend(entries:)` - 目标数量趋势
- [x] `analyzeAchievability(entries:)` - 可达成性评估

### 5.3 InsightGenerator 综合生成器

- [x] `InsightEngine.generateInsights(for:topK:)` 统一生成洞察
- [x] 洞察优先级排序
- [x] Top-K 筛选

### 5.4 洞察模板系统

- [x] 创建 `InsightTemplate.swift`
- [x] 预定义模板库（11 个模板）
- [x] 模板变量填充

---

## Phase 6: UI 开发 (进行中)

### 6.1 洞察统计仪表盘

- [x] 创建 `InsightStatsTab.swift` (统计Tab)
- [x] 快速统计卡片
- [x] 周期选择器 (日/周/月)

### 6.2 图表组件

- [x] 情感趋势折线图（Charts）
- [x] 目标数量趋势图（Charts）
- [x] 主题/星期分布展示

### 6.3 InsightCard 组件

- [x] 创建 `InsightCardView.swift`
- [x] 图标 + 标题
- [x] 描述文本

### 6.4 InsightViewModel

- [ ] 创建 `InsightViewModel.swift`
- [ ] 管理洞察数据加载
- [ ] 管理周期切换
- [ ] 管理趋势数据

### 6.5 集成到主界面

- [x] RootView 集成 InsightStatsTab
- [x] InsightsView 列表展示洞察卡片

---

## Phase 7: 测试 & 优化 (进行中)

### 7.1 单元测试

- [ ] GoalDataAggregator 测试
- [x] InsightEngineTests
- [ ] Balance/Trend/Achievability 相关测试

### 7.2 集成测试

- [ ] 端到端分类流程测试
- [ ] 洞察生成流程测试
- [ ] 数据持久化测试

### 7.3 性能测试

- [ ] 7 个分类器并行推理延迟 (目标 < 50ms)
- [ ] 洞察生成延迟 (目标 < 100ms)
- [ ] 内存占用测试
- [ ] 电池消耗测试

### 7.4 用户体验测试

- [ ] 洞察文案优化
- [ ] UI 交互流畅度测试

---

## 关键指标

| 指标 | 目标值 | 当前状态 |
|-----|-------|---------|
| 推理延迟 (7个分类器) | < 50ms | ⏳ 待测试 |
| 洞察生成延迟 | < 100ms | ⏳ 待测试 |
| 模型总体积增量 | < 100KB | ✅ ~72KB |
| 新分类器准确率 | > 80% | ✅ 78%~99% |
| 端侧更新收敛速度 | < 10 epochs | ⏳ 待测试 |

---

## 已完成交付物

### iOS 集成文件 (Phase 3.5 - 6)

```
ios/MorningGoal/MorningGoal/
├── Services/
│   ├── InsightModelManager.swift      # 模型加载与推理
│   ├── InsightUpdateManager.swift     # 设备端训练
│   ├── GoalDataAggregator.swift       # 数据聚合查询服务
│   ├── InsightEngine.swift            # 洞察生成引擎
│   └── InsightTemplate.swift          # 洞察模板系统
├── Views/
│   ├── InsightStatsTab.swift          # 洞察统计Tab (Tab 2)
│   ├── InsightCardView.swift          # 洞察卡片与列表
│   └── InsightModelDebugTab.swift     # 调试界面
├── Models/
│   └── InsightClassifierLabels.swift  # 标签定义
├── coreml/
│   ├── BertFeatureExtractor.mlpackage
│   ├── TopicClassifier_Updatable.mlpackage
│   ├── SentimentClassifier_Updatable.mlpackage
│   ├── UrgencyClassifier_Updatable.mlpackage
│   ├── TimeFrameClassifier_Updatable.mlpackage
│   ├── ActionTypeClassifier_Updatable.mlpackage
│   ├── DifficultyClassifier_Updatable.mlpackage
│   └── SpecificityClassifier_Updatable.mlpackage
└── docs/
    └── Insight_Model_Walkthrough.md   # 技术文档
```

### 模型训练文件 (Phase 1-3)

```
models/exported_coreml/
├── BertFeatureExtractor.mlpackage     # 共享特征提取器 (静态)
├── TopicClassifier_Updatable.mlpackage     # 主题分类 (16类)
├── SentimentClassifier_Updatable.mlpackage # 情感分类 (3类)
├── UrgencyClassifier_Updatable.mlpackage   # 紧急度 (3类)
├── TimeFrameClassifier_Updatable.mlpackage # 时间范围 (4类)
├── ActionTypeClassifier_Updatable.mlpackage# 行动类型 (5类)
├── DifficultyClassifier_Updatable.mlpackage# 难度 (3类)
└── SpecificityClassifier_Updatable.mlpackage# 具体程度 (3类)

src/training/
└── train_insight_classifiers.py       # 训练脚本

src/export/
└── export_insight_classifiers.py      # 导出脚本
```

### 模型性能

| 分类器 | 类别数 | 测试准确率 | 测试 F1 |
|--------|--------|-----------|---------|
| Topic | 16 | - | - |
| Sentiment | 3 | - | - |
| Urgency | 3 | 94.6% | 66.7% |
| TimeFrame | 4 | 99.9% | 99.2% |
| ActionType | 5 | 98.3% | 97.2% |
| Difficulty | 3 | 78.3% | 66.5% |
| Specificity | 3 | 88.2% | 62.2% |

---

## 下一步行动

**建议按以下顺序推进：**

1. **Phase 4.4**: 后台聚合定时任务（可选）
2. **Phase 6**: 完成 InsightViewModel 与仪表盘补齐
3. **Phase 7**: 完成测试与性能优化

---

## 后续迭代 (V1.1+)

| 版本 | 功能 | 优先级 |
|-----|------|-------|
| V1.1 | 集成 Apple Intelligence (iOS 18.1+) | P1 |
| V1.2 | 云端 LLM 周报功能 | P2 |
| V2.0 | 完整混合架构 (规则 + 端侧 + 云端) | P3 |
