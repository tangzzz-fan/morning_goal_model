# 用户数据洞察系统 - 剩余任务清单

> 基于 `User_Insight_System_Design.md` 设计文档
> 更新时间: 2026-01-24

---

## 进度总览

| 阶段 | 内容 | 状态 | 交付物 |
|-----|------|------|--------|
| **Phase 1** | 数据准备 & 标注 | ✅ 已完成 | 标注数据集 (5个新维度) |
| **Phase 2** | 训练新分类器 | ✅ 已完成 | 5个 PyTorch 分类器 |
| **Phase 3** | CoreML 导出 | ✅ 已完成 | 5个 .mlpackage 文件 |
| **Phase 4** | iOS 数据层 | ⏳ 待开始 | Core Data 模型, 聚合服务 |
| **Phase 5** | 洞察引擎 | ⏳ 待开始 | 规则引擎, 模板系统 |
| **Phase 6** | UI 开发 | ⏳ 待开始 | 洞察仪表盘, 趋势图表 |
| **Phase 7** | 测试 & 优化 | ⏳ 待开始 | 性能报告, 用户测试 |

---

## Phase 4: iOS 数据层 (Core Data / SQLite)

### 4.1 GoalRecord 数据模型
- [ ] 创建 `GoalRecord` Core Data 实体
  - `id: UUID` - 主键
  - `text: String` - 原始目标文本
  - `timestamp: Date` - 创建时间
  - `embedding: Data` - 512维向量 (可选)
  - 分类结果字段 (7个维度 + 置信度)
  - 用户反馈字段 (完成状态、纠正标签)

### 4.2 数据库 Schema
- [ ] 创建 `goal_records` 表
- [ ] 创建 `aggregated_stats` 缓存表
- [ ] 添加索引 (timestamp, topic, action_type)

### 4.3 GoalDataAggregator 服务
- [ ] 实现 `getAggregatedStats(from:to:)` - 获取时间范围内统计
- [ ] 实现 `getTopicDistribution(period:)` - 获取主题分布
- [ ] 实现 `getSentimentTrend(days:)` - 获取情感趋势
- [ ] 实现 `getCompletionRate(groupBy:period:)` - 获取完成率
- [ ] 实现 `getSimilarGoals(embedding:limit:)` - 基于 embedding 的相似目标搜索

### 4.4 数据聚合定时任务
- [ ] 每日凌晨计算日统计
- [ ] 每周日计算周统计
- [ ] 每月1日计算月统计

---

## Phase 5: 洞察生成引擎 (规则引擎)

### 5.1 InsightModelManager 扩展
- [ ] 加载所有 7 个分类器 (.mlpackage)
- [ ] 实现 `analyze(text:)` - 返回所有维度分类结果
- [ ] 实现并行推理 (DispatchGroup)
- [ ] 实现 `update(dimension:embedding:correctLabel:)` - 端侧更新

### 5.2 洞察类型定义
- [ ] 定义 `InsightType` 枚举 (pattern, balance, trend, achievability, completion, comparison, encouragement)
- [ ] 定义 `Insight` 结构体

### 5.3 PatternAnalyzer - 模式识别
- [ ] 实现 `analyzeWeekdayPatterns(records:)` - 识别周期性模式
- [ ] 实现 `findRecurringGoals(records:similarityThreshold:)` - 识别高频目标

### 5.4 BalanceAdvisor - 平衡建议
- [ ] 实现 `checkBalance(stats:)` - 检测目标类型不平衡
- [ ] 实现 `suggestComplementary(type:)` - 建议互补类型

### 5.5 TrendAnalyzer - 趋势分析
- [ ] 实现 `analyzeSentimentTrend(dailyStats:)` - 分析情感趋势
- [ ] 实现 `analyzeVolumeTrend(dailyStats:)` - 分析目标数量趋势

### 5.6 AchievabilityPredictor - 可达成性评估
- [ ] 实现 `evaluateGoal(goal:historicalRecords:)` - 评估当前目标
- [ ] 基于难度 + 具体程度评估
- [ ] 基于历史完成率评估
- [ ] 基于当日目标数量评估

### 5.7 InsightGenerator - 综合洞察生成器
- [ ] 实现 `generateInsights(for:stats:historicalRecords:)` - 综合生成洞察
- [ ] 实现洞察优先级排序
- [ ] 实现 Top-K 洞察筛选

### 5.8 洞察模板系统
- [ ] 定义 `InsightTemplate` 结构体
- [ ] 创建预定义模板库 (10-20个模板)
- [ ] 实现模板变量填充逻辑

---

## Phase 6: UI 开发

### 6.1 GoalInputView 增强
- [ ] 显示当前目标的多维度分类结果
- [ ] 实时显示可达成性评估
- [ ] 支持用户纠正分类标签

### 6.2 InsightDashboardView - 洞察仪表盘
- [ ] 快速统计卡片 (QuickStatsCard)
- [ ] 洞察列表 (InsightCard)
- [ ] 周期选择器 (日/周/月)

### 6.3 TrendChartView - 趋势图表
- [ ] 情感趋势折线图 (SwiftUI Charts)
- [ ] 目标类型分布饼图
- [ ] 完成率柱状图

### 6.4 InsightCard 组件
- [ ] 图标 + 标题
- [ ] 描述文本
- [ ] 行动建议 (可选)
- [ ] 交互 (展开详情、跳转相关目标)

### 6.5 InsightViewModel
- [ ] 管理洞察数据加载
- [ ] 管理周期切换
- [ ] 管理趋势数据

---

## Phase 7: 测试 & 优化

### 7.1 单元测试
- [ ] GoalDataAggregator 测试
- [ ] PatternAnalyzer 测试
- [ ] BalanceAdvisor 测试
- [ ] TrendAnalyzer 测试
- [ ] AchievabilityPredictor 测试

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
- [ ] 洞察文案 A/B 测试
- [ ] UI 交互流畅度测试
- [ ] 用户反馈收集

### 7.5 模型优化 (可选)
- [ ] 分类器准确率提升 (目标 > 80%)
- [ ] 端侧更新收敛速度优化 (目标 < 10 epochs)

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

### Phase 1-3 完成的文件

```
models/exported_coreml/
├── BertFeatureExtractor.mlpackage          # 共享特征提取器 (静态, 56KB)
├── TopicClassifier_Updatable.mlpackage     # 主题分类 (16类, 40KB)
├── SentimentClassifier_Updatable.mlpackage # 情感分类 (3类, 12KB)
├── UrgencyClassifier_Updatable.mlpackage   # 紧急度 (3类, 12KB) [新增]
├── TimeframeClassifier_Updatable.mlpackage # 时间范围 (4类, 16KB) [新增]
├── ActiontypeClassifier_Updatable.mlpackage# 行动类型 (6类, 20KB) [新增]
├── DifficultyClassifier_Updatable.mlpackage# 难度 (3类, 12KB) [新增]
└── SpecificityClassifier_Updatable.mlpackage# 具体程度 (3类, 12KB) [新增]

src/training/
└── train_insight_classifiers.py            # 训练脚本

src/export/
└── export_insight_classifiers.py           # 导出脚本

models/trained/insight_classifiers/
├── urgency_classifier.pt
├── timeFrame_classifier.pt
├── actionType_classifier.pt
├── difficulty_classifier.pt
├── specificity_classifier.pt
└── training_metrics.json
```

### 模型性能

| 分类器 | 类别数 | 测试准确率 | 测试 F1 |
|--------|--------|-----------|---------|
| Urgency | 3 | 94.6% | 66.7% |
| TimeFrame | 4 | 99.9% | 99.2% |
| ActionType | 6 | 98.3% | 97.2% |
| Difficulty | 3 | 78.3% | 66.5% |
| Specificity | 3 | 88.2% | 62.2% |
