# 洞察模型系统 Walkthrough

本文档详细说明了 MorningGoal 应用中洞察分类系统的架构设计、实现细节和使用指南。

---

## 目录

1. [系统概述](#1-系统概述)
2. [架构设计](#2-架构设计)
3. [模型说明](#3-模型说明)
4. [代码结构](#4-代码结构)
5. [设备端训练](#5-设备端训练)
6. [使用指南](#6-使用指南)
7. [最佳实践](#7-最佳实践)
8. [常见问题排查](#8-常见问题排查)

---

## 1. 系统概述

### 1.1 功能目标

洞察分类系统旨在对用户输入的目标文本进行多维度分析，提供以下 7 个维度的分类：

| 维度 | 英文名 | 分类数 | 标签 |
|-----|--------|-------|------|
| 主题分类 | topic | 16 | 工作、健康、家庭、个人发展、理财、社交、家务、学习、睡眠、饮食、心态、娱乐、出行、职业发展、沟通、育儿 |
| 情感倾向 | sentiment | 3 | 消极、中性、积极 |
| 紧急度 | urgency | 3 | 低、中、高 |
| 时间范围 | timeFrame | 4 | 今天、本周、本月、长期 |
| 行动类型 | actionType | 5 | 学习、运动、工作、生活、社交 |
| 难度 | difficulty | 3 | 简单、中等、困难 |
| 具体程度 | specificity | 3 | 模糊、一般、具体 |

### 1.2 核心特性

- **实时分析**：毫秒级推理速度
- **设备端训练**：支持用户纠正后的模型更新
- **隐私保护**：所有数据和训练都在设备本地完成
- **持续学习**：模型随用户反馈不断改进

---

## 2. 架构设计

### 2.1 Fan-out 架构

系统采用 **Fan-out (扇出) 架构**，核心思想是：

```
                    ┌─────────────────────┐
                    │  BertFeatureExtractor │ (静态，不可更新)
                    │  bert-base-chinese   │
                    └──────────┬──────────┘
                               │
                          embedding
                          (768维向量)
                               │
           ┌───────────────────┼───────────────────┐
           │                   │                   │
           ▼                   ▼                   ▼
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │   Topic     │     │  Sentiment  │     │   Urgency   │
    │ Classifier  │     │ Classifier  │     │ Classifier  │
    │  (可更新)   │     │  (可更新)   │     │  (可更新)   │
    └─────────────┘     └─────────────┘     └─────────────┘
           │                   │                   │
           ▼                   ▼                   ▼
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │  TimeFrame  │     │ ActionType  │     │ Difficulty  │
    │ Classifier  │     │ Classifier  │     │ Classifier  │
    │  (可更新)   │     │  (可更新)   │     │  (可更新)   │
    └─────────────┘     └─────────────┘     └─────────────┘
           │
           ▼
    ┌─────────────┐
    │ Specificity │
    │ Classifier  │
    │  (可更新)   │
    └─────────────┘
```

### 2.2 架构优势

| 优势 | 说明 |
|-----|------|
| **计算效率** | 特征提取只做一次，7 个分类器共享嵌入向量 |
| **独立更新** | 每个分类器可以独立训练，不影响其他维度 |
| **模块化** | 可以轻松添加或移除分类维度 |
| **内存优化** | BERT 模型只加载一份 |

### 2.3 模型类型

| 模型 | 类型 | 格式 | 可更新 |
|-----|------|------|--------|
| BertFeatureExtractor | MLProgram | .mlpackage | ❌ |
| TopicClassifier | NeuralNetwork | .mlpackage | ✅ |
| SentimentClassifier | NeuralNetwork | .mlpackage | ✅ |
| UrgencyClassifier | NeuralNetwork | .mlpackage | ✅ |
| TimeframeClassifier | NeuralNetwork | .mlpackage | ✅ |
| ActiontypeClassifier | NeuralNetwork | .mlpackage | ✅ |
| DifficultyClassifier | NeuralNetwork | .mlpackage | ✅ |
| SpecificityClassifier | NeuralNetwork | .mlpackage | ✅ |

---

## 3. 模型说明

### 3.1 特征提取器

**BertFeatureExtractor** 基于 `bert-base-chinese`：

- **输入**：
  - `input_ids`: [1, 128] Int32
  - `attention_mask`: [1, 128] Int32
  
- **输出**：
  - `embedding`: [1, 768] Float32

- **Tokenizer**：使用 `swift-transformers` 库的 `AutoTokenizer`

### 3.2 分类器

每个分类器都是一个简单的 2 层全连接网络：

```
embedding (768) → Dense(128) → ReLU → Dense(num_classes) → Softmax
```

**输入**：
- `embedding`: [1, 768] Float32

**输出**：
- `{dimension}_probs`: [1, num_classes] Float32

### 3.3 模型文件位置

```
ios/MorningGoal/MorningGoal/coreml/
├── BertFeatureExtractor.mlpackage
├── TopicClassifier_Updatable.mlpackage
├── SentimentClassifier_Updatable.mlpackage
├── UrgencyClassifier_Updatable.mlpackage
├── TimeframeClassifier_Updatable.mlpackage
├── ActiontypeClassifier_Updatable.mlpackage
├── DifficultyClassifier_Updatable.mlpackage
└── SpecificityClassifier_Updatable.mlpackage
```

---

## 4. 代码结构

### 4.1 核心文件

```
ios/MorningGoal/MorningGoal/
├── Models/
│   └── InsightClassifierLabels.swift    # 标签定义和辅助方法
├── Services/
│   ├── InsightModelManager.swift        # 模型加载和推理
│   └── InsightUpdateManager.swift       # 设备端训练管理
└── Views/
    └── InsightModelDebugTab.swift       # 调试界面
```

### 4.2 InsightModelManager

**职责**：管理模型加载和推理

```swift
@MainActor
final class InsightModelManager: ObservableObject {
    // 主要方法
    func loadModels() async                           // 加载所有模型
    func analyze(text: String) async throws -> InsightAnalysisResult  // 分析文本
    func extractEmbedding(from text: String) throws -> MLMultiArray   // 提取嵌入
    func getModelVersions() -> [String: String]       // 获取模型版本
    func resetAllClassifiers() throws                 // 重置模型（删除更新版本）
    func reloadModelsAfterTraining() async            // 训练后重新加载模型
}
```

**关键属性**：
- `isInitialized`: 模型是否就绪
- `statusMessage`: 加载状态消息
- `loadedClassifiers`: 已加载的分类器列表

**重要方法说明**：
- `resetAllClassifiers()`: 删除 App Support 中的更新版本，恢复到 Bundle 中的原始模型
- `reloadModelsAfterTraining()`: 训练完成后调用，重新加载更新后的模型（不删除文件）

### 4.3 InsightUpdateManager

**职责**：管理训练样本和设备端更新

```swift
@MainActor
final class InsightUpdateManager: ObservableObject {
    // 主要方法
    func addTrainingSample(...)           // 添加训练样本
    func updateAllModels() async throws   // 更新所有模型
    func clearSamples()                   // 清空样本
    
    // 属性
    var pendingSampleCount: Int           // 待训练样本数
    var isTraining: Bool                  // 是否正在训练
    var trainingProgress: Float           // 训练进度
}
```

### 4.4 InsightClassifierLabels

**职责**：定义标签和提供辅助方法

```swift
enum InsightLabels {
    enum Topic: String, CaseIterable { ... }
    enum Sentiment: String, CaseIterable { ... }
    enum Urgency: String, CaseIterable { ... }
    // ...
    
    static func dimensionDisplayName(_ dimension: String) -> String
    static func dimensionIcon(_ dimension: String) -> String
}
```

---

## 5. 设备端训练

### 5.1 训练流程

```
┌─────────────────────────────────────────────────────────────────┐
│  用户输入目标文本                                                │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  模型预测 7 个维度                                               │
└──────────────────────────────┬──────────────────────────────────┘
                               │
              ┌────────────────┴────────────────┐
              │                                 │
              ▼                                 ▼
┌─────────────────────┐               ┌─────────────────────┐
│  预测正确？          │               │  预测错误？          │
│  → 继续使用          │               │  → 用户纠正          │
└─────────────────────┘               └──────────┬──────────┘
                                                 │
                                                 ▼
                                    ┌─────────────────────┐
                                    │  选择正确的 7 个标签  │
                                    └──────────┬──────────┘
                                               │
                                               ▼
                                    ┌─────────────────────┐
                                    │  添加到训练集        │
                                    │  (嵌入 + 7 标签)     │
                                    └──────────┬──────────┘
                                               │
                                               ▼
                                    ┌─────────────────────┐
                                    │  累积多个样本后      │
                                    │  点击"训练模型"     │
                                    └──────────┬──────────┘
                                               │
                                               ▼
                                    ┌─────────────────────┐
                                    │  MLUpdateTask       │
                                    │  更新 7 个分类器     │
                                    └──────────┬──────────┘
                                               │
                                               ▼
                                    ┌─────────────────────┐
                                    │  保存到 App Support  │
                                    │  重新加载模型        │
                                    └─────────────────────┘
```

### 5.2 MLUpdateTask

CoreML 的 `MLUpdateTask` 用于设备端模型更新：

```swift
let updateTask = try MLUpdateTask(
    forModelAt: modelURL,           // 模型路径
    trainingData: batchProvider,    // 训练数据
    configuration: nil,
    progressHandlers: MLUpdateProgressHandlers(
        forEvents: [.epochEnd],
        progressHandler: { context in
            // 更新进度
        },
        completionHandler: { context in
            // 保存更新后的模型
            try context.model.write(to: modelURL)
        }
    )
)
updateTask.resume()
```

### 5.3 训练数据格式

```swift
struct InsightTrainingSample {
    let embedding: MLMultiArray     // [1, 768] 嵌入向量
    let text: String                // 原始文本
    let topicLabel: Int             // 0-15
    let sentimentLabel: Int         // 0-2
    let urgencyLabel: Int           // 0-2
    let timeframeLabel: Int         // 0-3
    let actiontypeLabel: Int        // 0-4
    let difficultyLabel: Int        // 0-2
    let specificityLabel: Int       // 0-2
    let timestamp: Date
}
```

### 5.4 模型存储位置

| 类型 | 位置 | 文件名格式 | 说明 |
|-----|------|-----------|------|
| 原始模型 | Bundle | `{Name}Classifier_Updatable.mlpackage` | 应用安装时自带 |
| 更新后模型 | App Support/Models/ | `{Name}Classifier_Updatable_Updated.mlmodelc` | 用户训练后保存 |

**路径一致性**（重要）：

`InsightUpdateManager` 和 `InsightModelManager` 必须使用相同的路径命名规则：

```swift
// 统一的更新模型路径格式
let updatedModelPath = modelsDirectory.appendingPathComponent(
    "\(bundleName)_Updated.mlmodelc"
)
// 例如: App Support/Models/TopicClassifier_Updatable_Updated.mlmodelc
```

**加载顺序**：
1. 首先检查 App Support 是否有 `{Name}Classifier_Updatable_Updated.mlmodelc`
2. 如果有，加载更新版本
3. 如果没有，从 Bundle 加载原始版本

### 5.5 训练后模型重新加载

训练完成后，必须调用 `reloadModelsAfterTraining()` 而不是 `resetAllClassifiers()`：

```swift
// ✅ 正确：重新加载更新后的模型
await modelManager.reloadModelsAfterTraining()

// ❌ 错误：这会删除更新的模型！
try modelManager.resetAllClassifiers()
```

### 5.6 并发保护

训练期间禁止执行推理操作，UI 层面实现互斥：

```swift
// 分析按钮在训练期间被禁用
Button(action: runAnalysis) { ... }
    .disabled(
        !modelManager.isInitialized || 
        inputText.isEmpty || 
        isAnalyzing || 
        updateManager.isTraining  // 训练时禁用
    )
```

**状态互斥**：
- 训练中 → 禁用分析按钮
- 分析中 → 禁用训练按钮
- 确保模型文件不会被同时读写

---

## 6. 使用指南

### 6.1 在 Xcode 中运行

1. 打开 `ios/MorningGoal/MorningGoal.xcodeproj`
2. 选择 iPhone 模拟器或真机
3. 点击 Run (⌘R)
4. 在应用中切换到"洞察"Tab

### 6.2 调试界面使用

#### 分析文本
1. 在输入框中输入目标文本
2. 或点击预设按钮快速填充
3. 点击"运行分析"
4. 查看 7 个维度的预测结果

#### 纠正预测
1. 分析后，点击"预测错误？点击纠正"
2. 在 7 个下拉菜单中选择正确的标签
3. 点击"添加到训练集"

#### 训练模型
1. 收集多个纠正样本（建议 5-10 个）
2. 点击"训练 7 个模型"
3. 等待训练完成
4. 新的预测将使用更新后的模型

#### 重置模型
- 点击"重置所有模型"可恢复到原始版本

### 6.3 集成到其他视图

```swift
// 1. 创建管理器
@StateObject private var modelManager = InsightModelManagerWrapper()

// 2. 加载模型
.onAppear {
    Task {
        await modelManager.loadModels()
    }
}

// 3. 分析文本
func analyzeGoal(_ text: String) async {
    do {
        let result = try await modelManager.analyze(text: text)
        // 使用结果
        print("主题: \(result.topic?.label ?? "未知")")
        print("情感: \(result.sentiment?.label ?? "未知")")
    } catch {
        print("分析失败: \(error)")
    }
}
```

---

## 7. 最佳实践

### 7.1 性能优化

| 优化点 | 说明 |
|-------|------|
| **延迟加载** | 模型在 `onAppear` 时异步加载 |
| **并行推理** | 7 个分类器使用 `TaskGroup` 并行执行 |
| **共享嵌入** | 特征提取只做一次，所有分类器共享 |
| **GPU 加速** | 配置 `computeUnits = .cpuAndNeuralEngine` |

### 7.2 训练建议

| 建议 | 说明 |
|-----|------|
| **样本数量** | 每次训练建议 5-20 个样本 |
| **样本质量** | 确保纠正标签准确 |
| **定期训练** | 累积一定样本后再训练，避免频繁更新 |
| **备份原始模型** | 可以通过"重置"恢复到初始状态 |

### 7.3 错误处理

```swift
do {
    let result = try await modelManager.analyze(text: text)
} catch AnalysisError.modelNotLoaded {
    // 模型未加载
    await modelManager.loadModels()
} catch {
    // 其他错误
    print("错误: \(error.localizedDescription)")
}
```

**训练错误处理**：

```swift
do {
    try await updateManager.updateAllModels()
    await modelManager.reloadModelsAfterTraining()
} catch InsightUpdateError.noSamples {
    print("没有训练样本")
} catch InsightUpdateError.modelNotFound(let name) {
    print("找不到模型: \(name)")
} catch InsightUpdateError.trainingFailed(let msg) {
    print("训练失败: \(msg)")
}
```

### 7.4 内存管理

- 模型在 `InsightModelManager` 中作为属性持有
- 使用 `@MainActor` 确保线程安全
- 推理时使用 `async/await` 避免阻塞主线程

---

## 附录

### A. 依赖库

| 库 | 版本 | 用途 |
|---|------|-----|
| swift-transformers | 1.1.6 | Tokenizer |
| CoreML | iOS 17+ | 模型推理和更新 |

### B. 相关文档

- [User_Insight_System_Design.md](../../docs/User_Insight_System_Design.md) - 系统设计文档
- [iOS_IMPLEMENT_PLAN.md](iOS_IMPLEMENT_PLAN.md) - iOS 实现计划

### C. 更新历史

| 日期 | 变更 |
|-----|------|
| 2026-01-24 | 初始版本，7 维度分类器 + 设备端训练 |
| 2026-01-24 | 修复模型路径一致性问题（UpdateManager 和 ModelManager 使用相同路径） |
| 2026-01-24 | 修复训练后重新加载逻辑（使用 reloadModelsAfterTraining 替代 resetAllClassifiers） |
| 2026-01-24 | 添加训练期间的并发保护（训练时禁用分析按钮） |

---

## 8. 常见问题排查

### 8.1 训练后模型没有更新

**症状**：训练完成后，预测结果没有变化

**原因**：训练后调用了 `resetAllClassifiers()` 而不是 `reloadModelsAfterTraining()`

**解决方案**：
```swift
// 训练后正确的调用
await modelManager.reloadModelsAfterTraining()
```

### 8.2 模型路径不匹配

**症状**：训练成功但加载失败

**原因**：`InsightUpdateManager` 和 `InsightModelManager` 使用了不同的路径格式

**检查点**：
- 两者都应使用 `{bundleName}_Updated.mlmodelc` 格式
- 确保 `modelsDirectory` 路径一致（App Support/Models/）

### 8.3 训练期间崩溃

**症状**：训练过程中应用崩溃

**可能原因**：
- 训练期间同时进行推理操作
- 内存不足

**解决方案**：
- 确保 UI 层面的并发保护生效
- 减少单次训练的样本数量
