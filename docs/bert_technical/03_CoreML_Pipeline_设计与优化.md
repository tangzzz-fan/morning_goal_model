Core ML Pipeline 设计与优化

设计目的
- 将文本预处理（Tokenizer/Padding）与模型推理解耦，支持基座 + updatable 的热更新。
- 控制内存与延迟，保证稳定的用户体验与可观察性。

Pipeline 结构
```mermaid
flowchart LR
  A[原始文本] --> B[Tokenizer (Swift)]
  B --> C[Padding/截断]
  C --> D[MLModel 推理]
  D --> E[任务头: 分类/序列标注]
  E --> F[后处理: 标签/置信度/解释]
  F --> G[日志与上报]
```

输入/输出约定
- 输入：`input_ids:int32`、`attention_mask:int32`、`token_type_ids:int32`（可选）。
- 输出：`logits:float32/float16`；序列标注则输出 `token_logits`。

Swift 端调用示例（iOS）
```swift
import CoreML

let url = Bundle.main.url(forResource: "BERTClassifier", withExtension: "mlmodelc")!
let cfg = MLModelConfiguration()
cfg.computeUnits = .all
let model = try MLModel(contentsOf: url, configuration: cfg)

// 预先分配 MultiArray，避免重复分配
let ids = try MLMultiArray(shape: [1, seqLen] as [NSNumber], dataType: .int32)
let mask = try MLMultiArray(shape: [1, seqLen] as [NSNumber], dataType: .int32)

let inputs: [String: MLFeatureValue] = [
  "input_ids": MLFeatureValue(multiArray: ids),
  "attention_mask": MLFeatureValue(multiArray: mask)
]
let provider = try MLDictionaryFeatureProvider(dictionary: inputs)
let out = try model.prediction(from: provider)
```

优化策略
- 统一固定序列长度（如 128/256），预分配 `MLMultiArray` 缓冲。
- 使用 FP16 权重；评估 INT8 激活仅在严格延迟目标下使用。
- 预热（warmup）一到两次，避免首帧抖动。
- 任务头分离：通用 Encoder + 多任务头（分类/序列标注）独立更新。

可更新设计（Updatable）
- 将通用 Encoder 作为基座，任务头以独立 `mlmodelc` 包分发与热更新。
- 版本管理：`encoder@vX` + `task_head@vY`，维护兼容矩阵与回滚策略。
- 安全：包签名校验、SHA256 哈希、沙盒内加载，异常时回退基座。