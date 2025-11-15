# iOS 端侧可更新模型方案

## 现状评估
- 直接由 ONNX → CoreML 转换的 Transformer 分类模型默认不包含训练配置，无法通过 `MLUpdateTask` 在 iOS 端侧执行参数更新。

## 可更新设计路径
- 方案 A：特征抽取 + 可更新分类器
  - 将 Transformer 作为特征抽取器（冻结，不可更新），输出特征向量
  - 在 CoreML 中构建小型可更新分类器（如 Logistic Regression/小型 NN），添加训练配置（损失/更新参数）
  - 组合为 Pipeline：`FeatureExtractor.mlmodel` + `UpdatableClassifier.mlmodel`
  - 在 iOS 使用 `MLUpdateTask` 对分类器进行增量更新
- 方案 B：仅开放末层线性权重为可更新
  - 转换时保留末层权重为可更新，添加训练配置（损失为交叉熵）
  - 更新范围最小，端侧训练成本低

## 实施要点
- CoreML 训练配置：
  - 在可更新模型中定义 loss 与 updateParameters
  - 准备带标签的 `MLBatchProvider` 作为增量数据
- iOS 代码：
  - 使用 `MLUpdateTask` 执行更新，监听 `MLUpdateProgressHandlers` 获取指标
  - 更新完成后替换或持久化新的 `mlmodelc`

## 下一步行动
- 输出特征抽取器与可更新分类器的转换脚本与示例 Pipeline 说明
- 在 macOS/iOS 环境验证更新流程，记录精度差异与端侧时延
- 制定增量数据采集与隐私策略（仅本地、差分隐私可选）