BERT 技术栈：移动端适配与场景落地总览

目标
- 以 `bert-base-chinese` 为教师模型，经过剪枝、蒸馏与量化优化，得到适配移动端的轻量学生模型（如 MobileBERT/TinyBERT 风格）。
- 使用 ONNX 或 Core ML 将模型部署到 iOS/Android 端，形成可更新的基座 + updatable 架构。
- 提供两个端侧场景的落地方案与步骤：文本洞察与端侧自训练 + 云 API 结合。

文档导航
- [01 模型剪枝/蒸馏/优化流程](./01_模型剪枝_蒸馏_优化流程.md)
- [02 ONNX / Core ML 转换与脚本](./02_ONNX_CoreML_转换与脚本.md)
- [03 Core ML Pipeline 设计与优化](./03_CoreML_Pipeline_设计与优化.md)
- [04 移动端架构：基座 + Updatable](./04_移动端架构_基座与Updatable.md)
- [05 MobileBERT 与移动端适配实践](./05_MobileBERT_与移动端适配.md)
- [场景 01：移动端文本洞察](./场景_01_移动端文本洞察.md)
- [场景 02：端侧自训练 + 云 API 结合](./场景_02_端侧自训练_云API结合.md)

流程总览
```mermaid
flowchart LR
  A[任务与数据定义] --> B[教师模型: bert-base-chinese]
  B --> C[剪枝: 注意力头/层/FFN]
  C --> D[蒸馏: 任务/中间层/对齐损失]
  D --> E[优化: 量化/PTQ/QAT]
  E --> F[导出: ONNX / Core ML]
  F --> G[移动端打包: mlmodelc / .onnx]
  G --> H[架构: 基座 + Updatable]
  H --> I[端侧推理与场景落地]
```

选型建议
- iOS 首选 Core ML `mlprogram`（iOS 15+），可用 `compute_units=ALL` 与 `precision=FLOAT16`；Android 可用 ONNX Runtime Mobile/NCNN/MNN。
- 中文任务可用 `bert-base-chinese` 作为教师，学生建议 TinyBERT/MobileBERT 风格，序列长度控制在 128/256。
- 量化优先使用权重量化 + 激活半精度（FP16）；对延迟极端敏感时再评估 INT8 激活（需校准）。

验证与度量
- 指标：下游任务 F1/准确率/延迟（P50/P90）/峰值内存/包体大小。
- 基准设备：A14/A17/M1/M2；安卓侧 778G/8 Gen 系列。统一测试脚本与数据集，保证横向对比公平。