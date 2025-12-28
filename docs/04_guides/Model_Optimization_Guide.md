# 模型调优实战指南 (Model Optimization Guide)

## 1. 目标
本文档旨在指导开发者如何对 Morning Goal 项目中的模型进行体积与质量的优化。我们的目标是在保持高精度 (Accuracy > 97%) 的前提下，尽可能减小模型体积，以适配移动端设备。

## 2. 核心优化工作流
我们的优化流水线遵循以下三个主要阶段：

```
[教师模型微调] -> [知识蒸馏 Knowledge Distillation] -> [量化 Quantization]
```

### 阶段一：教师模型微调 (Teacher Fine-tuning)
首先，我们需要训练一个高精度的“教师”模型（通常是 `bert-base-chinese`）。
*   **目的**: 获得特定任务（意图分类/情感分析）的最佳性能上限。
*   **脚本**: `src/training/finetune_bert.py`
*   **输入**: 原始 BERT 模型 + 标注数据。
*   **输出**: `models/trained/teacher_bert_best.pt`

### 阶段二：知识蒸馏 (Knowledge Distillation)
这是模型轻量化的**最关键步骤**。我们将教师模型的知识迁移到一个更小的“学生”模型（如 `MobileBERT` 或 4层 `RoBERTa`）。
*   **目的**: 在大幅减少参数量（减少 75%+）的同时，保留 99% 的精度。
*   **脚本**: `src/training/distill_student.py`
*   **关键参数**:
    *   `--teacher_model`: 指向阶段一训练好的教师模型。
    *   `--student_arch`: 选择学生模型架构 (推荐 `uer/chinese_roberta_L-4_H-512`)。
    *   `--temperature`: 蒸馏温度 (建议 T=3.0 ~ 5.0)。
    *   `--alpha`: 软硬标签的 loss 权重比例 (建议 0.5)。
*   **性能参考**:
    | 模型 | 参数量 | F1 Score | 体积 (FP32) |
    | :--- | :--- | :--- | :--- |
    | Teacher (BERT-Base) | 110M | 0.976 | 420MB |
    | Student (L-4_H-512) | 25M | 0.978 | 100MB |

### 阶段三：模型量化 (Quantization)
最后，我们将模型权重从 FP32 转换为 FP16 或 INT8。
*   **目的**: 利用移动端硬件加速 (NPU/ANE)，减少体积，提升推理速度。
*   **操作位置**: 在导出 Core ML 时进行 (`src/export/export_coreml.py`)。
*   **方案选择**:
    *   **FP16**: 推荐默认方案。精度几乎无损，体积减半。适用于 ANE。
    *   **INT8**: 极致压缩方案。体积为 FP32 的 1/4，但可能会有 <1% 的精度损失。需进行评估。
    *   **[不推荐] 剪枝 (Pruning)**: 对于 MobileBERT 这类已经紧凑设计的架构，进一步的非结构化剪枝往往弊大于利。

## 3. 如何操作：调优 Checklist

如果你需要优化模型，请按以下步骤操作：

- [ ] **数据准备**: 确保 `data/processed/` 下的数据是最新的且质量过关。
- [ ] **基准测试**: 先跑一遍 `finetune_bert.py`，记录下教师模型的 F1 值作为基准。
- [ ] **执行蒸馏**: 运行 `distill_student.py`。
    - *尝试*: 如果精度不够，尝试增加 Epochs 或微调 Temperature 参数。
- [ ] **转换与量化**: 运行 `export_coreml.py` 并开启 `--quantize fp16`。
- [ ] **端侧验证**: 将生成的 `.mlpackage` 放入 iOS 项目，使用真实 iPhone 进行推理耗时测试。

## 4. 常见问题 (FAQ)

**Q: 蒸馏后的模型精度不升反降？**
A: 检查教师模型是否本身就欠拟合？或者蒸馏时的 Temperature 设置是否不合理（过高会导致分布过于平滑）。

**Q: INT8 量化后某些分类错误率飙升？**
A: 这是一个常见现象。建议回退到 FP16，或者尝试使用“量化感知训练 (QAT)”，但这需要修改训练代码。对于本项目，推荐坚持使用 **FP16** 以保证稳定性。
