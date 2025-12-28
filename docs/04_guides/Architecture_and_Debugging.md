# 架构设计与排查指南 (Architecture & Debugging)

## 1. 项目主体设计实现思路

本项目采用 **"端侧推理优先 (On-Device First)"** 和 **"基座+可更新 (Base + Updatable)"** 的混合架构设计。

### 核心架构图解
```
[ iOS App ]
    |
    |--- 输入 (用户文本)
    v
[ Core ML Pipeline ]
    |
    |--- 1. Tokenizer (分词)
    |       (将文本转为 input_ids)
    |
    |--- 2. Base Model (基座模型 - 冻结)
    |       Feature Extractor (MobileBERT / Student Model)
    |       (负责提取通用的语义特征向量)
    |       (权重不可变)
    |
    v
    |--- 3. Updatable Head (任务头 - 可训练)
    |       Classifier Layer
    |       (负责具体的分类任务，如意图识别)
    |       (支持在设备上通过 MLUpdateTask 进行增量更新)
    |
    v
[ 输出 (分类结果) ]
```

### 关键设计决策
1.  **隐私至上**: 所有推理和训练数据 100% 留在本地。
2.  **解耦设计**:
    *   **基座 (Encoder)**: 负责“懂语言”，由我们在云端训练好，能力强且通用。
    *   **任务头 (Head)**: 负责“懂用户”，轻量级，可以在端侧根据用户反馈实时微调。
3.  **计算流水线化**: 将分词 (Tokenization) 逻辑封装进 Core ML 模型中，使得 App 层面只需传入 String，极大简化了 iOS 开发者的调用成本。

## 2. 问题排查指南 (Troubleshooting)

在开发过程中遇到问题，请按照以下思路排查：

### 场景 A: 模型训练/导出失败
*   **症状**: Python 脚本报错，loss 不下降，或导出 coreml 时崩溃。
*   **排查步骤**:
    1.  **检查数据**: 80% 的问题源于数据。检查 `data/processed` 下的 CSV 文件格式是否正确？是否有空行？Labels 是否对应？
    2.  **检查 Tensor 形状**: 在 `src/model.py` 中打印 `input_ids` 和 `logits` 的 shape。确保维度匹配。
    3.  **CoreML 转换报错**: 通常是因为包含不支持的算子 (Op)。检查报错信息中的 Op 名称，并在 `export_coreml.py` 中尝试简化模型结构，或手动实现 Custom Layer。

### 场景 B: iOS 端推理结果不正确
*   **症状**: 模型能运行，但预测结果总是乱猜，或者始终输出同一个类别。
*   **排查步骤**:
    1.  **检查分词 (Tokenization)**: 这是最常见原因。确保 Python 端的 Tokenizer 和 iOS 端的分词逻辑（如果未封装）完全一致。如果有封装，检查 Vocabulary 是否遗漏。
    2.  **输入预处理**: 检查输入的 String 是否包含了意外的特殊字符？是否被截断了？
    3.  **Label 映射**: 检查 Python 训练时的 Label ID (0, 1, 2) 与 iOS 端的 Label 字符串 ("工作", "生活") 映射关系是否搞反了。

### 场景 C: 端侧更新 (On-Device Training) 不生效
*   **症状**: 运行了 `MLUpdateTask`，但模型表现没有变化。
*   **排查步骤**:
    1.  **检查 Updatable 标记**: 打开 `.mlpackage` (在 Xcode 中)，查看 Model Description。确保分类层的权重参数被标记为 `Updatable`。如果都是 `Fixed`，说明导出脚本有问题。
    2.  **检查 Loss**: 在 iOS 代码中监听训练进度，打印 Loss 变化。如果 Loss 始终是 0 或 NaN，说明输入数据格式有误。
    3.  **保存路径**: 确保更新后的模型被正确保存到了 `Application Support` 目录，并且下次启动时加载的是这个新路径，而不是 Bundle 里的旧模型。

## 3. 产出物位置速查

*   **设计文档**: `docs/01_architecture/`
*   **训练脚本**: `src/training/`
*   **转换脚本**: `src/export/`
*   **模型文件**: `models/`

---
*Created for Morning Goal Team onboarding.*
