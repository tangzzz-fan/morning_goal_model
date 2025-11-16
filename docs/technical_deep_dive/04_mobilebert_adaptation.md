# 技术深潜：MobileBERT 适配方案

本文档详细阐述将 MobileBERT 模型适配到移动端中文文本分类任务的完整技术方案，覆盖从预训练、微调到量化部署的全过程。

## 1. 核心流程概述

MobileBERT 的适配遵循一个标准但针对移动端优化的流程：

1.  **教师模型准备**：使用 `bert-base-chinese` 作为知识蒸馏的教师模型。
2.  **学生模型选择**：选择与 MobileBERT 结构相似的轻量级模型作为学生，或直接使用 MobileBERT 的预训练权重。
3.  **知识蒸馏 (Knowledge Distillation)**：将教师模型的知识迁移到学生模型。这是核心步骤，目的是在保留关键信息的同时大幅缩小模型体积。
4.  **任务微调 (Task Fine-tuning)**：在下游任务（如文本分类）的数据集上对蒸馏后的学生模型进行微调。
5.  **量化感知训练 (Quantization-Aware Training, QAT)**：在微调过程中模拟量化操作，使模型在转换为 INT8 等低精度格式时性能损失最小。
6.  **模型转换与部署**：将量化后的模型转换为 Core ML 格式，并集成到 iOS 应用中。

## 2. 预训练与微调流程

### 2.1. 中文文本预处理

与标准 BERT 不同，针对中文 MobileBERT 的预处理需注意：

-   **Tokenizer**：必须使用与 `bert-base-chinese` 兼容的 Tokenizer，以确保词表一致性。推荐使用 `BertTokenizer`。
-   **文本长度**：移动端应用通常处理短文本。建议将最大序列长度 (max_seq_length) 设置为 128 或更低，以减少计算开销和内存占用。
-   **特殊标记**：确保 `[CLS]`、`[SEP]` 和 `[PAD]` 标记被正确添加和处理。

### 2.2. 知识蒸馏的损失函数配置

知识蒸馏的目标是让学生模型 (`S`) 的输出尽可能接近教师模型 (`T`)。这通过一个组合损失函数实现：

**L_total = α * L_distill + (1 - α) * L_task**

-   **L_distill (蒸馏损失)**：衡量教师和学生模型输出分布的差异。通常使用 **KL 散度 (Kullback-Leibler Divergence)**。为了让学生模型学习教师模型的 "软标签" (soft labels)，即 logits 在经过 Softmax 时的概率分布，会引入一个温度参数 `T`。

    ```python
    # T \u003e 1, 使概率分布更平滑
    loss_kl = nn.KLDivLoss(reduction='batchmean')(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1)
    ) * (temperature ** 2)
    ```

-   **L_task (任务损失)**：学生模型在真实标签上的损失，通常是**交叉熵损失 (Cross-Entropy Loss)**。

    ```python
    loss_ce = nn.CrossEntropyLoss()(student_logits, labels)
    ```

-   **α (权重因子)**：平衡蒸馏损失和任务损失的超参数，通常需要根据实验调优。

### 2.3. 训练策略

-   **渐进式知识迁移**：可以先进行预训练蒸馏（在通用语料上），再进行任务蒸馏（在特定任务数据上），效果通常更优。
-   **中间层匹配**：除了匹配最终的 logits 输出，还可以增加匹配 Transformer 中间隐藏层或注意力矩阵的损失，让学生模型学习教师模型的内部表示。

    ```python
    # 示例：匹配隐藏层表示
    loss_hidden = nn.MSELoss()(student_hidden_states, teacher_hidden_states)
    ```

## 3. 移动端量化训练策略

为了在保持高精度的同时实现极致性能，必须采用**量化感知训练 (QAT)**。

### 3.1. QAT 原理

QAT 在前向传播时模拟量化过程（将 FP32 权重和激活值伪量化为 INT8），但在反向传播时依然使用 FP32 计算梯度。这使得模型能够“适应”量化带来的精度损失。

### 3.2. 实现方案

PyTorch 提供了原生的 QAT 支持。

1.  **模型准备**：在模型中插入 `QuantStub` 和 `DeQuantStub` 来指定量化的起点和终点。
2.  **配置 QAT**：定义量化配置，包括量化器类型（如 fbgemm）和激活函数的观察者 (Observer)。

    ```python
    import torch.quantization as quantization

    model.qconfig = quantization.get_default_qat_qconfig('fbgemm')
    model_fused = quantization.fuse_modules(model, [['linear', 'relu']])
    model_prepared = quantization.prepare_qat(model_fused.train())
    ```

3.  **训练**：使用 `model_prepared` 进行正常的微调训练。训练过程中，观察者会收集激活值的范围信息，用于后续的量化转换。
4.  **转换**：训练完成后，将模型转换为真正的量化模型。

    ```python
    model_prepared.eval()
    model_quantized = quantization.convert(model_prepared)
    ```

## 4. 模型评估与部署检查清单

### 4.1. 评估指标

| 指标 | 描述 | 基准参考 (iPhone 13) |
| :--- | :--- | :--- |
| **模型体积** | INT8 量化后的 `.mlmodel` 文件大小 | \u003c 20 MB |
| **推理延迟** | 单次文本分类任务的端到端耗时 | \u003c 25 ms |
| **CPU 占用** | 推理期间的 CPU 使用率峰值 | \u003c 40% |
| **内存占用** | 模型加载和推理时的内存峰值 | \u003c 100 MB |
| **精度 (Accuracy)** | 在标准测试集上的分类准确率 | \u003e 95% (对比教师模型损失 \u003c 2%) |

### 4.2. 部署检查清单

-   [ ] **模型验证**：确认转换后的 Core ML 模型在 Xcode 预览中输出与 PyTorch 端一致。
-   [ ] **计算单元 (Compute Units)**：在 Core ML 配置中设置为 `ct.ComputeUnit.CPU_AND_GPU`，以充分利用 ANE 和 GPU。
-   [ ] **性能剖析**：使用 Xcode Instruments 的 Core ML 和 Metal 分析工具检查模型每一层的执行时间，定位性能瓶颈。
-   [ ] **线程管理**：确保模型推理在后台线程执行，避免阻塞 UI 主线程。
-   [ ] **版本控制**：实现模型版本管理机制，支持模型热更新和回滚。
-   [ ] **异常处理**：添加模型加载失败、推理异常的捕获和处理逻辑。