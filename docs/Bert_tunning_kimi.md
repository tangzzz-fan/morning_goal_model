# 端侧 NLP 模型训练与部署里程碑路线图

针对你的需求（使用 `bert-base-chinese` 或 `MobileBERT` 分析用户每日目标并生成提示），我为你制定了一个从 PC 训练到 iOS 端侧部署的完整里程碑计划。此计划结合搜索结果中的最佳实践和 Apple 官方 Core ML 指南。

---

## **里程碑 1: 环境搭建与数据准备 (1-2 周)**

### 目标任务
- 配置 macOS/Windows 训练环境
- 收集和预处理用户目标数据
- 建立评估基准

### 关键行动
1. **环境配置**：
   - 安装 Python 3.8+、PyTorch/TensorFlow
   - 安装 `transformers` 库：`pip install transformers`
   - 安装 Core ML Tools：`pip install coremltools`
   - 准备 GPU 支持（CUDA 或 MPS for Mac）

2. **数据工程**：
   - 收集至少 5000+ 条用户目标文本样本（覆盖不同类别如健康、学习、工作等）
   - 数据清洗：去除噪声、统一编码格式
   - 标注数据：为每个目标添加标签（如完成度、类别、难度）
   - 数据增强：使用回译、同义词替换扩充数据集

3. **基线建立**：
   - 划分训练/验证/测试集（70/15/15）
   - 定义评估指标：准确率、F1 分数、推理速度、模型大小

### 验证标准
- 训练脚本能在本地成功运行小型测试
- 数据质量检查通过（无乱码、标签分布合理）

---

## **里程碑 2: 基线模型训练与评估 (2-3 周)**

### 目标任务
- 在 PC 上训练初始 `bert-base-chinese` 模型
- 实现目标分析任务（分类或生成）
- 建立性能基准线

### 关键行动
1. **模型加载**：
   ```python
   from transformers import BertTokenizer, BertForSequenceClassification
   tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
   model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=5)
   ```

2. **训练实现**：
   - 参考搜索结果中的训练模板：
     - 使用 `AdamW` 优化器，学习率 2e-5
     - Batch size: 16（根据 GPU 显存调整）
     - 最大序列长度：128（平衡精度与速度）
     - 训练 3-5 个 epoch，使用验证集早停
   - 添加梯度裁剪（clip_grad_norm）防止梯度爆炸

3. **性能评估**：
   - 记录训练时间、验证集准确率、损失值
   - 测试集上推理速度（tokens/秒）
   - 模型大小：原始 `bert-base-chinese` 约 400MB

### 验证标准
- 验证集准确率达到可接受水平（如 >85%）
- 训练日志完整，可复现结果

---

## **里程碑 3: 模型优化与分析 (3-4 周)**

### 目标任务
- 优化模型以满足端侧部署要求（<100MB，实时推理）
- 深入分析模型性能瓶颈

### 关键行动

1. **超参数优化**（提升精度）：
   - 使用 Optuna 或 Ray Tune 自动化调参
   - 调整关键参数：学习率 (1e-5 to 5e-5)、batch size、dropout (0.1-0.5)
   - 尝试不同的优化器：AdamW vs SGD with warmup

2. **模型压缩**（减小尺寸）：
   - **剪枝 (Pruning)**：使用 `torch.nn.utils.prune` 移除 30-50% 不重要的权重
   - **量化 (Quantization)**：将 FP32 转换为 INT8，模型大小缩小 75%
     - 使用 PyTorch 的 `torch.quantization` 或 Hugging Face 的 `optimum` 库
   - **知识蒸馏**：训练一个 3-4 层的小模型模仿 12 层的 BERT

3. **性能分析**：
   - 使用 PyTorch Profiler 分析每个层的计算耗时
   - 识别瓶颈层（通常是 Transformer 层和全连接层）
   - 记录优化前后对比：训练时间缩短 30%、推理速度加快 20%

### 验证标准
- 模型大小 < 100MB（量化后）
- 在测试集上推理速度提升 2-3 倍
- 精度损失 < 5%（相比基线模型）

---

## **里程碑 4: Core ML 模型转换 (1-2 周)**

### 目标任务
- 将优化后的 PyTorch 模型转换为 Core ML 格式
- 验证转换后模型的功能和性能

### 关键行动

1. **转换准备**：
   - 确保模型为 traced 或 script 模式
   - 准备示例输入：`input_ids` 和 `attention_mask`

2. **执行转换**：
   ```python
   import coremltools as ct
   
   # 加载优化后的模型
   traced_model = torch.jit.trace(model, example_inputs)
   
   # 转换为 Core ML
   mlmodel = ct.convert(
       traced_model,
       inputs=[ct.TensorType(shape=(1, 128), dtype=np.int32),
               ct.TensorType(shape=(1, 128), dtype=np.int32)],
       compute_units=ct.ComputeUnit.ALL  # 使用 CPU/GPU/Neural Engine
   )
   
   # 保存模型
   mlmodel.save("GoalAnalyzer.mlmodel")
   ```

3. **转换后验证**：
   - 在 Xcode 中打开 `.mlmodel` 文件，检查输入输出格式
   - 使用 Core ML 的预测 API 测试转换后的模型
   - 验证模型在 macOS 上的推理结果与 PyTorch 一致

### 验证标准
- `.mlmodel` 文件成功生成且 < 100MB
- 在 macOS 上推理结果与 PyTorch 版本误差 < 1%
- Xcode 模型查看器显示所有层都支持 Neural Engine

---

## **里程碑 5: iOS 端侧优化与训练 (2-3 周)**

### 目标任务
- 实现 Core ML 的**现场训练 (On-Device Training)** 功能
- 针对 iOS 设备进行专项优化

### 关键行动

1. **NLP Updatable 配置**：
   - 修改 `.mlmodel` 为可更新模式（需使用 Core ML Tools 的 MIL 构建器手动定义）
   - 将最后几层（如分类头）标记为可训练，冻结底层参数以节省计算
   - 配置优化器：SGD with momentum 或 Adam

2. **iOS 特定优化**：
   - **Compute Units**：设置为 `.all` 以自动利用 Neural Engine
   - **输入形状优化**：使用固定形状 (1, 128) 而非动态形状，提升效率
   - **量化感知训练**：在转换前使用 QAT 进一步减小模型
   - **内存管理**：使用 `MLModelConfiguration` 限制内存占用 < 50MB

3. **现场训练实现**：
   ```swift
   // 在 iOS 应用中
   let updateTask = try MLUpdateTask(forModelAt: modelURL,
                                     trainingData: trainingData,
                                     configuration: config,
                                     completionHandler: { ... })
   updateTask.resume()
   ```

4. **性能测试**：
   - 在真实设备上测试（iPhone 12+ 或 iPad Air+）
   - 测量单次推理时间（目标 < 50ms）
   - 测量单次现场训练时间（目标 < 5秒）

### 验证标准
- Updatable 模型在 iOS 设备上成功更新
- 推理延迟 < 50ms on iPhone 13
- 模型更新后精度提升 3-5%

---

## **里程碑 6: iOS App 集成与部署 (2 周)**

### 目标任务
- 将模型集成到 iOS 应用中
- 实现每日目标分析功能
- 准备 App Store 发布

### 关键行动

1. **Xcode 集成**：
   - 将 `.mlmodel` 文件拖入 Xcode 项目
   - 自动生成模型接口：`GoalAnalyzer()` 类
   - 实现批量预测和实时预测两种模式

2. **功能开发**：
   - 每日目标文本预处理：分词、截断、填充
   - 提示生成逻辑：基于模型输出（分类概率或生成文本）
   - 用户反馈收集：记录用户对提示的评分，作为新的训练数据

3. **隐私与合规**：
   - 所有数据处理都在设备本地完成
   - 在 App Store 描述中明确说明"不收集用户数据"
   - 遵循 Apple 的隐私政策要求

4. **A/B 测试**：
   - 部署两个模型版本（基础 vs 优化）
   - 收集用户满意度指标
   - 根据反馈决定最终发布版本

### 验证标准
- App 在 TestFlight 上运行稳定
- 用户反馈积极（提示有用率 >70%）
- 通过 App Store 审核

---

## **关键技术决策点**

### 选择 `bert-base-chinese` vs `MobileBERT`
| 维度 | bert-base-chinese | MobileBERT (推荐) |
|------|-------------------|-------------------|
| 大小 | 400MB → 100MB (量化后) | 25MB → 10MB (量化后) |
| 精度 | 高 (91%+) | 稍低 (88%+) |
| 速度 | 较慢 (100ms) | **快 (30ms)** |
| 训练成本 | 高 | **低** |
| Core ML 支持 | 良好 | **优秀（专为移动设计）** |

**建议**：优先使用 **MobileBERT**，若精度不足再考虑蒸馏后的 `bert-base-chinese`。

### 优化策略优先级
1. **量化**（必须）→ 大小↓75%，速度↑2x
2. **剪枝**（推荐）→ 大小↓50%，速度↑1.5x
3. **蒸馏**（可选）→ 大小↓80%，精度损失最小

---

## **风险与应对**

| 风险 | 应对策略 |
|------|----------|
| Core ML 转换失败 | 使用最新版 coremltools；在 PyTorch 2.0+ 上重新导出 |
| 端侧训练内存溢出 | 冻结 90% 参数；减少 batch size 到 1；使用梯度累积 |
| 精度下降过多 | 采用混合精度训练；使用知识蒸馏；收集更多数据 |
| iOS 版本兼容性 | 最低支持 iOS 16+；使用 Core ML 3+ API |

---

此路线图可根据你的具体需求调整。建议每个里程碑结束后进行一次代码审查和性能评估，确保达到预期目标后再进入下一阶段。