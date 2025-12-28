# 多任务模型使用指南 (Multitask Model Quick Start)

## 概述 (Overview)

多任务模型实现了**主题分类**和**情感分析**两个任务的联合学习，采用**共享编码器 + 双任务头**架构。

**Architecture:**
```
用户输入文本
    ↓
[Tokenizer] → input_ids, attention_mask
    ↓
[共享 BERT Encoder] (4层, 512维)
    ↓
    ├─→ [主题分类头] → 16类主题
    └─→ [情感分析头] → 3类情感 (消极/中性/积极)
```

---

## 快速开始 (Quick Start)

### 1. 准备数据 (Data Preparation)

已完成！数据集已自动添加情感标签：

```bash
# 数据位置
data/processed/train_multitask.csv  # 训练集 (42,000条)
data/processed/val_multitask.csv    # 验证集 (12,000条)
data/processed/test_multitask.csv   # 测试集 (6,000条)
```

**数据格式:**
```csv
text,topic_label,sentiment_label
专注完成：睡前不看手机📈,8,2
优先事项：行李整理… study,12,1
```

**情感标签分布:**
- `0` (消极): ~1.3%
- `1` (中性): ~80.8%
- `2` (积极): ~17.9%

---

### 2. 训练模型 (Train Model)

**方法 A: 使用便捷脚本**
```bash
./scripts/train_multitask.sh
```

**方法 B: 手动运行**
```bash
python src/training/train_multitask.py \
    --data_dir data/processed \
    --base_model models/trained/distill_student \
    --output_dir models/trained/multitask_model \
    --batch_size 24 \
    --epochs 3 \
    --lr 2e-5 \
    --num_topic_labels 16 \
    --num_sentiment_labels 3
```

**训练参数说明:**
- `--base_model`: 基础模型路径（会加载BERT权重，重新初始化分类头）
- `--batch_size`: 批次大小（根据GPU内存调整）
- `--epochs`: 训练轮数
- `--lr`: 学习率
- `--limit_train`: 限制训练样本数（用于快速测试）

**预期训练时间:**
- CPU: ~2-3小时 (20k样本)
- GPU: ~20-30分钟 (20k样本)

---

### 3. 导出CoreML (Export to CoreML)

**方法 A: 使用便捷脚本**
```bash
./scripts/export_multitask_coreml.sh
```

**方法 B: 手动运行**
```bash
python src/export/export_multitask_coreml.py \
    --model_dir models/trained/multitask_model \
    --output_dir models/coreml \
    --seq_len 128 \
    --num_topic_labels 16 \
    --num_sentiment_labels 3
```

**输出文件:**
- `models/coreml/multitask_model.mlpackage` - CoreML模型
- `models/coreml/label_mapping.json` - 标签映射

---

## 模型使用 (Model Usage)

### Python 推理

```python
from transformers import AutoTokenizer
from src.models.multitask_bert import MultitaskBertForClassification
import torch

# 加载模型
model = MultitaskBertForClassification.from_pretrained(
    "models/trained/multitask_model",
    num_topic_labels=16,
    num_sentiment_labels=3
)
tokenizer = AutoTokenizer.from_pretrained("models/trained/multitask_model")
model.eval()

# 预测
text = "今天完成了重要项目，很开心！"
inputs = tokenizer(text, return_tensors="pt", max_length=128, 
                   padding="max_length", truncation=True)

with torch.no_grad():
    outputs = model(**inputs)
    topic_pred = outputs['topic_logits'].argmax(-1).item()
    sentiment_pred = outputs['sentiment_logits'].argmax(-1).item()

print(f"主题: {topic_pred}, 情感: {sentiment_pred}")
# 输出: 主题: 0, 情感: 2 (积极)
```

### iOS 使用 (Swift)

```swift
import CoreML

// 加载模型
let model = try MultitaskModel(configuration: MLModelConfiguration())

// 准备输入 (需要先tokenize)
let inputIDs: [Int32] = [...] // 128个token IDs
let attentionMask: [Int32] = [...] // 128个attention mask

// 预测
let prediction = try model.prediction(
    input_ids: MLMultiArray(inputIDs),
    attention_mask: MLMultiArray(attentionMask)
)

// 获取结果
let topicLogits = prediction.topic_logits
let sentimentLogits = prediction.sentiment_logits

let topicLabel = topicLogits.argmax() // 主题类别
let sentimentLabel = sentimentLogits.argmax() // 情感类别 (0=消极, 1=中性, 2=积极)

print("主题: \(topicLabel), 情感: \(sentimentLabel)")
```

---

## 文件结构 (File Structure)

```
MorningGoalModel/
├── src/
│   ├── models/
│   │   └── multitask_bert.py          # 多任务模型架构
│   ├── data/
│   │   └── add_sentiment_labels.py    # 情感标签自动标注
│   ├── training/
│   │   └── train_multitask.py         # 多任务训练脚本
│   └── export/
│       └── export_multitask_coreml.py # CoreML导出脚本
├── scripts/
│   ├── train_multitask.sh             # 训练便捷脚本
│   └── export_multitask_coreml.sh     # 导出便捷脚本
├── data/processed/
│   ├── train_multitask.csv            # 多任务训练集
│   ├── val_multitask.csv              # 多任务验证集
│   └── test_multitask.csv             # 多任务测试集
└── models/
    ├── trained/
    │   └── multitask_model/           # 训练好的多任务模型
    └── coreml/
        └── multitask_model.mlpackage  # CoreML模型
```

---

## 性能指标 (Performance Metrics)

训练完成后，模型会输出以下指标：

- `topic_f1`: 主题分类F1分数
- `topic_accuracy`: 主题分类准确率
- `sentiment_f1`: 情感分析F1分数
- `sentiment_accuracy`: 情感分析准确率
- `avg_f1`: 平均F1分数
- `avg_accuracy`: 平均准确率

**预期性能:**
- 主题分类: F1 > 0.95, Accuracy > 0.97
- 情感分析: F1 > 0.85, Accuracy > 0.88

---

## 常见问题 (FAQ)

### Q1: 如何调整情感标签的自动标注规则？

编辑 `src/data/add_sentiment_labels.py` 中的关键词列表：
```python
POSITIVE_KEYWORDS = ['完成', '成功', '开心', ...]
NEGATIVE_KEYWORDS = ['失败', '焦虑', '压力', ...]
```

### Q2: 如何调整两个任务的损失权重？

编辑 `src/models/multitask_bert.py` 第147行：
```python
# 默认 1:1
total_loss = topic_loss + sentiment_loss

# 调整为 7:3 (更重视主题分类)
total_loss = 0.7 * topic_loss + 0.3 * sentiment_loss
```

### Q3: 模型体积增加了多少？

仅增加了一个小分类头（~5MB），总体积约为原模型的 105%。

### Q4: 推理速度有影响吗？

几乎无影响，因为两个任务共享编码器，只需一次前向传播。

### Q5: 如何只使用其中一个任务的输出？

在推理时忽略不需要的输出即可：
```python
outputs = model(**inputs)
topic_pred = outputs['topic_logits'].argmax(-1)  # 只用主题
# 忽略 outputs['sentiment_logits']
```

---

## 下一步 (Next Steps)

1. **训练模型**: `./scripts/train_multitask.sh`
2. **评估性能**: 查看 `models/trained/multitask_model/metrics_test.json`
3. **导出CoreML**: `./scripts/export_multitask_coreml.sh`
4. **iOS集成**: 将 `multitask_model.mlpackage` 添加到Xcode项目

---

## 参考文档 (References)

- [多任务模型改进方案](./Multitask_Model_Guide.md) - 详细设计文档
- [模型调优实战指南](./Model_Optimization_Guide.md) - 优化技巧
- [架构设计与排查指南](./Architecture_and_Debugging.md) - 问题排查
