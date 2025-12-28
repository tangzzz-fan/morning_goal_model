# 多任务模型改进方案：主题分类 + 情感分析

## 1. 当前状态评估

### 现有模型
- **架构**: `BertForSequenceClassification` (单任务)
- **输出**: 16类主题分类
- **缺失**: 情感分析能力

### 目标需求（来自 PRD）
根据 `docs/00_product/PRD.md` 和 `docs/01_architecture/Technical_Overview.md`：
- **主题分类**: 工作、健康、家庭、个人发展等
- **情感分析**: 积极、中性、消极

---

## 2. 多任务学习架构设计

### 2.1 核心思路
采用 **"共享编码器 + 双任务头"** 架构：

```
用户输入文本
    ↓
[Tokenizer] → input_ids, attention_mask
    ↓
[共享 BERT Encoder] (4层, 512维)
    ↓
    ├─→ [主题分类头] → 16类主题 (Softmax)
    └─→ [情感分析头] → 3类情感 (Softmax)
```

### 2.2 优势
1. **参数共享**: 两个任务共享 BERT 编码器，模型体积增加极小（仅增加一个分类头）
2. **互补学习**: 情感和主题存在相关性，联合训练可提升两者精度
3. **端侧友好**: 单次推理同时输出两个结果，效率高

---

## 3. 实施步骤

### Step 1: 数据准备

#### 3.1 标注情感标签
为现有的 `data/processed/train.csv` 增加情感列：

**原始格式**:
```csv
text,label
专注完成：睡前不看手机📈,8
```

**新格式**:
```csv
text,topic_label,sentiment_label
专注完成：睡前不看手机📈,8,1
```

其中 `sentiment_label`:
- `0`: 消极
- `1`: 中性
- `2`: 积极

#### 3.2 自动标注方案
如果手动标注成本高，可以使用以下策略：

**方案 A: 规则 + 关键词**
```python
# 示例规则
positive_keywords = ['完成', '成功', '开心', '进步', '✅', '😊', '💪']
negative_keywords = ['失败', '焦虑', '压力', '困难', '😢', '❌']

def auto_label_sentiment(text):
    if any(kw in text for kw in positive_keywords):
        return 2  # 积极
    elif any(kw in text for kw in negative_keywords):
        return 0  # 消极
    else:
        return 1  # 中性
```

**方案 B: 使用现有情感分析 API**
```python
from transformers import pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model="uer/roberta-base-finetuned-chinanews-chinese")

def auto_label_sentiment(text):
    result = sentiment_analyzer(text)[0]
    # 映射到你的标签体系
    return map_to_your_labels(result)
```

---

### Step 2: 模型架构修改

创建新的多任务模型类：

**文件**: `src/models/multitask_bert.py`

```python
import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel

class MultitaskBertForClassification(BertPreTrainedModel):
    def __init__(self, config, num_topic_labels=16, num_sentiment_labels=3):
        super().__init__(config)
        self.num_topic_labels = num_topic_labels
        self.num_sentiment_labels = num_sentiment_labels
        
        # 共享的 BERT 编码器
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # 主题分类头
        self.topic_classifier = nn.Linear(config.hidden_size, num_topic_labels)
        
        # 情感分析头
        self.sentiment_classifier = nn.Linear(config.hidden_size, num_sentiment_labels)
        
        self.init_weights()
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                topic_labels=None, sentiment_labels=None):
        # 共享编码器
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # 两个任务的 logits
        topic_logits = self.topic_classifier(pooled_output)
        sentiment_logits = self.sentiment_classifier(pooled_output)
        
        # 计算损失（如果提供了标签）
        total_loss = None
        if topic_labels is not None and sentiment_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            topic_loss = loss_fct(topic_logits, topic_labels)
            sentiment_loss = loss_fct(sentiment_logits, sentiment_labels)
            # 可以调整权重，默认 1:1
            total_loss = topic_loss + sentiment_loss
        
        return {
            'loss': total_loss,
            'topic_logits': topic_logits,
            'sentiment_logits': sentiment_logits
        }
```

---

### Step 3: 训练脚本修改

**文件**: `src/training/train_multitask.py`

```python
from transformers import Trainer, TrainingArguments
from src.models.multitask_bert import MultitaskBertForClassification
import pandas as pd

# 加载数据
train_df = pd.read_csv('data/processed/train_multitask.csv')

# 自定义 Dataset
class MultitaskDataset(torch.utils.data.Dataset):
    def __init__(self, texts, topic_labels, sentiment_labels, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.topic_labels = topic_labels
        self.sentiment_labels = sentiment_labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['topic_labels'] = torch.tensor(self.topic_labels[idx])
        item['sentiment_labels'] = torch.tensor(self.sentiment_labels[idx])
        return item
    
    def __len__(self):
        return len(self.topic_labels)

# 初始化模型
model = MultitaskBertForClassification.from_pretrained(
    'models/trained/distill_student',  # 从现有模型初始化
    num_topic_labels=16,
    num_sentiment_labels=3
)

# 训练配置
training_args = TrainingArguments(
    output_dir='models/trained/multitask_model',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
```

---

### Step 4: CoreML 导出修改

**文件**: `src/export/export_multitask_coreml.py`

```python
class MultitaskWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # 返回两个输出
        return outputs['topic_logits'], outputs['sentiment_logits']

# 转换时
wrapper = MultitaskWrapper(model)
traced_model = torch.jit.trace(wrapper, (dummy_input_ids, dummy_attention_mask))

# CoreML 转换（需要定义两个输出）
mlmodel = ct.convert(
    traced_model,
    inputs=[...],
    outputs=[
        ct.TensorType(name="topic_logits"),
        ct.TensorType(name="sentiment_logits")
    ],
    ...
)
```

---

## 4. iOS 端使用示例

```swift
let model = try MultitaskModel(configuration: MLModelConfiguration())
let prediction = try model.prediction(input_ids: ids, attention_mask: mask)

// 获取两个任务的结果
let topicLabel = prediction.topic_logits.argmax()  // 主题
let sentimentLabel = prediction.sentiment_logits.argmax()  // 情感

print("主题: \(topicLabel), 情感: \(sentimentLabel)")
```

---

## 5. 渐进式实施建议

### 阶段 1: 快速验证（1-2天）
1. 使用规则或现有 API 为 1000 条数据自动标注情感
2. 训练一个小型多任务模型
3. 评估双任务精度是否可接受

### 阶段 2: 数据优化（3-5天）
1. 人工校验自动标注结果
2. 补充边界 case 的标注
3. 重新训练并评估

### 阶段 3: 端侧部署（2-3天）
1. 修改导出脚本
2. 生成 `.mlpackage`
3. iOS 集成测试

---

## 6. 预期效果

- **模型体积增加**: < 5MB（仅增加一个小分类头）
- **推理速度**: 几乎无影响（单次前向传播）
- **精度**: 
  - 主题分类: 保持 97%+
  - 情感分析: 预计 85-90%（取决于标注质量）

---

## 7. 备选方案：两个独立模型

如果多任务学习效果不理想，可以训练两个独立模型：
- **优点**: 每个任务独立优化
- **缺点**: 模型体积翻倍，推理需要两次

**不推荐**，除非多任务方案失败。
