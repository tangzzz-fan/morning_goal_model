"""
Multitask BERT Model for Topic Classification + Sentiment Analysis
共享编码器 + 双任务头架构
"""
import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel


class MultitaskBertForClassification(BertPreTrainedModel):
    """
    多任务BERT分类模型
    
    架构:
        用户输入文本
            ↓
        [Tokenizer] → input_ids, attention_mask
            ↓
        [共享 BERT Encoder] (4层, 512维)
            ↓
            ├─→ [主题分类头] → 16类主题 (Softmax)
            └─→ [情感分析头] → 3类情感 (Softmax)
    
    Args:
        config: BERT配置
        num_topic_labels: 主题分类类别数 (默认16)
        num_sentiment_labels: 情感分析类别数 (默认3: 消极/中性/积极)
    """
    
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
        
        # 初始化权重
        self.post_init()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        topic_labels=None,
        sentiment_labels=None,
        return_dict=True,
    ):
        """
        前向传播
        
        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            token_type_ids: token类型IDs
            topic_labels: 主题标签 (用于训练)
            sentiment_labels: 情感标签 (用于训练)
            return_dict: 是否返回字典格式
        
        Returns:
            dict: {
                'loss': 总损失 (如果提供了标签),
                'topic_logits': 主题分类logits,
                'sentiment_logits': 情感分析logits,
                'topic_loss': 主题分类损失,
                'sentiment_loss': 情感分析损失
            }
        """
        # 共享编码器
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        
        # 使用 [CLS] token 的表示
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # 两个任务的 logits
        topic_logits = self.topic_classifier(pooled_output)
        sentiment_logits = self.sentiment_classifier(pooled_output)
        
        # 计算损失（如果提供了标签）
        total_loss = None
        topic_loss = None
        sentiment_loss = None
        
        if topic_labels is not None and sentiment_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            topic_loss = loss_fct(topic_logits, topic_labels)
            sentiment_loss = loss_fct(sentiment_logits, sentiment_labels)
            # 可以调整权重，默认 1:1
            # 如果某个任务更重要，可以设置权重，例如: total_loss = 0.7 * topic_loss + 0.3 * sentiment_loss
            total_loss = topic_loss + sentiment_loss
        
        if not return_dict:
            output = (topic_logits, sentiment_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output
        
        return {
            'loss': total_loss,
            'topic_logits': topic_logits,
            'sentiment_logits': sentiment_logits,
            'topic_loss': topic_loss,
            'sentiment_loss': sentiment_loss,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            'attentions': outputs.attentions if hasattr(outputs, 'attentions') else None,
        }
