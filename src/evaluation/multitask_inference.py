"""
多任务模型推理脚本
用于加载训练好的模型并进行预测
"""
import torch
from transformers import AutoTokenizer, AutoConfig
import numpy as np
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.multitask_bert import MultitaskBertForClassification


class MultitaskPredictor:
    def __init__(self, model_path, use_gpu=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        print(f"Loading model from {model_path} to {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        config = AutoConfig.from_pretrained(model_path)
        
        self.model = MultitaskBertForClassification.from_pretrained(
            model_path,
            config=config,
            num_topic_labels=config.num_topic_labels if hasattr(config, 'num_topic_labels') else 16,
            num_sentiment_labels=config.num_sentiment_labels if hasattr(config, 'num_sentiment_labels') else 3
        )
        self.model.to(self.device)
        self.model.eval()
        
        # 标签映射（可以根据实际情况修改）
        self.topic_map = {
            0: "健康", 1: "学习", 2: "工作", 3: "财务", 
            4: "社交", 5: "家庭", 6: "娱乐", 7: "旅行",
            8: "生活习惯", 9: "技能提升", 10: "心理健康", 11: "运动",
            12: "饮食", 13: "睡眠", 14: "阅读", 15: "其他"
        }
        self.sentiment_map = {
            0: "消极", 1: "中性", 2: "积极"
        }

    def predict(self, text):
        """
        对单个文本进行预测
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        
        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # 获取预测结果
        topic_logits = outputs['topic_logits']
        sentiment_logits = outputs['sentiment_logits']
        
        # 获取概率
        topic_probs = torch.softmax(topic_logits, dim=1)
        sentiment_probs = torch.softmax(sentiment_logits, dim=1)
        
        # 获取最大概率的类别
        topic_id = torch.argmax(topic_probs, dim=1).item()
        sentiment_id = torch.argmax(sentiment_probs, dim=1).item()
        
        return {
            "text": text,
            "topic": {
                "id": topic_id,
                "label": self.topic_map.get(topic_id, str(topic_id)),
                "confidence": topic_probs[0][topic_id].item()
            },
            "sentiment": {
                "id": sentiment_id,
                "label": self.sentiment_map.get(sentiment_id, str(sentiment_id)),
                "confidence": sentiment_probs[0][sentiment_id].item()
            }
        }

    def predict_batch(self, texts):
        """
        批量预测
        """
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results

def run_inference(model_path, texts):
    predictor = MultitaskPredictor(model_path)
    results = predictor.predict_batch(texts)
    
    print("\n" + "="*50)
    print("Inference Results")
    print("="*50)
    
    for res in results:
        print(f"\nInput: {res['text']}")
        print(f"Topic: {res['topic']['label']} (Conf: {res['topic']['confidence']:.4f})")
        print(f"Sentiment: {res['sentiment']['label']} (Conf: {res['sentiment']['confidence']:.4f})")
        
    return results
