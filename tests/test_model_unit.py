"""
模型单元测试
测试模型基本功能、输入输出形状、组件正常工作
"""
import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.multitask_bert import MultitaskBertForClassification
from transformers import AutoConfig


class TestModelArchitecture:
    """测试模型架构"""
    
    def test_model_initialization(self):
        """测试模型能正确初始化"""
        config = AutoConfig.from_pretrained("bert-base-chinese")
        model = MultitaskBertForClassification(
            config,
            num_topic_labels=16,
            num_sentiment_labels=3
        )
        
        assert model is not None
        assert model.num_topic_labels == 16
        assert model.num_sentiment_labels == 3
        assert hasattr(model, 'bert')
        assert hasattr(model, 'topic_classifier')
        assert hasattr(model, 'sentiment_classifier')
    
    def test_model_forward_shape(self, mock_model_inputs):
        """测试模型前向传播输出形状正确"""
        config = AutoConfig.from_pretrained("bert-base-chinese")
        model = MultitaskBertForClassification(
            config,
            num_topic_labels=16,
            num_sentiment_labels=3
        )
        model.eval()
        
        with torch.no_grad():
            outputs = model(
                input_ids=mock_model_inputs["input_ids"],
                attention_mask=mock_model_inputs["attention_mask"]
            )
        
        # 检查输出
        assert 'topic_logits' in outputs
        assert 'sentiment_logits' in outputs
        assert outputs['topic_logits'].shape == (2, 16)
        assert outputs['sentiment_logits'].shape == (2, 3)
    
    def test_model_loss_computation(self, mock_model_inputs):
        """测试模型能正确计算损失"""
        config = AutoConfig.from_pretrained("bert-base-chinese")
        model = MultitaskBertForClassification(
            config,
            num_topic_labels=16,
            num_sentiment_labels=3
        )
        model.train()
        
        outputs = model(
            input_ids=mock_model_inputs["input_ids"],
            attention_mask=mock_model_inputs["attention_mask"],
            topic_labels=mock_model_inputs["topic_labels"],
            sentiment_labels=mock_model_inputs["sentiment_labels"]
        )
        
        # 检查损失
        assert 'loss' in outputs
        assert 'topic_loss' in outputs
        assert 'sentiment_loss' in outputs
        assert outputs['loss'] is not None
        assert outputs['topic_loss'] is not None
        assert outputs['sentiment_loss'] is not None
        assert outputs['loss'].item() > 0
    
    def test_model_gradient_flow(self, mock_model_inputs):
        """测试模型梯度能正常反向传播"""
        config = AutoConfig.from_pretrained("bert-base-chinese")
        model = MultitaskBertForClassification(
            config,
            num_topic_labels=16,
            num_sentiment_labels=3
        )
        model.train()
        
        outputs = model(
            input_ids=mock_model_inputs["input_ids"],
            attention_mask=mock_model_inputs["attention_mask"],
            topic_labels=mock_model_inputs["topic_labels"],
            sentiment_labels=mock_model_inputs["sentiment_labels"]
        )
        
        loss = outputs['loss']
        loss.backward()
        
        # 检查梯度
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradients = True
                break
        
        assert has_gradients, "模型参数没有梯度"
    
    def test_model_output_range(self, mock_model_inputs):
        """测试模型输出概率范围正确（经过softmax后应在[0,1]）"""
        config = AutoConfig.from_pretrained("bert-base-chinese")
        model = MultitaskBertForClassification(
            config,
            num_topic_labels=16,
            num_sentiment_labels=3
        )
        model.eval()
        
        with torch.no_grad():
            outputs = model(
                input_ids=mock_model_inputs["input_ids"],
                attention_mask=mock_model_inputs["attention_mask"]
            )
            
            topic_probs = torch.softmax(outputs['topic_logits'], dim=1)
            sentiment_probs = torch.softmax(outputs['sentiment_logits'], dim=1)
        
        # 检查概率和为1
        assert torch.allclose(topic_probs.sum(dim=1), torch.ones(2), atol=1e-5)
        assert torch.allclose(sentiment_probs.sum(dim=1), torch.ones(2), atol=1e-5)
        
        # 检查概率范围
        assert (topic_probs >= 0).all() and (topic_probs <= 1).all()
        assert (sentiment_probs >= 0).all() and (sentiment_probs <= 1).all()


class TestModelSaveLoad:
    """测试模型保存和加载"""
    
    def test_model_save_load(self, tmp_path):
        """测试模型能正确保存和加载"""
        config = AutoConfig.from_pretrained("bert-base-chinese")
        model = MultitaskBertForClassification(
            config,
            num_topic_labels=16,
            num_sentiment_labels=3
        )
        
        # 保存模型
        save_path = tmp_path / "test_model"
        model.save_pretrained(save_path)
        config.save_pretrained(save_path)
        
        # 加载模型
        loaded_model = MultitaskBertForClassification.from_pretrained(
            save_path,
            num_topic_labels=16,
            num_sentiment_labels=3
        )
        
        assert loaded_model is not None
        assert loaded_model.num_topic_labels == 16
        assert loaded_model.num_sentiment_labels == 3

