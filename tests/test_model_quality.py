"""
模型质量测试
测试模型在测试集上的性能指标是否达到预设阈值
"""
import pytest
import torch
import pandas as pd
import sys
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import json

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.evaluation.multitask_inference import MultitaskPredictor


class TestModelQuality:
    """测试模型质量指标"""
    
    @pytest.fixture(scope="class")
    def predictor(self, model_config_path):
        """加载预测器"""
        model_path = model_config_path["model_path"]
        if not Path(model_path).exists():
            pytest.skip(f"模型路径不存在: {model_path}")
        
        return MultitaskPredictor(model_path, use_gpu=torch.cuda.is_available())
    
    @pytest.fixture(scope="class")
    def test_predictions(self, predictor, test_data_path):
        """在测试集上运行预测（缓存结果）"""
        if not Path(test_data_path).exists():
            pytest.skip(f"测试数据不存在: {test_data_path}")
        
        # 加载测试数据
        df = pd.read_csv(test_data_path)
        
        # 限制测试样本数量以加快测试（可以通过环境变量控制）
        import os
        max_samples = int(os.environ.get("TEST_MAX_SAMPLES", "500"))
        if len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42)
        
        texts = df['text'].tolist()
        true_topics = df['topic_label'].tolist()
        true_sentiments = df['sentiment_label'].tolist()
        
        # 批量预测
        print(f"\n正在对 {len(texts)} 个样本进行预测...")
        pred_topics = []
        pred_sentiments = []
        
        for i, text in enumerate(texts):
            if i % 100 == 0:
                print(f"进度: {i}/{len(texts)}")
            
            result = predictor.predict(text)
            pred_topics.append(result['topic']['id'])
            pred_sentiments.append(result['sentiment']['id'])
        
        return {
            'true_topics': true_topics,
            'pred_topics': pred_topics,
            'true_sentiments': true_sentiments,
            'pred_sentiments': pred_sentiments
        }
    
    def test_topic_accuracy(self, test_predictions, model_config_path):
        """测试主题分类准确率"""
        accuracy = accuracy_score(
            test_predictions['true_topics'],
            test_predictions['pred_topics']
        )
        
        min_accuracy = model_config_path["quality_thresholds"]["topic_accuracy_min"]
        
        print(f"\n主题分类准确率: {accuracy:.4f} (最低要求: {min_accuracy:.4f})")
        
        assert accuracy >= min_accuracy, \
            f"主题分类准确率 {accuracy:.4f} 低于阈值 {min_accuracy:.4f}"
    
    def test_topic_f1_score(self, test_predictions, model_config_path):
        """测试主题分类F1分数"""
        f1 = f1_score(
            test_predictions['true_topics'],
            test_predictions['pred_topics'],
            average='macro'
        )
        
        min_f1 = model_config_path["quality_thresholds"]["topic_f1_min"]
        
        print(f"\n主题分类Macro F1: {f1:.4f} (最低要求: {min_f1:.4f})")
        
        assert f1 >= min_f1, \
            f"主题分类F1分数 {f1:.4f} 低于阈值 {min_f1:.4f}"
    
    def test_sentiment_accuracy(self, test_predictions, model_config_path):
        """测试情感分析准确率"""
        accuracy = accuracy_score(
            test_predictions['true_sentiments'],
            test_predictions['pred_sentiments']
        )
        
        min_accuracy = model_config_path["quality_thresholds"]["sentiment_accuracy_min"]
        
        print(f"\n情感分析准确率: {accuracy:.4f} (最低要求: {min_accuracy:.4f})")
        
        assert accuracy >= min_accuracy, \
            f"情感分析准确率 {accuracy:.4f} 低于阈值 {min_accuracy:.4f}"
    
    def test_sentiment_f1_score(self, test_predictions, model_config_path):
        """测试情感分析F1分数"""
        f1 = f1_score(
            test_predictions['true_sentiments'],
            test_predictions['pred_sentiments'],
            average='macro'
        )
        
        min_f1 = model_config_path["quality_thresholds"]["sentiment_f1_min"]
        
        print(f"\n情感分析Macro F1: {f1:.4f} (最低要求: {min_f1:.4f})")
        
        assert f1 >= min_f1, \
            f"情感分析F1分数 {f1:.4f} 低于阈值 {min_f1:.4f}"
    
    def test_detailed_metrics(self, test_predictions, project_root):
        """计算并保存详细指标"""
        from sklearn.metrics import classification_report
        
        # 主题分类报告
        topic_report = classification_report(
            test_predictions['true_topics'],
            test_predictions['pred_topics'],
            output_dict=True,
            zero_division=0
        )
        
        # 情感分析报告
        sentiment_report = classification_report(
            test_predictions['true_sentiments'],
            test_predictions['pred_sentiments'],
            output_dict=True,
            zero_division=0
        )
        
        # 保存报告
        report_dir = project_root / "tests" / "reports"
        report_dir.mkdir(exist_ok=True)
        
        with open(report_dir / "quality_test_report.json", "w") as f:
            json.dump({
                "topic_classification": topic_report,
                "sentiment_analysis": sentiment_report
            }, f, indent=2)
        
        print(f"\n详细报告已保存到: {report_dir / 'quality_test_report.json'}")


class TestModelRobustness:
    """测试模型鲁棒性"""
    
    @pytest.fixture(scope="class")
    def predictor(self, model_config_path):
        """加载预测器"""
        model_path = model_config_path["model_path"]
        if not Path(model_path).exists():
            pytest.skip(f"模型路径不存在: {model_path}")
        
        return MultitaskPredictor(model_path, use_gpu=torch.cuda.is_available())
    
    def test_confidence_distribution(self, predictor, test_data_path):
        """测试置信度分布（避免模型过于自信或不自信）"""
        if not Path(test_data_path).exists():
            pytest.skip(f"测试数据不存在: {test_data_path}")
        
        df = pd.read_csv(test_data_path)
        texts = df['text'].tolist()[:100]  # 取100个样本
        
        confidences = []
        for text in texts:
            result = predictor.predict(text)
            confidences.append(result['topic']['confidence'])
        
        avg_confidence = sum(confidences) / len(confidences)
        
        print(f"\n平均置信度: {avg_confidence:.4f}")
        
        # 置信度应该在合理范围内（不应该总是接近1或接近1/16）
        assert 0.15 < avg_confidence < 0.95, \
            f"置信度分布异常: {avg_confidence:.4f}"
    
    def test_label_distribution(self, predictor, test_data_path):
        """测试预测标签分布（避免模型总是预测某几个类别）"""
        if not Path(test_data_path).exists():
            pytest.skip(f"测试数据不存在: {test_data_path}")
        
        df = pd.read_csv(test_data_path)
        texts = df['text'].tolist()[:200]  # 取200个样本
        
        topic_counts = {}
        for text in texts:
            result = predictor.predict(text)
            topic_id = result['topic']['id']
            topic_counts[topic_id] = topic_counts.get(topic_id, 0) + 1
        
        # 至少应该预测到5个不同的类别
        unique_labels = len(topic_counts)
        print(f"\n预测到的不同主题数: {unique_labels}/16")
        
        assert unique_labels >= 5, \
            f"模型预测的类别过少: {unique_labels}/16"

