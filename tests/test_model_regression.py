"""
模型回归测试
确保新模型不比基准模型差
"""
import pytest
import torch
import pandas as pd
import sys
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.evaluation.multitask_inference import MultitaskPredictor


class TestModelRegression:
    """测试模型回归（对比基准模型）"""
    
    @pytest.fixture(scope="class")
    def baseline_metrics(self, project_root):
        """加载基准模型的性能指标"""
        baseline_file = project_root / "tests" / "baseline_metrics.json"
        
        if not baseline_file.exists():
            pytest.skip("基准指标文件不存在，跳过回归测试")
        
        with open(baseline_file) as f:
            return json.load(f)
    
    @pytest.fixture(scope="class")
    def current_predictor(self, model_config_path):
        """加载当前模型"""
        model_path = model_config_path["model_path"]
        if not Path(model_path).exists():
            pytest.skip(f"模型路径不存在: {model_path}")
        
        return MultitaskPredictor(model_path, use_gpu=torch.cuda.is_available())
    
    @pytest.fixture(scope="class")
    def current_predictions(self, current_predictor, test_data_path):
        """在测试集上运行当前模型预测"""
        if not Path(test_data_path).exists():
            pytest.skip(f"测试数据不存在: {test_data_path}")
        
        df = pd.read_csv(test_data_path)
        
        # 使用固定的测试子集进行回归测试
        import os
        max_samples = int(os.environ.get("REGRESSION_TEST_SAMPLES", "300"))
        if len(df) > max_samples:
            # 使用固定随机种子确保每次测试使用相同的样本
            df = df.sample(n=max_samples, random_state=42)
        
        texts = df['text'].tolist()
        true_topics = df['topic_label'].tolist()
        true_sentiments = df['sentiment_label'].tolist()
        
        print(f"\n正在对 {len(texts)} 个样本进行回归测试...")
        pred_topics = []
        pred_sentiments = []
        
        for text in texts:
            result = current_predictor.predict(text)
            pred_topics.append(result['topic']['id'])
            pred_sentiments.append(result['sentiment']['id'])
        
        return {
            'true_topics': true_topics,
            'pred_topics': pred_topics,
            'true_sentiments': true_sentiments,
            'pred_sentiments': pred_sentiments
        }
    
    @pytest.fixture(scope="class")
    def current_metrics(self, current_predictions):
        """计算当前模型的指标"""
        return {
            "topic_accuracy": accuracy_score(
                current_predictions['true_topics'],
                current_predictions['pred_topics']
            ),
            "topic_f1": f1_score(
                current_predictions['true_topics'],
                current_predictions['pred_topics'],
                average='macro'
            ),
            "sentiment_accuracy": accuracy_score(
                current_predictions['true_sentiments'],
                current_predictions['pred_sentiments']
            ),
            "sentiment_f1": f1_score(
                current_predictions['true_sentiments'],
                current_predictions['pred_sentiments'],
                average='macro'
            )
        }
    
    def test_topic_accuracy_regression(self, current_metrics, baseline_metrics, model_config_path):
        """测试主题分类准确率没有显著下降"""
        current_acc = current_metrics["topic_accuracy"]
        baseline_acc = baseline_metrics["topic_accuracy"]
        max_drop = model_config_path["regression_thresholds"]["max_accuracy_drop"]
        
        accuracy_drop = baseline_acc - current_acc
        
        print(f"\n主题准确率对比:")
        print(f"  基准模型: {baseline_acc:.4f}")
        print(f"  当前模型: {current_acc:.4f}")
        print(f"  变化: {-accuracy_drop:+.4f}")
        
        assert accuracy_drop <= max_drop, \
            f"主题准确率下降过多: {accuracy_drop:.4f} > {max_drop:.4f}"
    
    def test_topic_f1_regression(self, current_metrics, baseline_metrics, model_config_path):
        """测试主题分类F1没有显著下降"""
        current_f1 = current_metrics["topic_f1"]
        baseline_f1 = baseline_metrics["topic_f1"]
        max_drop = model_config_path["regression_thresholds"]["max_f1_drop"]
        
        f1_drop = baseline_f1 - current_f1
        
        print(f"\n主题F1对比:")
        print(f"  基准模型: {baseline_f1:.4f}")
        print(f"  当前模型: {current_f1:.4f}")
        print(f"  变化: {-f1_drop:+.4f}")
        
        assert f1_drop <= max_drop, \
            f"主题F1下降过多: {f1_drop:.4f} > {max_drop:.4f}"
    
    def test_sentiment_accuracy_regression(self, current_metrics, baseline_metrics, model_config_path):
        """测试情感分析准确率没有显著下降"""
        current_acc = current_metrics["sentiment_accuracy"]
        baseline_acc = baseline_metrics["sentiment_accuracy"]
        max_drop = model_config_path["regression_thresholds"]["max_accuracy_drop"]
        
        accuracy_drop = baseline_acc - current_acc
        
        print(f"\n情感准确率对比:")
        print(f"  基准模型: {baseline_acc:.4f}")
        print(f"  当前模型: {current_acc:.4f}")
        print(f"  变化: {-accuracy_drop:+.4f}")
        
        assert accuracy_drop <= max_drop, \
            f"情感准确率下降过多: {accuracy_drop:.4f} > {max_drop:.4f}"
    
    def test_sentiment_f1_regression(self, current_metrics, baseline_metrics, model_config_path):
        """测试情感分析F1没有显著下降"""
        current_f1 = current_metrics["sentiment_f1"]
        baseline_f1 = baseline_metrics["sentiment_f1"]
        max_drop = model_config_path["regression_thresholds"]["max_f1_drop"]
        
        f1_drop = baseline_f1 - current_f1
        
        print(f"\n情感F1对比:")
        print(f"  基准模型: {baseline_f1:.4f}")
        print(f"  当前模型: {current_f1:.4f}")
        print(f"  变化: {-f1_drop:+.4f}")
        
        assert f1_drop <= max_drop, \
            f"情感F1下降过多: {f1_drop:.4f} > {max_drop:.4f}"
    
    def test_save_current_metrics(self, current_metrics, project_root):
        """保存当前模型指标（用于生成报告）"""
        report_dir = project_root / "tests" / "reports"
        report_dir.mkdir(exist_ok=True)
        
        with open(report_dir / "current_metrics.json", "w") as f:
            json.dump(current_metrics, f, indent=2)
        
        print(f"\n当前指标已保存到: {report_dir / 'current_metrics.json'}")

