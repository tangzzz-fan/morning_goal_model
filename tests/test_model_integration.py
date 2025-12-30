"""
模型集成测试
测试完整的推理pipeline，包括tokenizer、模型加载、预测等
"""
import pytest
import torch
import pandas as pd
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.evaluation.multitask_inference import MultitaskPredictor


class TestInferencePipeline:
    """测试推理pipeline"""
    
    @pytest.fixture(scope="class")
    def predictor(self, model_config_path):
        """加载预测器（在类级别共享以加快测试）"""
        model_path = model_config_path["model_path"]
        if not Path(model_path).exists():
            pytest.skip(f"模型路径不存在: {model_path}")
        
        return MultitaskPredictor(model_path, use_gpu=torch.cuda.is_available())
    
    def test_predictor_initialization(self, predictor):
        """测试预测器能正确初始化"""
        assert predictor is not None
        assert predictor.model is not None
        assert predictor.tokenizer is not None
        assert predictor.device is not None
    
    def test_single_prediction(self, predictor, sample_texts):
        """测试单个文本预测"""
        text = sample_texts[0]
        result = predictor.predict(text)
        
        # 检查返回结果结构
        assert 'text' in result
        assert 'topic' in result
        assert 'sentiment' in result
        
        # 检查主题预测
        assert 'id' in result['topic']
        assert 'label' in result['topic']
        assert 'confidence' in result['topic']
        assert 0 <= result['topic']['id'] < 16
        assert 0 <= result['topic']['confidence'] <= 1
        
        # 检查情感预测
        assert 'id' in result['sentiment']
        assert 'label' in result['sentiment']
        assert 'confidence' in result['sentiment']
        assert 0 <= result['sentiment']['id'] < 3
        assert 0 <= result['sentiment']['confidence'] <= 1
    
    def test_batch_prediction(self, predictor, sample_texts):
        """测试批量预测"""
        results = predictor.predict_batch(sample_texts)
        
        assert len(results) == len(sample_texts)
        
        for result in results:
            assert 'topic' in result
            assert 'sentiment' in result
            assert result['topic']['id'] in range(16)
            assert result['sentiment']['id'] in range(3)
    
    def test_prediction_consistency(self, predictor, sample_texts):
        """测试预测一致性（同样的输入应该得到同样的输出）"""
        text = sample_texts[0]
        
        result1 = predictor.predict(text)
        result2 = predictor.predict(text)
        
        assert result1['topic']['id'] == result2['topic']['id']
        assert result1['sentiment']['id'] == result2['sentiment']['id']
        assert abs(result1['topic']['confidence'] - result2['topic']['confidence']) < 1e-5
        assert abs(result1['sentiment']['confidence'] - result2['sentiment']['confidence']) < 1e-5
    
    def test_empty_input_handling(self, predictor):
        """测试空输入处理"""
        # 测试空字符串
        result = predictor.predict("")
        assert result is not None
        assert 'topic' in result
        assert 'sentiment' in result
    
    def test_long_input_handling(self, predictor):
        """测试长文本输入处理（超过max_length的情况）"""
        long_text = "今天要完成的任务 " * 100  # 创建一个很长的文本
        result = predictor.predict(long_text)
        
        assert result is not None
        assert 'topic' in result
        assert 'sentiment' in result
    
    def test_special_characters_handling(self, predictor):
        """测试特殊字符处理"""
        special_texts = [
            "今天要跑步！！！🏃",
            "完成#项目@报告💪",
            "学习Python？？？",
            "😊😊😊心情很好",
        ]
        
        for text in special_texts:
            result = predictor.predict(text)
            assert result is not None
            assert 'topic' in result
            assert 'sentiment' in result


class TestInferencePerformance:
    """测试推理性能"""
    
    @pytest.fixture(scope="class")
    def predictor(self, model_config_path):
        """加载预测器"""
        model_path = model_config_path["model_path"]
        if not Path(model_path).exists():
            pytest.skip(f"模型路径不存在: {model_path}")
        
        return MultitaskPredictor(model_path, use_gpu=torch.cuda.is_available())
    
    def test_inference_speed(self, predictor, sample_texts, model_config_path):
        """测试推理速度"""
        max_time_ms = model_config_path["quality_thresholds"]["inference_time_max_ms"]
        
        # 预热
        predictor.predict(sample_texts[0])
        
        # 测试单样本推理时间
        times = []
        for text in sample_texts:
            start = time.time()
            predictor.predict(text)
            end = time.time()
            times.append((end - start) * 1000)  # 转换为毫秒
        
        avg_time = sum(times) / len(times)
        print(f"\n平均推理时间: {avg_time:.2f}ms")
        
        # 允许在CPU上运行更慢
        if not torch.cuda.is_available():
            max_time_ms *= 5  # CPU上允许5倍时间
        
        assert avg_time < max_time_ms, f"推理速度过慢: {avg_time:.2f}ms > {max_time_ms}ms"
    
    def test_memory_usage(self, predictor, sample_texts):
        """测试内存使用（确保不会OOM）"""
        # 运行大批量预测
        large_batch = sample_texts * 20
        
        try:
            results = predictor.predict_batch(large_batch)
            assert len(results) == len(large_batch)
        except RuntimeError as e:
            if "out of memory" in str(e):
                pytest.fail("模型推理时内存溢出")
            raise

