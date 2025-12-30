"""
Pytest配置和共享fixtures
"""
import pytest
import torch
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.multitask_bert import MultitaskBertForClassification


@pytest.fixture(scope="session")
def project_root():
    """返回项目根目录"""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_path(project_root):
    """返回测试数据路径"""
    return project_root / "data" / "processed" / "test_multitask.csv"


@pytest.fixture(scope="session")
def model_config_path(project_root):
    """返回模型配置路径"""
    config_file = project_root / "tests" / "test_config.json"
    if config_file.exists():
        with open(config_file) as f:
            return json.load(f)
    # 默认配置
    return {
        "model_path": str(project_root / "models" / "trained" / "distill_student"),
        "quality_thresholds": {
            "topic_accuracy_min": 0.60,
            "topic_f1_min": 0.55,
            "sentiment_accuracy_min": 0.70,
            "sentiment_f1_min": 0.65,
            "inference_time_max_ms": 100  # 单样本推理时间上限
        },
        "regression_thresholds": {
            "max_accuracy_drop": 0.05,  # 最大准确率下降5%
            "max_f1_drop": 0.05
        }
    }


@pytest.fixture(scope="session")
def device():
    """返回计算设备"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def sample_texts():
    """返回测试文本样本"""
    return [
        "今天要跑步5公里",
        "完成项目报告",
        "学习Python编程",
        "和家人一起吃晚饭",
        "阅读一本好书",
    ]


@pytest.fixture
def mock_model_inputs():
    """返回模拟的模型输入"""
    return {
        "input_ids": torch.randint(0, 1000, (2, 128)),
        "attention_mask": torch.ones(2, 128, dtype=torch.long),
        "topic_labels": torch.tensor([0, 1]),
        "sentiment_labels": torch.tensor([1, 2])
    }

