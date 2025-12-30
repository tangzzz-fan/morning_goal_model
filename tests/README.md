# 模型质量测试指南

本目录包含对MobileBERT多任务模型的全面测试套件，用于确保模型质量和防止性能退化。

## 📋 目录结构

```
tests/
├── __init__.py                    # 测试模块初始化
├── conftest.py                    # Pytest配置和共享fixtures
├── test_model_unit.py             # 单元测试：模型架构和组件
├── test_model_integration.py      # 集成测试：推理pipeline
├── test_model_quality.py          # 质量测试：性能指标验证
├── test_model_regression.py       # 回归测试：对比基准模型
├── test_config.json               # 测试配置（阈值、路径等）
├── baseline_metrics.json          # 基准模型性能指标
└── reports/                       # 测试报告输出目录
```

## 🧪 测试层级

### 1. 单元测试 (Unit Tests)
**文件**: `test_model_unit.py`

测试模型的基本功能和架构：
- ✅ 模型初始化
- ✅ 前向传播输出形状
- ✅ 损失计算
- ✅ 梯度反向传播
- ✅ 模型保存和加载

**运行**:
```bash
pytest tests/test_model_unit.py -v
```

### 2. 集成测试 (Integration Tests)
**文件**: `test_model_integration.py`

测试完整的推理pipeline：
- ✅ 预测器初始化
- ✅ 单样本预测
- ✅ 批量预测
- ✅ 预测一致性
- ✅ 特殊输入处理（空文本、长文本、特殊字符）
- ✅ 推理速度
- ✅ 内存使用

**运行**:
```bash
pytest tests/test_model_integration.py -v
```

### 3. 质量测试 (Quality Tests)
**文件**: `test_model_quality.py`

验证模型性能指标是否达到预设阈值：
- ✅ 主题分类准确率（≥60%）
- ✅ 主题分类F1分数（≥55%）
- ✅ 情感分析准确率（≥70%）
- ✅ 情感分析F1分数（≥65%）
- ✅ 置信度分布
- ✅ 标签分布

**运行**:
```bash
pytest tests/test_model_quality.py -v
# 可以通过环境变量控制测试样本数
TEST_MAX_SAMPLES=1000 pytest tests/test_model_quality.py -v
```

### 4. 回归测试 (Regression Tests)
**文件**: `test_model_regression.py`

确保新模型不比基准模型差：
- ✅ 主题准确率不显著下降（<5%）
- ✅ 主题F1不显著下降（<5%）
- ✅ 情感准确率不显著下降（<5%）
- ✅ 情感F1不显著下降（<5%）

**运行**:
```bash
pytest tests/test_model_regression.py -v
```

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
pip install pytest pytest-html pytest-cov pytest-xdist
```

### 2. 运行所有测试
```bash
# 使用便捷脚本
bash scripts/run_tests.sh

# 或直接使用pytest
pytest tests/ -v
```

### 3. 运行特定测试
```bash
# 只运行单元测试
bash scripts/run_tests.sh -t unit

# 只运行质量测试
bash scripts/run_tests.sh -t quality -s 1000

# 测试特定模型
bash scripts/run_tests.sh -m models/trained/new_model
```

## ⚙️ 配置

### 测试配置文件 (`test_config.json`)

```json
{
  "model_path": "models/trained/distill_student",
  "quality_thresholds": {
    "topic_accuracy_min": 0.60,
    "topic_f1_min": 0.55,
    "sentiment_accuracy_min": 0.70,
    "sentiment_f1_min": 0.65,
    "inference_time_max_ms": 100
  },
  "regression_thresholds": {
    "max_accuracy_drop": 0.05,
    "max_f1_drop": 0.05
  }
}
```

### 修改质量阈值

编辑 `test_config.json` 文件，调整 `quality_thresholds` 中的值。

### 更新基准模型

当有一个新的、性能更好的模型时：

```bash
# 1. 运行测试生成当前指标
bash scripts/run_tests.sh -t quality

# 2. 更新基准
python scripts/update_baseline.py --model-version "v2.0"

# 3. 验证回归测试通过
bash scripts/run_tests.sh -t regression
```

## 📊 查看测试报告

测试运行后，报告保存在以下位置：

1. **HTML报告**: `test_reports/*.html`
   - 打开浏览器查看详细的测试结果

2. **JSON报告**: `tests/reports/*.json`
   - 包含详细的性能指标和分类报告

3. **覆盖率报告**: `htmlcov/index.html`
   - 显示代码测试覆盖率

## 🔄 GitHub Actions 集成

项目已配置GitHub Actions自动化测试：

### 1. 质量检查工作流 (`.github/workflows/model_quality_test.yml`)

**触发条件**:
- 推送到 `main` 或 `develop` 分支
- 创建Pull Request
- 手动触发

**功能**:
- 运行所有测试
- 生成测试报告
- 上传覆盖率到Codecov
- 在PR中评论测试结果

### 2. 训练后测试工作流 (`.github/workflows/model_training_test.yml`)

**触发条件**:
- 手动触发

**功能**:
- 运行模型训练
- 自动进行质量检查
- 验证模型是否达标
- 保存模型artifact

### 手动触发测试

在GitHub仓库页面：
1. 进入 **Actions** 标签页
2. 选择 **Model Quality Test** 工作流
3. 点击 **Run workflow**
4. 选择测试级别和参数
5. 点击 **Run workflow** 开始

## 🛠️ 开发指南

### 添加新的测试

1. 在适当的测试文件中添加测试函数
2. 使用pytest的fixture获取共享资源
3. 添加适当的标记（markers）

示例：
```python
import pytest

@pytest.mark.unit
def test_new_feature(predictor):
    """测试新功能"""
    result = predictor.predict("测试文本")
    assert result is not None
```

### 添加新的fixture

在 `conftest.py` 中添加：
```python
@pytest.fixture
def my_fixture():
    """描述"""
    # 设置代码
    yield resource
    # 清理代码
```

### 使用标记运行特定测试

```bash
# 只运行单元测试
pytest -m unit

# 只运行不需要GPU的测试
pytest -m "not gpu"

# 运行快速测试（排除慢测试）
pytest -m "not slow"
```

## 📈 性能基准

当前基准模型性能（更新于：见 `baseline_metrics.json`）：

| 指标 | 目标值 | 当前值 |
|------|--------|--------|
| 主题准确率 | ≥60% | 65% |
| 主题F1 | ≥55% | 60% |
| 情感准确率 | ≥70% | 75% |
| 情感F1 | ≥65% | 70% |
| 推理时间 | ≤100ms | ~50ms |

## 🐛 故障排查

### 测试失败：模型路径不存在

**问题**: `Model not found at models/trained/distill_student`

**解决**:
1. 确认模型已训练并保存在正确位置
2. 或修改 `test_config.json` 中的 `model_path`

### 测试失败：内存不足

**问题**: `RuntimeError: CUDA out of memory`

**解决**:
1. 减少测试样本数：`TEST_MAX_SAMPLES=100 pytest ...`
2. 使用CPU模式：在代码中设置 `use_gpu=False`

### 质量测试不通过

**问题**: 模型性能低于阈值

**解决**:
1. 检查训练过程是否正常
2. 分析错误预测样本
3. 考虑调整模型或训练策略
4. 如果是合理的性能，可以调整阈值

## 📚 相关文档

- [开发流程指南](../docs/04_guides/Dev_Pipeline_Guide.md)
- [模型优化指南](../docs/04_guides/Model_Optimization_Guide.md)
- [多任务模型指南](../docs/04_guides/Multitask_Model_Guide.md)

## 🤝 贡献

添加新测试或改进现有测试时，请确保：
1. 测试有清晰的文档字符串
2. 测试是确定性的（可重复）
3. 测试失败时有清晰的错误信息
4. 更新此README文档

## 📝 许可

与主项目保持一致。

