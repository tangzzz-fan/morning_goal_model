# 🚀 模型质量测试快速开始

## 📌 TL;DR (太长不看版)

```bash
# 运行所有测试
bash scripts/run_tests.sh

# 运行特定测试
bash scripts/run_tests.sh -t quality  # 质量测试
bash scripts/run_tests.sh -t unit     # 单元测试

# 查看报告
open test_reports/all_tests_report.html
```

## 🎯 5分钟快速上手

### 步骤1: 安装依赖（首次使用）

```bash
# 激活虚拟环境
source venv/bin/activate

# 安装测试工具
pip install pytest pytest-html pytest-cov
```

### 步骤2: 运行你的第一个测试

```bash
# 运行单元测试（不需要训练好的模型）
pytest tests/test_model_unit.py -v
```

预期输出：
```
✅ test_model_initialization PASSED
✅ test_model_forward_shape PASSED
✅ test_model_loss_computation PASSED
...
====== 6 passed in 12.5s ======
```

### 步骤3: 测试你的模型

```bash
# 方法1: 使用便捷脚本（推荐）
bash scripts/run_tests.sh -t quality

# 方法2: 直接使用pytest
pytest tests/test_model_quality.py -v
```

### 步骤4: 查看测试报告

```bash
# 在浏览器中打开HTML报告
open test_reports/quality_test_report.html

# 或查看JSON报告
cat tests/reports/current_metrics.json
```

## 📊 测试说明

### 测试层级

| 命令 | 测试内容 | 需要模型 | 运行时间 |
|------|----------|----------|----------|
| `-t unit` | 模型架构和组件 | ❌ 否 | 快 (~30s) |
| `-t integration` | 完整推理流程 | ✅ 是 | 中 (~2min) |
| `-t quality` | 性能指标验证 | ✅ 是 | 慢 (~5min) |
| `-t regression` | 对比基准模型 | ✅ 是 | 慢 (~5min) |
| `-t all` | 所有测试 | ✅ 是 | 最慢 (~10min) |

### 质量标准

你的模型需要达到以下标准才能通过质量测试：

- ✅ 主题分类准确率 ≥ 60%
- ✅ 主题分类F1 ≥ 55%
- ✅ 情感分析准确率 ≥ 70%
- ✅ 情感分析F1 ≥ 65%
- ✅ 单样本推理时间 ≤ 100ms

可以在 `tests/test_config.json` 中修改这些阈值。

## 🔧 常见使用场景

### 场景1: 开发时快速验证

```bash
# 修改代码后快速测试
pytest tests/test_model_unit.py -x  # -x: 第一个失败就停止

# 只运行快速测试
pytest -m "not slow"
```

### 场景2: 训练新模型后验证

```bash
# 1. 更新配置指向新模型
vim tests/test_config.json  # 修改 model_path

# 2. 运行质量测试
bash scripts/run_tests.sh -t quality -s 1000

# 3. 如果通过，运行回归测试
bash scripts/run_tests.sh -t regression

# 4. 如果新模型更好，更新基准
python scripts/update_baseline.py --model-version "v2.0"
```

### 场景3: 准备发布前的完整测试

```bash
# 运行所有测试并生成完整报告
bash scripts/run_tests.sh -v

# 检查覆盖率
open htmlcov/index.html

# 确认所有指标达标
cat tests/reports/current_metrics.json
```

### 场景4: 只测试特定模型

```bash
# 不修改配置文件，临时测试其他模型
bash scripts/run_tests.sh \
  -m models/trained/new_model \
  -t quality \
  -s 500
```

## 🐛 常见问题速查

### Q: 找不到模型文件

```bash
# 检查模型是否存在
ls -la models/trained/distill_student

# 如果不存在，训练模型或修改配置
vim tests/test_config.json
```

### Q: 测试太慢

```bash
# 减少测试样本数
bash scripts/run_tests.sh -t quality -s 100

# 只运行快速测试
pytest -m "not slow"
```

### Q: 质量测试不通过

```bash
# 查看详细报告找出问题
cat tests/reports/quality_test_report.json | python -m json.tool

# 临时降低阈值用于调试（不要提交）
vim tests/test_config.json
```

### Q: 内存不足

```bash
# 减少测试样本
TEST_MAX_SAMPLES=100 pytest tests/test_model_quality.py

# 或使用CPU模式（在conftest.py中设置）
```

## 🎨 高级用法

### 并行运行测试

```bash
# 安装pytest-xdist
pip install pytest-xdist

# 自动并行
pytest tests/ -n auto
```

### 只运行某些测试

```bash
# 使用关键字过滤
pytest -k "accuracy" -v  # 只运行包含accuracy的测试

# 使用标记
pytest -m unit -v  # 只运行单元测试

# 测试特定类
pytest tests/test_model_quality.py::TestModelQuality -v
```

### 生成不同格式的报告

```bash
# HTML报告
pytest --html=report.html --self-contained-html

# JUnit XML（用于CI）
pytest --junitxml=junit.xml

# 覆盖率报告
pytest --cov=src --cov-report=html --cov-report=term
```

## 📚 下一步

- 📖 阅读 [完整测试指南](docs/04_guides/Model_Testing_Guide.md)
- 📖 阅读 [测试目录README](tests/README.md)
- 🔧 配置 [GitHub Actions](.github/workflows/model_quality_test.yml)
- 🎯 自定义测试阈值 (`tests/test_config.json`)

## 💡 提示

1. **开发时先跑单元测试** - 快速验证基本功能
2. **提交前跑完整测试** - 确保没有破坏现有功能
3. **训练新模型后跑质量测试** - 验证性能达标
4. **定期更新基准** - 保持测试标准与最佳模型一致
5. **查看报告细节** - 不仅看通过/失败，还要分析具体指标

## 🎉 完成！

现在你已经掌握了基本的测试流程。开始测试你的模型吧！

有问题？查看 [完整文档](docs/04_guides/Model_Testing_Guide.md) 或提交Issue。

