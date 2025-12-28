# Morning Goal Model (Morning Goal 模型工程)

**Morning Goal** 是一个极简主义的 iOS 目标追踪应用，利用端侧 NLP 模型 (BERT/MobileBERT) 实现用户意图理解和个性化洞察。
本仓库包含该项目的模型训练、优化、评估以及部署到移动端 (Core ML) 的完整工程流水线。

---

## 🚀 快速开始 (Quick Start)

### 1. 环境准备
本项目需要 Python 3.10+ 环境。推荐使用 `venv` 进行管理。

```bash
# 1. 运行设置脚本 (自动创建 venv 并安装依赖)
python3 setup_venv.py

# 2. 激活虚拟环境
source activate_venv.sh

# 3. 验证环境
python3 -c "import torch; print(torch.__version__)"
```

### 2. 运行示例
你可以运行以下脚本来体验模型训练或转换流程：

```bash
# 运行小规模微调示例
python3 src/training/train_small_classifier.py
```

---

## 📂 项目产出与位置 (Key Outputs)

作为一个新人，你最关心的产出物都在这里：

| 产出物类型 | 存放位置 | 说明 |
| :--- | :--- | :--- |
| **训练好的模型** | `models/trained/` | 包含微调后的 BERT (`.pt`) 和蒸馏后的学生模型。 |
| **Core ML 模型** | `models/coreml/` | 最终部署到 iOS 的 `.mlpackage` 或 `.mlmodel` 文件。 |
| **数据集** | `data/processed/` | 处理好的训练集 (`train.csv`), 验证集 (`val.csv`)。 |
| **文档** | `docs/` | 项目的所有技术文档。 |

---

## 📖 核心文档导航 (Documentation)

为了帮助你快速上手并深入理解项目，我们准备了以下核心指南：

### 1. 入门与架构
*   **[技术架构概览](docs/01_architecture/Technical_Overview.md)**: 了解项目的整体设计目标、分层架构以及为何选择端侧 AI 方案。
*   **[架构设计与排查指南](docs/04_guides/Architecture_and_Debugging.md)** (⭐ **新人必读**): 深入理解“基座+可更新”模型架构，以及开发中常见问题的排查思路。

### 2. 模型训练与优化
*   **[模型调优实战指南](docs/04_guides/Model_Optimization_Guide.md)** (⭐ **核心实操**): 如何训练一个更小、更准的模型？包含**知识蒸馏**、**量化**和**剪枝**的详细操作步骤。
*   **[MobileBERT 分析报告](docs/02_training/MobileBERT_Analysis.md)**: 为什么我们选择 MobileBERT？它与学生模型有何不同？

### 3. 移动端部署
*   **[Core ML 转换与集成](docs/04_guides/Technical_Series/02_mobile_conversion.md)**: 如何将 PyTorch 模型转换为 iOS 可用的 Core ML 格式。
*   **[iOS Guides](docs/03_deployment/iOS_Guides/)**: 包含具体的 iOS 集成文档。

---

## 🛠️ 常用开发命令

**1. 知识蒸馏 (Training Student Model):**
```bash
python3 src/training/distill_student.py \
    --teacher_model models/trained/teacher_bert \
    --output_dir models/trained/student_model
```

**2. 导出 Core ML 模型 (Export):**
```bash
python3 src/export/export_coreml.py \
    --onnx_path models/onnx/model.onnx \
    --output_dir models/coreml
```

---

## 👥 贡献与排查
遇到问题？
1. 请首先查阅 **[架构设计与排查指南](docs/04_guides/Architecture_and_Debugging.md)** 中的常见问题部分。
2. 确保你的环境依赖已正确安装 (参考 `requirements.txt`)。
3. 检查 `logs/` 目录下的训练日志。
