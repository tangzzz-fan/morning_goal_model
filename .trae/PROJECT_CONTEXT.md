# Morning Goal Model - 项目上下文锚点 (Project Context Anchor)

> **⚠️ 重要**：此文件是项目的“短期记忆”与“状态快照”。每次对话开始时请阅读此文件，结束任务时请更新此文件。

## 1. 项目概览 (Project Overview)
*   **项目名称**: Morning Goal Model
*   **核心目标**: 为 iOS 极简目标应用开发端侧 NLP 模型 (Core ML)，实现用户意图识别（分类+情感）。
*   **当前版本**: v0.5 (Model Optimization Phase)
*   **技术栈**:
    *   **Training**: PyTorch, HuggingFace Transformers (BERT/MobileBERT), Distillation
    *   **Deployment**: Core ML, Quantization (INT8/FP16)
    *   **Infrastructure**: Python 3.10+, venv

## 2. 当前状态 (Current Status)
*   **里程碑**: [Milestone 4: CoreML 转换] & [Milestone 5: iOS 端侧适配]
*   **最近进展**:
    *   [2026-01-22] 清理了大量冗余文档与过时计划，精简了项目结构。
    *   [2026-01-22] 建立了 `.trae/` 下的工程实践规范。
    *   [Previous] 完成了 BERT 模型微调与蒸馏，初步产出了 Core ML 模型。

## 3. 活跃任务 (Active Tasks)
*   **[进行中]** Core ML 模型优化与验证
    *   目标：确保转换后的 `.mlpackage` 在 iOS 上推理速度 < 100ms，精度损失 < 3%。
    *   待办：检查 `export_coreml.py` 的配置，优化 FP16/INT8 量化策略。
*   **[待开始]** iOS 推理引擎集成
    *   目标：编写 Swift `ModelWrapper`，实现后台推理。

## 4. 关键决策记录 (Key Decisions)
*   **模型架构**: 采用 **Multitask Learning** (Topic Classification + Sentiment Analysis) 共享底层 Encoder。
*   **部署格式**: Core ML (`.mlpackage`)。
*   **优化策略**: 优先使用 FP16 量化平衡精度与速度；若体积过大再考虑 INT8。

## 5. 已知问题与阻碍 (Known Issues)
*   *暂无活跃阻碍 (No active blockers)*

## 6. 环境备忘 (Environment Memo)
*   **Venv**: `source activate_venv.sh`
*   **Test Script**: `bash scripts/run_tests.sh`
