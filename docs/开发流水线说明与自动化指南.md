# 开发流水线说明与自动化指南

## 概述
- 目标：端侧 NLP（中文）分类模型的训练、优化、导出与移动端集成准备
- 范围：M1 环境与数据、M2 微调、M3 优化（蒸馏/量化/剪枝）、M4 导出与CoreML转换准备

## 环境与依赖
- 虚拟环境创建：`py -3 -m venv venv`
- 关键依赖安装：`torch/transformers/datasets/accelerate`、`onnxruntime` 等
- GPU 校验与启用：已安装 `torch 2.5.1+cu121`，`CUDA 12.1`（检测见 `environment_setup.py:139-149`）

## 数据准备（M1）
- 合成数据生成：`src/data/generate_goals.py:1`
  - 类目扩充与噪声注入：`src/data/generate_goals.py:19-107`、`src/data/generate_goals.py:110-126`
  - 生成命令示例：`python -m src.data.generate_goals --count 60000 --noise_rate 0.35 --emoji_rate 0.25 --freeform_rate 0.5 --output_dir data\processed`
- 划分输出：`data/processed/train.csv`、`val.csv`、`test.csv`

## 微调训练（M2）
- 训练入口：`src/training/finetune_bert.py:58`
  - 冻结末3层：`src/training/finetune_bert.py:37-47`
  - GPU 自适应与 `fp16`：`src/training/finetune_bert.py:75`、`src/training/finetune_bert.py:111-124`
  - Windows DataLoader 适配：`DictDataset` 顶层定义 `src/training/finetune_bert.py:36`、`dataloader_num_workers=0` `src/training/finetune_bert.py:122`
  - 命令示例（GPU）：`python -m src.training.finetune_bert --data_dir data\processed --epochs 5 --batch_size 24 --max_length 128 --output_dir models\trained\bert_base_chinese_goals_gpu --limit_train 42000 --limit_eval 12000 --grad_accum 2`
- 指标输出：`metrics_val.json`、`metrics_test.json`
- 报告：`models/trained/bert_base_chinese_goals_gpu/evaluation_report.md`
- 对比报告汇总：`models/trained/comparison_report.md:14-23`

## 模型优化（M3）
- 蒸馏训练：`src/training/distill_student.py:82`
  - 自定义 `DistillTrainer.compute_loss`：`src/training/distill_student.py:66-79`
  - 教师设备迁移：`src/training/distill_student.py:63`
  - 学生/教师使用 `safetensors` 加载：`src/training/distill_student.py:123-130`
  - 保存学生模型：`src/training/distill_student.py:163-168`
  - 命令：`python -m src.training.distill_student --data_dir data\processed --teacher_name bert-base-chinese --student_name uer/chinese_roberta_L-4_H-512 --epochs 3 --batch_size 24 --temperature 3.0 --alpha 0.5 --output_dir models\trained\distill_student --grad_accum 1`
- 动态量化：`src/optimization/quantize.py:45`
  - 命令：`python -m src.optimization.quantize --model_dir models\trained\distill_student --data_dir data\processed --output_dir models\optimized\quant_int8_student`
- 剪枝：`src/optimization/prune.py:47`
  - 命令：`python -m src.optimization.prune --model_dir models\trained\distill_student --data_dir data\processed --output_dir models\optimized\pruned_student --amount 0.2`
- M3 总结：`models/trained/M3_summary.md`

## 导出与CoreML转换准备（M4）
- ONNX 导出与一致性评估：`src/export/export_onnx.py:1`
  - 默认 opset 14，命令：`python -m src.export.export_onnx --model_dir models\trained\distill_student --data_dir data\processed --output_dir models\onnx --opset 14 --max_length 128 --samples 64`
  - 报告：`models/onnx/onnx_evaluation_report.md`
- CoreML 转换脚本（macOS/iOS 环境）：`src/export/export_coreml.py:1`
  - 草案文档：`models/coreml/coreml_conversion_plan.md`

## 特征抽取与可更新分类器准备（M4 延伸）
- 特征抽取器 ONNX（仅输出嵌入）：`src/export/export_feature_extractor_onnx.py:1`
  - 命令：`python -m src.export.export_feature_extractor_onnx --model_dir models\trained\distill_student --data_dir data\processed --output_dir models\onnx --opset 14 --max_length 128 --samples 64`
  - 报告：`models/onnx/feature_onnx_evaluation_report.md`
- 小型分类器训练（参考端侧可更新初始权重）：`src/training/train_small_classifier.py:1`
  - 命令：`python -m src.training.train_small_classifier --model_dir models\trained\distill_student --data_dir data\processed --output_dir models\optimized\classifier_baseline --max_length 128`
  - 输出：`classifier.joblib` 与 `classifier_baseline_report.json`

## 一键脚本扩展
- 脚本：`tools/run_pipeline.ps1`
  - 新增步骤：`feature_extractor`、`classifier`
  - 配置：`configs/params.json`（新增 `feature_extractor` 与 `classifier` 段）

## 自动化现状
- 训练/评测：均有脚本入口（训练、蒸馏、量化、剪枝、导出、评估），支持命令行复现
- 文档：对比报告、评测报告与M3总结齐备；本指南提供整体串联
- 待补充建议：
  - 增加 Makefile/PowerShell 脚本聚合多个步骤一键执行（训练→评测→导出→报告）
  - 在 CI 中跑轻量一致性测试（ONNX metrics），自动生成报告
  - 将参数（批次/轮次/噪声比例）集中配置到 `configs/` 并在脚本读取

## 调整与优化建议
- 数据：接入真实数据（≥10k），完善标注规范与分层采样
- 训练：在 GPU 上验证更大批次与超参网格，联合早停
- 优化：结构化剪枝与蒸馏温度/alpha 搜索，提高性能/精度权衡
- 导出：在 macOS/iOS 环境验证 CoreML 精度差异与端侧性能