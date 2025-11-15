# 模型评测报告（bert-base-chinese / Goals-Noise / GPU-Long）

## 数据集概览
- 来源：移动端目标文本合成，含噪声与自由文本；16 类主题
- 划分规模：train=42000，val=12000，test=6000
- 生成参数：`count=60000`，`noise_rate=0.35`，`emoji_rate=0.25`，`freeform_rate=0.5`

## 训练配置
- 模型：`bert-base-chinese`
- 冻结策略：仅微调末 3 层
- 超参数：`epochs=10`（早停触发，最佳在第 3 轮），`batch_size=28`，`max_length=128`，`lr=2e-5`，`grad_accum=2`
- 设备：GPU（4060 Laptop），`fp16=True`

## 评测结果
- 验证（`models/trained/bert_base_chinese_goals_gpu_long/metrics_val.json`）
  - `eval_accuracy=0.9779166667`
  - `eval_f1=0.9770214056`
  - `eval_loss=0.0495366119`
  - `eval_runtime=4.7461s`，`eval_steps_per_second=90.39`
- 测试（`models/trained/bert_base_chinese_goals_gpu_long/metrics_test.json`）
  - `eval_accuracy=0.977`
  - `eval_f1=0.9761607171`
  - `eval_loss=0.049570296`
  - `eval_runtime=2.5448s`，`eval_steps_per_second=84.487`

## 结论
- 指标与 5 轮 GPU 训练相当（`acc/f1` 稳定）
- 早停于第 3 轮，说明当前数据与配置下更长轮次不会显著提升质量；建议通过更大批次或真实数据接入提升泛化