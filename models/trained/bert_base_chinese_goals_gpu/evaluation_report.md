# 模型评测报告（bert-base-chinese / Goals-Noise / GPU）

## 数据集概览
- 来源：合成目标文本（移动端场景），16 类主题（工作/健康/家庭/个人发展/理财/社交/家务/学习/睡眠/饮食/心态/娱乐/出行/职业发展/沟通/育儿）
- 训练入口：`src/training/finetune_bert.py:58`
- 生成参数：`count=60000`，`noise_rate=0.35`，`emoji_rate=0.25`，`freeform_rate=0.5`
- 划分规模：`train=42000`，`val=12000`，`test=6000`

## 训练配置
- 模型：`bert-base-chinese`
- 冻结策略：仅微调末 3 层（其余层冻结）
- 超参数：`epochs=5`（早停触发，最佳在第 3 轮），`batch_size=24`，`max_length=128`，`lr=2e-5`，`grad_accum=2`
- 设备：GPU（NVIDIA GeForce RTX 4060 Laptop GPU），`fp16=True`

## 评测结果（GPU）
- 验证集（`models/trained/bert_base_chinese_goals_gpu/metrics_val.json`）
  - `eval_accuracy=0.9779166667`
  - `eval_f1=0.9770214056`
  - `eval_loss=0.0496462397`
  - `eval_runtime=5.3419s`，`eval_steps_per_second=93.601`
- 测试集（`models/trained/bert_base_chinese_goals_gpu/metrics_test.json`）
  - `eval_accuracy=0.977`
  - `eval_f1=0.9761607171`
  - `eval_loss=0.0505046099`
  - `eval_runtime=2.7484s`，`eval_steps_per_second=90.961`

## 与 CPU 基线对比
- CPU 基线（`models/trained/bert_base_chinese_goals_large/metrics_val.json` / `metrics_test.json`）
  - 验证：`acc=0.9766`，`f1=0.9760`，`runtime≈39s`，`steps/s≈3.97`
  - 测试：`acc=0.9768`，`f1=0.9759`，`runtime≈39.6s`，`steps/s≈3.97`
- 提升：
  - 指标：`accuracy/f1` 小幅提升（约 +0.1%）
  - 吞吐：`steps/s` 提升约 23.6×（93.6 vs 3.97），评估时延显著缩短（≈5.34s vs ≈39s）

## 结论与建议
- 结论：GPU + `fp16` 下训练与评估显著加速，模型在噪声合成数据上保持稳定高准确率与宏平均 F1（≈97.7%）。
- 建议：
  - 在更大批次（如 `batch_size=32`）与更长训练（`epochs=5-7`）下进一步验证稳定性
  - 接入真实用户文本（≥10k）进行再训练；随后进入蒸馏/量化/剪枝（M3）以优化端侧性能