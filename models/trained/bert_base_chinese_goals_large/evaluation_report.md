# 模型评测报告（bert-base-chinese / Goals-Noise）

## 数据集概览
- 来源：合成目标文本（移动端场景），16 类主题（工作/健康/家庭/个人发展/理财/社交/家务/学习/睡眠/饮食/心态/娱乐/出行/职业发展/沟通/育儿）
- 生成参数：`count=60000`，`noise_rate=0.35`，`emoji_rate=0.25`，`freeform_rate=0.5`
- 划分规模：`train=42000`，`val=12000`，`test=6000`
- 类别分布（近似均衡）：
  - train：每类约 2499–2733 条
  - val：每类约 714–781 条
  - test：每类约 357–391 条

## 训练配置
- 模型：`bert-base-chinese`
- 冻结策略：仅微调末 3 层（其余层冻结）
- 超参数：`epochs=3`，`batch_size=32`，`max_length=128`，`lr=2e-5`
- 设备：CPU（自动检测 GPU，当前未启用；`fp16=False`）
- 脚本入口：`src/training/finetune_bert.py:58`

## 评测结果
- 验证集（文件：`models/trained/bert_base_chinese_goals_large/metrics_val.json`）
  - `eval_accuracy=0.9766`
  - `eval_f1=0.9760`
  - `eval_loss=0.0448`
- 测试集（文件：`models/trained/bert_base_chinese_goals_large/metrics_test.json`）
  - `eval_accuracy=0.9768`
  - `eval_f1=0.9759`
  - `eval_loss=0.0444`

## 结论与建议
- 结论：在含噪声与自由文本的移动端合成数据上，`bert-base-chinese` 达到稳定的高准确率与宏平均 F1（≈97.6%），说明模板与噪声增强对鲁棒性有效。
- 建议：
  - 增量扩充真实用户文本（≥10k），保持 `text,label` 格式与分层划分一致
  - 在 GPU 环境下提高批次与轮次进行全面训练；随后进行蒸馏与量化（M3）
  - 记录并对比不同噪声参数与类别集合对指标的影响，优化生成策略