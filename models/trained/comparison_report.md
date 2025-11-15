# 训练对比报告（CPU vs GPU）

## 数据与任务
- 数据来源：移动端目标文本合成，含噪声与自由文本
- 类别：16 类（工作/健康/家庭/个人发展/理财/社交/家务/学习/睡眠/饮食/心态/娱乐/出行/职业发展/沟通/育儿）
- 划分：train=42000，val=12000，test=6000

## 训练脚本与参数
- 入口：`src/training/finetune_bert.py:58`
- 冻结策略：仅微调末 3 层（`src/training/finetune_bert.py:37`）
- GPU 自适应与梯度累积：`fp16` 与 `--grad_accum`（`src/training/finetune_bert.py:71`，`src/training/finetune_bert.py:111-124`）
- DataLoader 适配（Windows）：`DictDataset` 顶层定义、`dataloader_num_workers=0`（`src/training/finetune_bert.py:36`、`src/training/finetune_bert.py:122`）

## 指标对比
- CPU（目录：`models/trained/bert_base_chinese_goals_large`）
  - 验证：acc=0.9766，f1=0.9760，loss=0.0448267，eval_runtime=38.9604s，steps/s=4.03
  - 测试：acc=0.9768，f1=0.9759138，loss=0.0444429，eval_runtime=39.5849s，steps/s=3.966
- GPU（目录：`models/trained/bert_base_chinese_goals_gpu`）
  - 验证：acc=0.9779167，f1=0.9770214，loss=0.0496462，eval_runtime=5.3419s，steps/s=93.601
  - 测试：acc=0.9770，f1=0.9761607，loss=0.0505046，eval_runtime=2.7484s，steps/s=90.961
- GPU-Long（目录：`models/trained/bert_base_chinese_goals_gpu_long`）
  - 验证：acc=0.9779167，f1=0.9770214，loss=0.0495366，eval_runtime=4.7461s，steps/s=90.39
  - 测试：acc=0.9770，f1=0.9761607，loss=0.0495703，eval_runtime=2.5448s，steps/s=84.487

## 结论
- 吞吐提升显著（评估 `steps/s` 提升 ≈ 23.6×），时延大幅下降
- 指标小幅提升（acc/f1 ≈ +0.1%），在含噪声与自由文本的场景下保持稳定

## 说明
- 更大批次与更长轮次的 GPU 重训结果已输出至 `models/trained/bert_base_chinese_goals_gpu_long`；早停于第 3 轮，指标与 5 轮训练相当。