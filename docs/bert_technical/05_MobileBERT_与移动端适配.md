MobileBERT 与移动端适配实践

简介
- MobileBERT 通过瓶颈结构与层重排在保持精度的同时显著降低计算量。
- 中文场景：可用 `bert-base-chinese` 作为教师，蒸馏到 MobileBERT/TinyBERT 学生结构。

结构要点
- 瓶颈层（inverted bottleneck）；减少宽度，保留深度。
- 注意力头减少与投影维度缩窄；Feed-Forward 层做低秩近似。

训练策略
- 教师-学生蒸馏：Logits + 中间层特征对齐；温度平滑。
- 数据增强：同义替换/随机遮蔽；固定最长序列长度，提升蒸馏稳定性。

部署建议
- iOS：Core ML `mlprogram` + FP16 权重；固定 `seq_len=128/256`。
- Android：ONNX Runtime Mobile；优先动态量化线性层。

性能目标（参考）
- A14：`seq_len=128`，分类头 P50 延迟 10–20ms；`seq_len=256` 20–35ms。
- 包体：FP16 权重 20–40MB；INT8 权重更小但需校准与精度评估。

验证与监控
- 统一延迟/内存/精度基准；版本灰度与指标看板；超阈即回退。