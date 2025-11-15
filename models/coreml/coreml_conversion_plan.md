# CoreML 转换流程草案

- 输入：ONNX 模型（ops: 14），路径示例：`models/onnx/student_sequence_classification.onnx`
- 转换代码入口：`src/export/export_coreml.py`
- 步骤：
  - 使用 `coremltools.converters.onnx.convert(onnx_path)` 进行转换
  - 保存到 `models/coreml/student_sequence_classification.mlpackage`
- 验证：
  - 在 macOS/iOS 环境加载 `.mlpackage`，运行验证样本，确保精度差异 < 3%
  - 对比 CPU/GPU/ONNX/CoreML 一致性报告，记录性能（时延/内存）与精度
- 优化：
  - 应用 CoreML 优化（如量化/压缩），在满足性能的前提下控制精度损失（FP16 <1%，INT8 <3%）
- 交付：
  - 输出转换日志、验证报告与 `.mlpackage`（不纳入仓库），供 iOS 集成使用