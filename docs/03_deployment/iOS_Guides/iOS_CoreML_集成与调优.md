# iOS 平台集成方案：Core ML 工具链与调优实操

> 覆盖环境搭建、从 PyTorch/TensorFlow 到 Core ML 的完整转换、Xcode 集成、性能调优与实时性测试指标。包含可复现命令与代码。

## 第三章：iOS 平台集成方案

### 3.1 环境搭建
- 前置条件：macOS（Xcode ≥ 15）、Python ≥ 3.9、`coremltools` ≥ 7。建议先启用网络代理。
- 操作步骤：
  ```bash
  proxy_on
  /usr/bin/python3 -m venv .venv && source .venv/bin/activate
  pip install --upgrade pip
  pip install coremltools onnx onnxruntime torch torchvision tensorflow
  ```
- 预期结果验证：`python -c "import coremltools as ct; print(ct.__version__)"` 输出版本号。
- 常见问题与解决：
  - 无法连接 PyPI：确认 `proxy_on` 已启用，或切换国内镜像源。

### 3.2 模型转换流程（PyTorch → Core ML）
- 前置条件：可 trace 或导出 ONNX 的 PyTorch 模型；明确输入形状与 dtype。
- 操作步骤 A（JIT trace 直转）：
  ```python
  import coremltools as ct
  import torch
  import torch.nn as nn

  class TinyClassifier(nn.Module):
      def __init__(self, in_dim=512, num_classes=10):
          super().__init__()
          self.fc = nn.Linear(in_dim, num_classes)
      def forward(self, x):
          return self.fc(x)

  model = TinyClassifier().eval()
  example = torch.randn(1, 512)
  traced = torch.jit.trace(model, example)

  mlmodel = ct.convert(
      traced,
      inputs=[ct.TensorType(name="input", shape=example.shape)],
      convert_to="mlprogram",
      compute_units=ct.ComputeUnit.ALL,
      compute_precision=ct.precision.FLOAT16,
      minimum_deployment_target=ct.target.iOS18,
  )
  mlmodel.save("TinyClassifier_FP16.mlmodel")
  ```
- 操作步骤 B（ONNX 中转，适配复杂网络）：
  ```python
  import torch
  import coremltools as ct

  model = ... # 复杂 Transformer 或 CNN
  model.eval()
  dummy = torch.randn(1, 3, 224, 224)
  torch.onnx.export(
      model, dummy, "model.onnx", opset_version=13,
      input_names=["input"], output_names=["logits"],
      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}
  )

  mlmodel = ct.convert(
      "model.onnx", source="onnx",
      convert_to="mlprogram",
      compute_units=ct.ComputeUnit.ALL,
      compute_precision=ct.precision.FLOAT16,
      minimum_deployment_target=ct.target.iOS18,
  )
  mlmodel.save("Model_FP16.mlmodel")
  ```
- 预期结果验证：比较 `.mlmodel` 与原模型在同一输入上的输出均值误差（MSE/MAE）是否在可接受范围。
- 常见问题与解决：
  - 某些算子不支持：限制到 opset 13 并使用等价替换层；或简化动态形状。

### 3.3 模型转换流程（TensorFlow/Keras → Core ML）
- 前置条件：Keras 模型可导出 SavedModel；输入形状明确（NHWC）。
- 操作步骤：
  ```python
  import tensorflow as tf
  import coremltools as ct

  model = tf.keras.applications.MobileNetV2(weights=None, input_shape=(224,224,3), classes=10)

  mlmodel = ct.convert(
      model, source="tensorflow", convert_to="mlprogram",
      inputs=[ct.TensorType(name="input", shape=(1,224,224,3))],
      compute_units=ct.ComputeUnit.ALL,
      compute_precision=ct.precision.FLOAT16,
      minimum_deployment_target=ct.target.iOS18,
  )
  mlmodel.save("KerasMobileNetV2_FP16.mlmodel")
  ```
- 预期结果验证：随机输入上对齐输出分布；真实数据集上精度差异不超过业务阈值。
- 常见问题与解决：
  - 自定义层：导出前替换为标准算子，或编写等价前后处理在 Swift 端实现。

### 3.4 Xcode 集成与端侧推理
- 前置条件：Xcode 项目，`.mlmodel` 已生成并加入工程；iOS 设备连接。
- 操作步骤：
  ```swift
  import CoreML

  // 配置计算单元，优先使用 Neural Engine
  let config = MLModelConfiguration()
  config.computeUnits = .all

  // 自动生成的 Swift 接口类，名称与模型文件对应
  let model = try TinyClassifier_FP16(configuration: config)

  // 准备输入，注意形状与 dtype（float32/float16 由框架内部管理）
  var inputArray = try MLMultiArray(shape: [512], dataType: .float32)
  // ... 填充输入

  let input = TinyClassifier_FP16Input(input: inputArray)
  let start = CACurrentMediaTime()
  let output = try model.prediction(input: input)
  let latency = CACurrentMediaTime() - start

  print("latency(s):", latency)
  // 解析输出
  let logits = output.logits
  ```
- 预期结果验证：在真机上运行，记录 p50/p90 延迟、内存峰值，结果稳定且 UI 无卡顿。
- 常见问题与解决：
  - 模型加载耗时：首次启动延迟可通过懒加载或后台预热解决。
  - 内存峰值过高：减少 batch、缩短序列长度、采用 FP16。

### 3.5 性能调优技巧
- 输入约束：缩短序列长度与图像分辨率；避免超大上下文窗口。
- 运算绑定：优先 `.all`/`.cpuAndGPU`，实际以设备 NE/GPU 可用性为准。
- 预/后处理：在 Swift 侧实现轻量化 Tokenizer 与归一化，减少 Python 侧开销。
- 算子替换：使用 Core ML 友好层（Conv/Linear/Activation），避免稀有算子。

### 3.6 实时性测试指标与采集
- 指标：
  - `model_load_ms`、`first_inference_ms`、`p50_ms/p90_ms`、`peak_mem_mb`、`battery_drop_%`。
- 采集示例（Swift）：
  ```swift
  import os.signpost

  let log = OSLog(subsystem: "com.company.app", category: "ml")
  let signpostID = OSSignpostID(log: log)

  os_signpost(.begin, log: log, name: "inference", signpostID: signpostID)
  let output = try model.prediction(input: input)
  os_signpost(.end, log: log, name: "inference", signpostID: signpostID)
  ```
- 验证：在多设备（见兼容矩阵）实测；设阈值告警与降级策略（缩短上下文、降采样）。

### 3.7 Core ML 特别说明（LLM 场景）
- 大模型直转受限：优先选择蒸馏后的小模型（≤3B），或拆分为判别式子模型 + 轻量生成头。
- 精度方案：以 FP16 为主；INT8 需谨慎（算子与数值稳定性）。
- 流式生成：可分步调用与增量缓存，以近似 tokens/s 显示；必要时在端侧实现轻量解码循环。