# 配套资源：示例工程、设备兼容性矩阵、持续集成方案

> 为快速落地提供工程骨架、设备兼容参考与自动化流水线模板。

## 示例工程代码结构（关键路径）

```text
MobileLLMShop/
  models/
    convert_onnx_to_coreml.py      # 转换脚本（ONNX→Core ML，FP16）
    README.md                      # 输入/输出约定与版本
  ios/
    App/                           # Xcode 工程
      Models/                      # .mlmodel 文件
      MLRuntime/                   # 推理封装（Swift）
      Monitoring/                  # 指标采集与上报
  data/
    samples/                       # 校准/验证样本（脱敏）
  docs/
    metrics.md                     # 指标定义与阈值
```

### 转换脚本关键片段
```python
# models/convert_onnx_to_coreml.py
import coremltools as ct
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--onnx', required=True)
parser.add_argument('--out', default='Model_FP16.mlmodel')
args = parser.parse_args()

mlmodel = ct.convert(
    args.onnx, source='onnx', convert_to='mlprogram',
    compute_units=ct.ComputeUnit.ALL,
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.iOS18,
)
mlmodel.save(args.out)
print('saved:', args.out)
```

## 设备兼容性矩阵（参考）

| 设备系列 | 芯片 | 内存（典型） | Neural Engine | 推荐模型规模 | 说明 |
|---|---|---|---|---|---|
| iPhone 12/13 | A14/A15 | 4–6 GB | 有 | ≤1B | FP16 优先，短上下文 |
| iPhone 14 | A16 | 6 GB | 有 | ≤2B | 更快 NE，注意内存峰值 |
| iPhone 15 | A17 Pro | 8 GB | 有 | ≤3B | 端侧 LLM 可用性最佳 |
| iPad Pro (M1/M2) | M1/M2 | 8–16 GB | 强 | ≤7B | 更大模型与上下文 |

> 注：实际可承载规模受算子支持与上下文长度影响。建议以 FP16 + 结构化剪枝为主，慎用 INT8。

## 持续集成方案（GitHub Actions 示例）

### 工作流 1：模型转换与产物上传
```yaml
name: model-convert
on: [workflow_dispatch]
jobs:
  convert:
    runs-on: macos-14
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.10' }
      - name: Install deps
        run: |
          python -m venv .venv && source .venv/bin/activate
          pip install coremltools onnx
      - name: Convert ONNX to Core ML
        run: |
          source .venv/bin/activate
          python models/convert_onnx_to_coreml.py --onnx model.onnx --out ios/App/Models/Model_FP16.mlmodel
      - uses: actions/upload-artifact@v4
        with:
          name: coreml-model
          path: ios/App/Models/Model_FP16.mlmodel
```

### 工作流 2：iOS 构建与基本验证
```yaml
name: ios-build
on: [workflow_dispatch]
jobs:
  build:
    runs-on: macos-14
    steps:
      - uses: actions/checkout@v4
      - name: Xcode build
        run: |
          xcodebuild -project ios/App/App.xcodeproj -scheme App -sdk iphoneos -configuration Release build | xcpretty
      - name: Unit tests
        run: |
          xcodebuild -project ios/App/App.xcodeproj -scheme AppTests -sdk iphonesimulator -destination 'platform=iOS Simulator,name=iPhone 15' test | xcpretty
```

### 指标与告警
- 在 `Monitoring/` 采集 `model_load_ms`、`first_inference_ms`、`p50_ms/p90_ms`、`peak_mem_mb`，定期上报。
- 设定阈值与降级策略：超阈值自动缩短上下文或切换基线推荐。