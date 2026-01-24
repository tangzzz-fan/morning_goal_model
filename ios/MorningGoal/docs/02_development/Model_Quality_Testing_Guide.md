# MobileBERT 模型质量测试指南

## 1. 测试概述

本文档介绍如何对导入的 `student_sequence_classification.mlpackage` 进行质量测试,确保模型在实际应用中的表现符合预期。

## 2. 现有测试基础设施

项目中已经包含以下测试组件:

### 2.1 核心测试服务
- **`ModelQualityValidator`**: 提供完整的模型验证功能
  - 准确率 (Accuracy)
  - 精确率 (Precision)
  - 召回率 (Recall)
  - F1 分数
  - 置信度分布
  - 推理时间
  - 内存使用

- **`ModelBench`**: 性能基准测试
  - 平均推理时间
  - P90 延迟
  - 分类准确率
  - 情感分析准确率
  - 内存占用

- **`MockDataService`**: 测试数据生成
  - 创建模拟目标数据
  - 设置连续记录
  - 生成历史数据

## 3. 测试方法

### 方法 1: 使用 ModelBench 快速测试

最简单的方法是使用现有的 `ModelBench.run()`:

```swift
import SwiftUI

struct ModelTestView: View {
    @State private var testResult: BenchSummary?
    @State private var isTesting = false
    
    var body: some View {
        VStack(spacing: 20) {
            Text("模型质量测试")
                .font(.title)
            
            if isTesting {
                ProgressView("测试中...")
            } else {
                Button("开始测试") {
                    runTest()
                }
                .buttonStyle(.borderedProminent)
            }
            
            if let result = testResult {
                VStack(alignment: .leading, spacing: 10) {
                    Text("测试结果").font(.headline)
                    Text("平均推理时间: \(String(format: "%.2f", result.avg))ms")
                    Text("P90 延迟: \(String(format: "%.2f", result.p90))ms")
                    Text("分类准确率: \(String(format: "%.1f", result.catAcc * 100))%")
                    Text("情感准确率: \(String(format: "%.1f", result.senAcc * 100))%")
                    Text("内存使用: \(String(format: "%.2f", result.memMB))MB")
                }
                .padding()
                .background(Color.gray.opacity(0.1))
                .cornerRadius(10)
            }
        }
        .padding()
    }
    
    private func runTest() {
        isTesting = true
        Task {
            let context = DataController.shared.container.viewContext
            let result = await ModelBench.run(in: context)
            await MainActor.run {
                testResult = result
                isTesting = false
            }
        }
    }
}
```

### 方法 2: 使用 ModelQualityValidator 详细测试

如果需要更详细的指标:

```swift
struct DetailedModelTestView: View {
    @State private var validationResult: ModelQualityValidator.ValidationResult?
    @State private var isTesting = false
    
    var body: some View {
        VStack(spacing: 20) {
            Text("详细模型验证")
                .font(.title)
            
            if isTesting {
                ProgressView("验证中...")
            } else {
                Button("开始验证") {
                    runValidation()
                }
                .buttonStyle(.borderedProminent)
            }
            
            if let result = validationResult {
                ScrollView {
                    VStack(alignment: .leading, spacing: 12) {
                        MetricRow(label: "准确率", value: result.accuracy)
                        MetricRow(label: "精确率", value: result.precision)
                        MetricRow(label: "召回率", value: result.recall)
                        MetricRow(label: "F1 分数", value: result.f1Score)
                        
                        Divider()
                        
                        Text("推理时间: \(String(format: "%.2f", result.inferenceTime))ms")
                        Text("内存使用: \(String(format: "%.2f", result.memoryUsage))MB")
                        
                        Divider()
                        
                        Text("置信度分布")
                            .font(.headline)
                        Text("平均: \(String(format: "%.3f", result.confidenceDistribution.reduce(0, +) / Double(result.confidenceDistribution.count)))")
                    }
                    .padding()
                }
            }
        }
        .padding()
    }
    
    private func runValidation() {
        isTesting = true
        Task {
            let validator = ModelQualityValidator()
            let service = try! AdaptiveModelService()
            let testData = createTestSamples()
            
            let result = await validator.validateModel(service, with: testData)
            
            await MainActor.run {
                validationResult = result
                isTesting = false
            }
        }
    }
    
    private func createTestSamples() -> [TestSample] {
        return [
            TestSample(text: "今天要完成项目开发", expectedCategory: "工作", expectedSentiment: "中性", difficulty: .easy),
            TestSample(text: "晚上去健身房锻炼", expectedCategory: "健康", expectedSentiment: "积极", difficulty: .easy),
            // ... 添加更多测试样本
        ]
    }
}

struct MetricRow: View {
    let label: String
    let value: Double
    
    var body: some View {
        HStack {
            Text(label)
            Spacer()
            Text(String(format: "%.1f%%", value * 100))
                .foregroundColor(colorForValue(value))
        }
    }
    
    private func colorForValue(_ value: Double) -> Color {
        if value >= 0.9 { return .green }
        if value >= 0.8 { return .orange }
        return .red
    }
}
```

## 4. 测试标准

### 4.1 性能指标
- **推理时间**: 应 < 100ms (理想 < 50ms)
- **P90 延迟**: 应 < 150ms
- **内存使用**: 应 < 100MB

### 4.2 准确率指标
- **分类准确率**: 应 >= 80%
- **情感准确率**: 应 >= 75%
- **平均置信度**: 应 >= 70%
- **低置信度率**: 应 < 30%

### 4.3 质量评级
根据 `ModelQualityValidator.assessQuality()`:
- **优秀**: 得分 >= 90
- **良好**: 得分 >= 80
- **一般**: 得分 >= 70
- **需改进**: 得分 < 70

## 5. 测试流程

### 步骤 1: 准备测试数据
使用 `ModelBench.createTestSamples()` 中的 24 个标准测试样本,覆盖:
- 8 个分类类别
- 3 种情感倾向
- 不同难度级别

### 步骤 2: 运行基准测试
```swift
let context = DataController.shared.container.viewContext
let result = await ModelBench.run(in: context)
```

### 步骤 3: 分析结果
检查日志输出:
```
bench_complete avg=45.23 p50=42.10 p90=67.89
bench_accuracy cat_acc=0.875 sen_acc=0.833 low_conf=0.125
bench_resources mem_mb=67.45 total_tokens=3456
```

### 步骤 4: 质量评估
如果发现问题:
- **准确率低**: 检查训练数据质量,考虑重新训练
- **推理慢**: 检查模型是否使用了 Neural Engine
- **内存高**: 考虑模型量化或优化

## 6. 集成到应用

### 6.1 添加测试入口
在开发模式下添加测试按钮:

```swift
#if DEBUG
Button("测试模型质量") {
    showModelTest = true
}
.sheet(isPresented: $showModelTest) {
    ModelTestView()
}
#endif
```

### 6.2 持续监控
在生产环境中记录关键指标:
```swift
logger.log("inference_time=\(inferenceTime) confidence=\(confidence)")
```

## 7. 故障排查

### 问题 1: 模型加载失败
- 检查 `student_sequence_classification.mlpackage` 是否在 Bundle 中
- 查看 Xcode 控制台错误信息

### 问题 2: 准确率异常低
- 验证测试数据的标签是否正确
- 检查模型输出的类别名称是否匹配

### 问题 3: 推理时间过长
- 确认 `MLModelConfiguration.computeUnits` 设置为 `.cpuAndNeuralEngine`
- 检查是否在主线程运行推理

## 8. 下一步

测试通过后:
1. 记录基线指标
2. 在真实数据上验证
3. 开始端侧训练实验
4. 监控训练后的模型质量变化
