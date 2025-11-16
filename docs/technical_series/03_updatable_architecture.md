# 移动端BERT模型优化与部署系列文档：可更新架构篇

## 1. 概述

传统的移动端AI模型是静态的：一次训练，永久使用，直到下次App更新替换整个模型。这种模式无法让模型适应每个用户的独特偏好和数据，也无法从用户的使用行为中学习和进化。为了打破这一局限，Apple在CoreML 3中引入了**可更新模型（Updatable Models）**的概念。

本篇将详细阐述如何利用这一特性，设计并实现一个“**基座模型 + 增量更新**”的混合架构，使我们的BERT模型能够在iOS设备上进行个性化、低功耗的自学习。

**核心目标**：
- **个性化体验**：让模型根据每个用户的具体使用数据进行微调，提供千人千面的智能服务。
- **隐私保护**：所有训练数据和模型更新都保留在用户本地，不上传云端，最大化保护用户隐私。
- **低成本迭代**：避免了频繁地为全体用户重新训练和下发大型基座模型，节约了云端计算成本和用户网络带宽。

## 2. 混合架构设计

我们设计的架构包含两个核心部分，其关系如下图所示：

![混合架构](https://developer.apple.com/documentation/coreml/images/updatable_model_architecture.png)
*\u003ccenter\u003e图4: 基座模型 + 增量更新的混合架构示意图\u003c/center\u003e*

1.  **基座模型 (Base Model)**
    -   即我们在前几章中通过**蒸馏+量化**得到的、高度优化的静态模型 (`student_model.mlpackage`)。
    -   它体积小、性能高，包含了从海量数据中学到的通用语言知识。
    -   这个模型随App打包分发，构成了所有用户体验的起点和基础。
    -   **它的权重是固定的，在端侧训练中不会被改变**。

2.  **增量模型 (Incremental Model / Update Head)**
    -   这是一个非常轻量级的、附加在基座模型之上的“头部”网络。通常只包含一到两个全连接层。
    -   **只有这个增量模型的参数是可训练的**。
    -   它在转换时被标记为`updatable`，CoreML会在端侧训练时专门更新这部分的权重。
    -   每次更新后，这个增量模型被独立保存，与基座模型共同构成一个完整的、个性化的模型。

**工作流程**：
1.  **推理时**：输入数据首先流经固定的基座模型，提取出高级特征（Feature Vector）。然后，这些特征被送入个性化的增量模型，产出最终的预测结果。
2.  **训练时**：当收集到足够的用户反馈数据后，App会启动一个`MLUpdateTask`。在这个任务中，只有增量模型的权重会根据新的数据进行调整，而基座模型全程处于冻结状态。由于增量模型极小，整个训练过程非常快速且耗电量低。

## 3. 创建可更新的CoreML模型

要实现上述架构，关键在于模型转换阶段的正确配置。

### 3.1 理论说明

在使用`coremltools`进行转换时，我们需要：
1.  将模型的某些层（我们选择最后的全连接分类层）标记为`updatable`。
2.  定义损失函数（Loss Function）、优化器（Optimizer）和训练时的输入输出。
3.  将模型编译为支持更新的格式。

### 3.2 实操步骤

我们在`src/export/export_coreml.py`脚本的基础上增加可更新配置。

**关键代码解析** (`src/export/export_coreml.py` - 更新版):

```python
# ...接上一篇的转换代码...

# 1. 将模型标记为可更新
mlmodel.is_updatable = True

# 2. 指定可训练的层 (我们只更新最后的分类器)
# 'classifier'是PyTorch模型中最后一层的名字
mlmodel.specification.trainingParameters.updatableModel.specificationVersion = 5 # 支持iOS 15+
updatable_layer = mlmodel.specification.neuralNetwork.layers[-1]
updatable_layer.isUpdatable = True

# 3. 定义损失函数和优化器
loss_layer = mlmodel.specification.trainingParameters.updatableModel.lossLayers.add()
loss_layer.name = "lossLayer"
loss_layer.categoricalCrossEntropyLoss.input = "output" # 模型输出
loss_layer.categoricalCrossEntropyLoss.target = "label_true" # 训练时提供的真实标签

optimizer = mlmodel.specification.trainingParameters.updatableModel.optimizer
optimizer.sgdOptimizer.learningRate.defaultValue = 0.01
optimizer.sgdOptimizer.miniBatchSize.defaultValue = 8

# 4. 定义训练输入
mlmodel.specification.trainingParameters.trainingInputs.add(name="input_ids", type=ct.proto.FeatureTypes_pb2.FeatureType.multiArrayType)
mlmodel.specification.trainingParameters.trainingInputs.add(name="attention_mask", type=ct.proto.FeatureTypes_pb2.FeatureType.multiArrayType)
mlmodel.specification.trainingParameters.trainingInputs.add(name="label_true", type=ct.proto.FeatureTypes_pb2.FeatureType.stringType)

# ...保存模型...
```

## 4. iOS端实现模型更新

在iOS App中，模型更新的核心是`MLUpdateTask`。

### 4.1 数据准备

首先，你需要一个机制来收集用户的反馈数据。例如，当用户发现一个分类错误时，可以提供一个“纠错”按钮。你需要将这些纠错样本（文本和正确标签）保存在本地数据库中。

当数据积累到一定数量（如50条）时，就可以触发一次更新任务。你需要将这些数据转换为`MLFeatureProvider`格式。

### 4.2 实操步骤 (Swift)

以下是在Swift中执行模型更新的示例代码：

```swift
import CoreML
import Combine

class ModelUpdater {
    func updateModel(modelURL: URL, trainingData: [Sample]) -\u003e AnyPublisher\u003cMLUpdateContext, Error\u003e {
        
        // 1. 将你的数据转换为MLBatchProvider
        let batchProvider = TrainingBatchProvider(samples: trainingData)
        
        // 2. 创建一个MLUpdateTask
        guard let updateTask = try? MLUpdateTask(forModelAt: modelURL, trainingData: batchProvider, configuration: nil) else {
            fatalError("无法创建更新任务")
        }
        
        // 3. 使用Combine的Publisher来监听更新进度
        let publisher = updateTask.publisher()
        
        // 4. 启动任务
        // 你可以订阅publisher来获取进度、完成状态或错误
        // 在任务完成后，CoreML会自动将更新后的模型保存在一个临时位置
        // 你需要将其移动到永久位置以供后续使用
        publisher
            .handleEvents(receiveCompletion: { completion in
                if case .finished = completion {
                    let updatedModelURL = updateTask.model.url
                    // 将 updatedModelURL 移动到你的App沙盒中的永久位置
                    self.saveUpdatedModel(from: updatedModelURL)
                }
            })
            .sink(receiveCompletion: { _ in }, receiveValue: { context in
                // 监控训练进度
                print("模型更新进度: \(context.event), 指标: \(context.metrics)")
            })
        
        return publisher
    }
    
    // ... TrainingBatchProvider 和 saveUpdatedModel 的实现 ...
}
```

### 4.3 版本管理

由于模型会不断更新，良好的版本管理至关重要：
- **保存更新**: 每次更新成功后，将新生成的模型文件（如`student_model_v2.mlpackage`）保存起来。
- **加载最新**: App启动时，总是检查并加载最新版本的模型。
- **回滚机制**: 如果更新后的模型出现性能下降或其他问题，应能方便地回滚到上一个稳定版本。
- **与基座模型的关系**: 保存的只是增量模型。推理时，需要同时加载基座模型和最新的增量模型。

## 5. 总结与权衡

可更新模型架构为移动端AI带来了前所未有的灵活性和个性化能力。

**优势**:
- **极致个性化**：模型能够持续学习，越来越懂用户。
- **强大的隐私保护**：数据不出设备，打消用户顾虑。
- **高效迭代**：训练开销极小，可在设备空闲和充电时自动完成。

**挑战与权衡**:
- **过拟合风险**：由于只在少量用户数据上训练，增量模型有可能会对特定样本过拟合。需要设计合理的训练策略和数据增强来缓解。
- **“灾难性遗忘”**：持续的更新可能会让模型忘记一些通用知识。虽然我们通过冻结基座模型在很大程度上避免了这个问题，但仍需监控。
- **复杂的工程实现**：相比静态模型，可更新模型需要更复杂的App端逻辑来管理数据、训练任务和模型版本。

下一篇，我们将讨论如何构建一个全面的验证体系，以确保从云端到端侧的整个模型生命周期的质量。