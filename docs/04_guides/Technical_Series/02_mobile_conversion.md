# 移动端BERT模型优化与部署系列文档：移动端转换篇

## 1. 概述

完成模型训练和优化后，下一步是将其转换为能在iOS设备上高效运行的格式。PyTorch模型本身无法直接在iOS上使用，需要通过一个或多个中间格式进行转换。本篇将聚焦于将我们优化后的学生模型转换为Apple生态系统中的标准格式——**CoreML**。

**转换路径**: `PyTorch -\u003e ONNX -\u003e CoreML`

1.  **ONNX (Open Neural Network Exchange)**: 作为一个开放的神经网络交换格式，ONNX扮演着关键的“桥梁”角色。大多数主流深度学习框架（如PyTorch, TensorFlow, PaddlePaddle）都支持导出到ONNX。我们将首先把PyTorch模型转换为ONNX格式。
2.  **CoreML**: Apple的官方机器学习框架，能在iOS、macOS等系统上利用硬件加速（CPU, GPU, Neural Engine）实现低延迟、高能效的推理。我们将使用`coremltools`将ONNX模型最终转换为`.mlmodel`或`.mlpackage`格式。

本篇的最终目标是创建一个包含**文本预处理（Tokenizer）**和**模型推理**的**CoreML Pipeline**，使其在iOS应用中可以实现一行代码调用。

## 2. PyTorch到ONNX的转换

这是转换流程的第一步，也是最容易出错的一步。关键在于正确处理模型的动态输入和输出。

### 2.1 理论说明

PyTorch通过`torch.onnx.export`函数支持ONNX导出。该函数会“追踪”一个模型的执行过程，并将计算图记录为ONNX格式。为了让导出的模型能够处理不同长度的句子，必须在导出时指定输入的`dynamic_axes`（动态轴）。

对于BERT模型，输入通常是`input_ids`和`attention_mask`，它们的形状都是`(batch_size, sequence_length)`。我们需要将`batch_size`和`sequence_length`都标记为动态，以适应不同的推理场景。

### 2.2 实操步骤

转换逻辑封装在`src/export/export_coreml.py`脚本中。

**关键代码解析** (`src/export/export_coreml.py`):

```python
# 1. 加载我们训练好的学生模型
model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
model.eval()

# 2. 准备一个符合输入格式的伪输入 (dummy input)
# 形状为 (batch_size=1, sequence_length=128)
dummy_input = {
    "input_ids": torch.zeros(1, 128, dtype=torch.long),
    "attention_mask": torch.zeros(1, 128, dtype=torch.long)
}

# 3. 定义动态轴
dynamic_axes = {
    'input_ids': {0: 'batch_size', 1: 'sequence_length'},
    'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
    'output': {0: 'batch_size'}
}

# 4. 执行导出
torch.onnx.export(
    model,
    (dummy_input['input_ids'], dummy_input['attention_mask']),
    onnx_path, # ONNX模型保存路径
    input_names=['input_ids', 'attention_mask'],
    output_names=['output'],
    dynamic_axes=dynamic_axes,
    opset_version=13 # 一个稳定且被coremltools良好支持的版本
)
```

**执行命令示例**:

```bash
python src/export/export_coreml.py \
    --model_path models/trained/distill_student \
    --output_path models/exported/student_model.mlpackage
```

此命令会首先在同目录下生成一个`student_model.onnx`文件。

## 3. ONNX到CoreML的转换

有了ONNX模型后，我们使用Apple官方提供的`coremltools`库将其转换为CoreML格式。

### 3.1 理论说明

`coremltools`提供了一个强大的转换器，能够将ONNX计算图中的操作（Ops）映射到CoreML支持的层。在转换过程中，我们可以指定一些重要的元数据和配置，例如：

- **输入类型 (Input Types)**: 指定输入是图像、多维数组还是序列。对于BERT，输入是`Tensor`。
- **计算精度 (Compute Precision)**: 允许模型在推理时使用FP16（半精度），以在GPU和Neural Engine上获得更好的性能。
- **分类标签 (Class Labels)**: 对于分类模型，可以直接将分类任务的标签（如“体育”、“财经”）嵌入模型中，这样模型输出的就是可读的标签而不是原始的Logits向量。

### 3.2 实操步骤

**关键代码解析** (`src/export/export_coreml.py`):

```python
import coremltools as ct

# 1. 定义输入
# 注意：形状中的-1表示该维度是动态的
inputs = [
    ct.TensorType(name="input_ids", shape=(1, ct.RangeDim(1, 256)), dtype=np.int32),
    ct.TensorType(name="attention_mask", shape=(1, ct.RangeDim(1, 256)), dtype=np.int32)
]

# 2. 加载分类标签
with open("data/labels.txt", 'r') as f:
    class_labels = [line.strip() for line in f.readlines()]

# 3. 执行转换
mlmodel = ct.convert(
    onnx_path,
    inputs=inputs,
    # 指定输出是分类器，并附上标签
    classifier_config=ct.ClassifierConfig(class_labels),
    # 允许在GPU上使用FP16以加速
    compute_units=ct.ComputeUnit.CPU_AND_GPU,
    minimum_deployment_target=ct.target.iOS15 # 支持Updatable Model的最低版本
)

# 4. 保存为.mlpackage格式
mlmodel.save(args.output_path)
```

`.mlpackage`是CoreML 4引入的新格式，它是一个目录，比旧的`.mlmodel`文件更易于管理和版本控制。

## 4. 构建CoreML Pipeline：集成Tokenizer

到目前为止，我们得到的CoreML模型只接受`input_ids`和`attention_mask`这样的数字输入。但在实际应用中，输入是原始的文本字符串。我们需要将文本预处理步骤（即Tokenizer）也包含进来，构建一个端到端的Pipeline模型。

### 4.1 理论说明

幸运的是，`coremltools`可以直接与`transformers`库中的Tokenizer集成。我们可以创建一个自定义的CoreML模型，它只做一件事：调用`BertTokenizer`。然后，将这个`Tokenizer`模型和我们之前转换的`BERT`模型链接（pipe）起来，形成一个单一的、接受文本输入、输出分类结果的Pipeline模型。

![CoreML Pipeline](https://developer.apple.com/assets/elements/icons/core-ml/core-ml-128x128_2x.png)
*\u003ccenter\u003e图2: CoreML Pipeline示意图\u003c/center\u003e*

### 4.2 实操步骤

**关键代码解析** (`src/export/export_coreml.py`):

```python
from transformers import BertTokenizer

# 1. 加载与模型匹配的Tokenizer
tokenizer = BertTokenizer.from_pretrained(args.model_path)

# 2. 将Tokenizer转换为CoreML模型
# `coremltools`在后台自动处理了这一复杂过程
tokenizer_model = ct.convert(tokenizer)

# 3. 将Tokenizer模型和BERT模型链接起来
pipeline = ct.models.pipeline.Pipeline(
    input_features=[('text', ct.models.datatypes.String())], # Pipeline的输入是字符串
    output_features=[('label', ct.models.datatypes.String())] # Pipeline的输出是分类标签
)

pipeline.add_model(tokenizer_model)
pipeline.add_model(mlmodel)

pipeline.spec.description.input[0].shortDescription = "输入需要分类的中文文本"
pipeline.spec.description.output[0].shortDescription = "预测出的分类标签"

# 4. 更新链接关系
# 将tokenizer的输出连接到bert模型的输入
pipeline.spec.connections.extend([
    ct.proto.Model_pb2.Connection(featureName='input_ids', connectedFeatureName='input_ids'),
    ct.proto.Model_pb2.Connection(featureName='attention_mask', connectedFeatureName='attention_mask')
])

# 5. 保存最终的Pipeline模型
pipeline.save(args.output_path)
```

### 4.3 结果验证

将最终生成的`student_model.mlpackage`拖入Xcode中，可以在“Predictions”标签页下直接测试。输入一段中文文本，模型应该能直接输出预测的分类标签和置信度。

![Xcode验证](https://docs-assets.developer.apple.com/published/072a7f6434/122964-100486-16x9.png)
*\u003ccenter\u003e图3: 在Xcode中预览和验证CoreML模型\u003c/center\u003e*

## 5. 总结

本篇详细介绍了将一个PyTorch BERT模型完整转换为一个即插即用的CoreML Pipeline的全过程。通过`PyTorch -\u003e ONNX -\u003e CoreML`的路径，并集成Tokenizer，我们得到了一个对iOS开发者极其友好的模型文件。

**核心优势**:
- **端到端**: 应用开发者无需关心分词和ID转换，直接输入文本即可。
- **高效**: CoreML能在Apple设备上自动选择最佳计算单元（CPU, GPU, ANE），实现硬件加速。
- **标准化**: `.mlpackage`格式易于版本管理和集成。

下一篇，我们将探讨如何在这个静态模型的基础上，构建一个支持端侧增量学习的**可更新模型架构**。