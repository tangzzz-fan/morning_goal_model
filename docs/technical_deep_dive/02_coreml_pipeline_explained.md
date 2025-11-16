# 技术深潜：Core ML Pipeline功能详解

## 1. 概述

在移动端部署NLP模型时，开发者面临一个普遍的痛点：模型本身只接受数字（如`input_ids`），而App的输入是原始的文本字符串。这意味着开发者需要在App代码中手动实现一套复杂的文本预处理逻辑（分词、ID化、Padding等），这不仅繁琐、容易出错，而且通常是用Swift/Objective-C实现的，性能远不如C++底层优化的原生库。

**Core ML Pipeline** 正是为解决这一问题而生的。它允许我们将多个独立的Core ML模型“串联”起来，形成一个单一的、端到端的模型。对于外部调用者（iOS App）而言，它看起来就像一个黑盒子：输入原始文本，直接输出最终的预测结果。

本篇将深入解析Pipeline在我们的文本分类任务中的具体构成、工作流程和配置技巧。

## 2. Pipeline的组成与工作流程

我们的文本分类Pipeline由三个核心组件（实际上是两个模型和一个配置）串联而成：

![Pipeline流程图](https://i.imgur.com/your_pipeline_flowchart.png) \u003c!-- 此处应有流程图 --\u003e
*\u003ccenter\u003e图2: 文本分类Pipeline工作流程\u003c/center\u003e*

### 组件1：Tokenizer模型 (输入预处理)

-   **作用**: 负责将输入的原始字符串转换为BERT模型能理解的数字输入。
-   **输入**: 一个字符串，例如 `"今天天气真好"`。
-   **输出**: 两个多维数组（`MLMultiArray`）：
    -   `input_ids`: 文本被分词后，每个token在词汇表中的对应ID。
    -   `attention_mask`: 一个与`input_ids`等长的二进制数组，用于标识哪些是真实的token（值为1），哪些是用于补齐长度的padding token（值为0）。
-   **实现**: `coremltools`可以直接将`transformers`库中的`BertTokenizer`对象转换为一个CoreML模型。这个过程是自动化的，`coremltools`在后台将分词、ID化等逻辑打包成一个原生模型。

    ```python
    # src/export/export_coreml.py
    from transformers import BertTokenizer
    import coremltools as ct

    tokenizer = BertTokenizer.from_pretrained("uer/chinese_roberta_L-4_H-512")
    # 这一步就创建了一个只包含预处理逻辑的CoreML模型
    tokenizer_model = ct.convert(tokenizer)
    ```

### 组件2：BERT模型 (模型推理)

-   **作用**: Pipeline的核心，负责执行主要的神经网络计算。
-   **输入**: `input_ids` 和 `attention_mask`，这两个输入直接来自于上一步Tokenizer模型的输出。
-   **输出**: 一个多维数组，通常名为`logits`或`output`。它是一个长度等于分类任务类别数的向量，每个值代表模型预测输入文本属于对应类别的原始分数。
-   **实现**: 这就是我们通过`PyTorch -\u003e ONNX -\u003e CoreML`路径转换得到的主模型。

### 组件3：分类器配置 (输出后处理)

-   **作用**: 负责将BERT模型输出的原始、无意义的`logits`向量转换为人类可读的、有意义的分类结果。
-   **输入**: `logits`向量。
-   **输出**: 
    -   `label`: 置信度最高的分类标签，是一个字符串，例如`"体育"`。
    -   `probabilities`: 一个字典，包含了所有类别及其对应的置信度（经过Softmax计算后的概率值）。
-   **实现**: 这并非一个独立的模型，而是在`coremltools`转换时附加到BERT模型上的一个`ClassifierConfig`。它告诉CoreML这个模型是一个分类器，并提供了类别标签。

    ```python
    # src/export/export_coreml.py
    # 提供类别标签
    class_labels = ["体育", "财经", ...]

    classifier_config = ct.ClassifierConfig(class_labels)

    bert_model = ct.convert(
        onnx_path,
        classifier_config=classifier_config, # 在此附加配置
        ...
    )
    ```

## 3. 构建Pipeline与参数调优

`coremltools`提供了`Pipeline`类来将上述组件“粘合”在一起。

**关键代码解析** (`src/export/export_coreml.py`):

```python
# 1. 初始化一个空的Pipeline
# 定义Pipeline的最终输入和输出
pipeline = ct.models.pipeline.Pipeline(
    input_features=[('text', ct.models.datatypes.String())], # 输入是字符串
    output_features=[('label', ct.models.datatypes.String())]  # 输出是字符串标签
)

# 2. 按顺序添加模型
pipeline.add_model(tokenizer_model)
pipeline.add_model(bert_model)

# 3. 定义连接关系 (最关键的一步)
# 将tokenizer的输出连接到bert模型的输入
pipeline.spec.connections.extend([
    ct.proto.Model_pb2.Connection(
        # bert_model的输入'input_ids'，其内容来自于...
        featureName='input_ids', 
        # ...tokenizer_model的输出'input_ids'
        connectedFeatureName='input_ids' 
    ),
    ct.proto.Model_pb2.Connection(
        featureName='attention_mask', 
        connectedFeatureName='attention_mask'
    )
])

# 4. 保存Pipeline模型
pipeline.save("TextClassifier.mlpackage")
```

### 性能调优建议

1.  **输入长度 (`ct.RangeDim`)**: 在转换BERT模型时，务必将输入Tensor的序列长度维度定义为`ct.RangeDim`，而不是一个固定的值。这允许Pipeline接受任意长度的文本输入，模型会在内部进行动态的截断或补齐。

    ```python
    # 定义一个长度在1到256之间动态变化的维度
    seq_len_dim = ct.RangeDim(1, 256)
    inputs = [ct.TensorType(name="input_ids", shape=(1, seq_len_dim), ...)]
    ```

2.  **Tokenizer的词表大小**: `BertTokenizer`转换时会把整个词汇表（`vocab.txt`）打包进模型。对于中文BERT，这个词表通常包含约2万个汉字。如果你的任务场景非常有限，可以考虑裁剪词汇表，移除一些生僻字，这能略微减小Tokenizer模型的大小。

3.  **计算精度 (`compute_units`)**: 如上一篇文档所述，对于BERT这类模型，`ct.ComputeUnit.CPU_AND_GPU`是兼顾性能和稳定性的最佳选择。它能确保大部分计算在GPU上高效执行。

4.  **模型融合**: `coremltools`和底层的CoreML运行时会自动进行一定程度的算子融合（Operator Fusion）。在转换日志中，你可以关注`coremltools`是否成功地将一些连续的操作（如`Conv+BN+ReLU`）融合成单一的、更高效的底层实现。如果融合失败，可能需要调整PyTorch层的实现方式。

## 4. 总结

Core ML Pipeline是将复杂NLP模型交付给App开发者的最佳方式。它的价值在于**封装复杂性**。

**对App开发者的优势**:
-   **接口极简**: 无需关心分词、ID化、Tensor操作，只需调用`model.prediction(text: "...")`即可。
-   **代码健壮**: 将复杂的预处理逻辑从易变的Swift代码转移到稳定、高效的原生CoreML模型中，减少了潜在的bug。
-   **性能更佳**: 由`coremltools`生成的原生Tokenizer通常比在Swift中手动实现要快得多。

**对AI工程师的优势**:
-   **责任分离**: AI工程师可以专注于模型本身的优化，而无需过多介入App端的工程细节。
-   **端到端交付**: 提供一个即插即用的`.mlpackage`文件，是更专业、更完整的交付产物。

通过精心设计和配置Pipeline，我们可以构建出对开发者友好、对用户高效的移动端AI功能。