ONNX / Core ML 转换与脚本

目标
- 将蒸馏/量化后的学生模型导出为 ONNX 或 Core ML，以适配移动端推理。

ONNX 导出（Optimum CLI）
```bash
pip install "optimum[onnxruntime]" transformers onnx onnxruntime onnxruntime-tools coremltools

optimum-cli export onnx \
  --model bert-base-chinese \
  --task sequence-classification \
  --opset 13 \
  --output ./artifacts/onnx_seqcls
```

ONNX 导出（PyTorch）
```python
import torch
from transformers import BertForSequenceClassification, AutoTokenizer

SEQ_LEN = 128
model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=NUM_LABELS).eval()
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

dummy = tokenizer("示例文本", return_tensors="pt", padding='max_length', max_length=SEQ_LEN)

torch.onnx.export(
    model,
    (dummy["input_ids"], dummy["attention_mask"], dummy.get("token_type_ids", torch.zeros_like(dummy["input_ids"]))),
    "model.onnx",
    input_names=["input_ids", "attention_mask", "token_type_ids"],
    output_names=["logits"],
    dynamic_axes={
      "input_ids": {0: "batch", 1: "seq"},
      "attention_mask": {0: "batch", 1: "seq"},
      "token_type_ids": {0: "batch", 1: "seq"},
      "logits": {0: "batch"}
    },
    opset_version=13
)
```

ONNX 量化与校验
```python
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort

quantize_dynamic("model.onnx", "model.int8.onnx", weight_type=QuantType.QInt8)

session = ort.InferenceSession("model.int8.onnx", providers=["CPUExecutionProvider"])
```

Core ML 转换（从 ONNX）
```python
import coremltools as ct

mlmodel = ct.converters.onnx.convert(
    model="model.int8.onnx",
    minimum_deployment_target=ct.target.iOS15,
)

# FP16 权重量化（减少体积与带宽）
from coremltools.models.neural_network.quantization_utils import quantize_weights
mlmodel_fp16 = quantize_weights(mlmodel, nbits=16)

mlmodel_fp16.save("BERTClassifier.mlmodel")
```

Core ML 转换（从 TorchScript → mlprogram）
```python
import torch, coremltools as ct
from transformers import BertForSequenceClassification, AutoTokenizer

SEQ_LEN = 128
model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=NUM_LABELS).eval()
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
dummy = tokenizer("示例文本", return_tensors="pt", padding='max_length', max_length=SEQ_LEN)

ts = torch.jit.trace(model, (dummy["input_ids"], dummy["attention_mask"], dummy.get("token_type_ids", torch.zeros_like(dummy["input_ids"]))))

mlmodel = ct.convert(
    ts,
    inputs=[
        ct.TensorType(name="input_ids", shape=(1, SEQ_LEN), dtype=ct.int32),
        ct.TensorType(name="attention_mask", shape=(1, SEQ_LEN), dtype=ct.int32),
        ct.TensorType(name="token_type_ids", shape=(1, SEQ_LEN), dtype=ct.int32),
    ],
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.iOS15,
    compute_units=ct.ComputeUnit.ALL,
)

mlmodel.save("BERTClassifier.mlmodel")
```

转换脚本编写要点
- 明确输入/输出名称与形状（`input_ids/attention_mask/token_type_ids`）。
- 使用 `dynamic_axes` 保持序列长度可伸缩；移动端可固定长度以简化内存管理。
- Core ML 推荐 `mlprogram` 格式；配置 `ComputeUnit.ALL` 以启用 CPU/GPU/ANE。
- 通过基准集校验精度与延迟，输出报告与阈值告警。