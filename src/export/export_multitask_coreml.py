"""
Export Multitask Model to CoreML
导出多任务模型到CoreML格式
"""
import argparse
from pathlib import Path
import torch
import coremltools as ct
from transformers import AutoTokenizer, AutoConfig
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.multitask_bert import MultitaskBertForClassification


class MultitaskWrapper(torch.nn.Module):
    """
    包装多任务模型以返回两个输出
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, input_ids, attention_mask):
        """
        前向传播
        
        Returns:
            tuple: (topic_logits, sentiment_logits)
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return outputs['topic_logits'], outputs['sentiment_logits']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_dir",
        required=True,
        help="Path to the trained multitask model directory",
    )
    ap.add_argument(
        "--output_dir",
        default="models/coreml",
        help="Directory to save the CoreML model",
    )
    ap.add_argument(
        "--seq_len",
        type=int,
        default=128,
        help="Sequence length for tracing",
    )
    ap.add_argument(
        "--num_topic_labels",
        type=int,
        default=16,
        help="Number of topic labels",
    )
    ap.add_argument(
        "--num_sentiment_labels",
        type=int,
        default=3,
        help="Number of sentiment labels",
    )
    args = ap.parse_args()

    model_path = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading multitask model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    
    model = MultitaskBertForClassification.from_pretrained(
        model_path,
        config=config,
        num_topic_labels=args.num_topic_labels,
        num_sentiment_labels=args.num_sentiment_labels,
    )
    model.eval()

    # 包装模型
    wrapper_model = MultitaskWrapper(model)
    wrapper_model.eval()

    # 创建示例输入
    print("Tracing model...")
    text = "这是一个测试句子"
    inputs = tokenizer(
        text,
        max_length=args.seq_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    
    # Trace the model
    dummy_input = (inputs["input_ids"], inputs["attention_mask"])
    
    with torch.no_grad():
        traced_model = torch.jit.trace(wrapper_model, dummy_input)
        
        # 验证traced model
        traced_output = traced_model(*dummy_input)
        print(f"Traced model output shapes:")
        print(f"  Topic logits: {traced_output[0].shape}")
        print(f"  Sentiment logits: {traced_output[1].shape}")

    # 定义CoreML输入类型
    input_tensors = [
        ct.TensorType(name="input_ids", shape=(1, args.seq_len), dtype=int),
        ct.TensorType(name="attention_mask", shape=(1, args.seq_len), dtype=int),
    ]

    print("Converting to CoreML...")
    
    # 获取主题标签
    topic_id2label = model.config.id2label if hasattr(model.config, 'id2label') else None
    if topic_id2label:
        topic_labels = [topic_id2label[i] for i in sorted(topic_id2label.keys())]
    else:
        topic_labels = [f"topic_{i}" for i in range(args.num_topic_labels)]
    
    # 情感标签
    sentiment_labels = ["消极", "中性", "积极"]
    
    print(f"Topic labels ({len(topic_labels)}): {topic_labels}")
    print(f"Sentiment labels ({len(sentiment_labels)}): {sentiment_labels}")

    # 转换为CoreML
    mlmodel = ct.convert(
        traced_model,
        inputs=input_tensors,
        outputs=[
            ct.TensorType(name="topic_logits", dtype=float),
            ct.TensorType(name="sentiment_logits", dtype=float),
        ],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS15,
        compute_units=ct.ComputeUnit.ALL,
    )
    
    # 添加元数据
    mlmodel.author = "Morning Goal Team"
    mlmodel.license = "MIT"
    mlmodel.short_description = "多任务模型：主题分类 + 情感分析"
    mlmodel.version = "1.0.0"
    
    # 添加输入描述
    mlmodel.input_description["input_ids"] = "Tokenized input IDs (shape: 1 x 128)"
    mlmodel.input_description["attention_mask"] = "Attention mask (shape: 1 x 128)"
    
    # 添加输出描述
    mlmodel.output_description["topic_logits"] = f"Topic classification logits ({args.num_topic_labels} classes)"
    mlmodel.output_description["sentiment_logits"] = f"Sentiment analysis logits ({args.num_sentiment_labels} classes: 消极/中性/积极)"
    
    # 保存模型
    output_path = output_dir / "multitask_model.mlpackage"
    print(f"Saving to {output_path}...")
    mlmodel.save(str(output_path))
    
    # 保存标签映射
    import json
    label_mapping = {
        "topic_labels": topic_labels,
        "sentiment_labels": sentiment_labels,
    }
    with open(output_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*50)
    print("CoreML export completed successfully!")
    print("="*50)
    print(f"Model saved to: {output_path}")
    print(f"Label mapping saved to: {output_dir / 'label_mapping.json'}")
    
    # 测试CoreML模型
    print("\nTesting CoreML model...")
    try:
        import coremltools.models as ct_models
        coreml_model = ct_models.MLModel(str(output_path))
        
        # 准备测试输入
        test_input = {
            "input_ids": inputs["input_ids"].numpy().astype('int32'),
            "attention_mask": inputs["attention_mask"].numpy().astype('int32'),
        }
        
        # 预测
        coreml_output = coreml_model.predict(test_input)
        print("CoreML prediction successful!")
        print(f"  Topic logits shape: {coreml_output['topic_logits'].shape}")
        print(f"  Sentiment logits shape: {coreml_output['sentiment_logits'].shape}")
        
        # 显示预测结果
        import numpy as np
        topic_pred = np.argmax(coreml_output['topic_logits'])
        sentiment_pred = np.argmax(coreml_output['sentiment_logits'])
        print(f"\nPrediction for '{text}':")
        print(f"  Topic: {topic_labels[topic_pred]} (class {topic_pred})")
        print(f"  Sentiment: {sentiment_labels[sentiment_pred]} (class {sentiment_pred})")
        
    except Exception as e:
        print(f"Warning: Could not test CoreML model: {e}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
