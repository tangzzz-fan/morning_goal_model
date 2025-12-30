"""
导出多任务模型为 CoreML (.mlpackage) 格式
"""
import torch
import torch.nn as nn
import coremltools as ct
import argparse
from pathlib import Path
import sys
from transformers import AutoTokenizer, AutoConfig

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.multitask_bert import MultitaskBertForClassification

class MultitaskTraceWrapper(nn.Module):
    """
    CoreML 导出专用的包装器
    简化输出格式，移除字典返回，仅返回 Tensor 元组
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, input_ids, attention_mask):
        # 调用原始模型，不传入 labels，不返回字典
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        # outputs 是一个元组: (topic_logits, sentiment_logits, ...)
        # 我们只需要前两个
        topic_logits = outputs[0]
        sentiment_logits = outputs[1]
        
        # 应用 Softmax 获取概率
        topic_probs = torch.softmax(topic_logits, dim=1)
        sentiment_probs = torch.softmax(sentiment_logits, dim=1)
        
        return topic_probs, sentiment_probs

def export_to_coreml(model_path, output_path):
    print(f"Loading model from {model_path}...")
    
    # 1. 加载模型
    config = AutoConfig.from_pretrained(model_path)
    model = MultitaskBertForClassification.from_pretrained(
        model_path,
        config=config,
        num_topic_labels=config.num_topic_labels if hasattr(config, 'num_topic_labels') else 16,
        num_sentiment_labels=config.num_sentiment_labels if hasattr(config, 'num_sentiment_labels') else 3
    )
    model.eval()
    
    # 2. 创建 Wrapper
    wrapper = MultitaskTraceWrapper(model)
    
    # 3. 创建 Dummy Input
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    text = "这是一个测试句子"
    inputs = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    example_input = (
        inputs["input_ids"],
        inputs["attention_mask"]
    )
    
    # 4. Tracing
    print("Tracing model...")
    traced_model = torch.jit.trace(wrapper, example_input)
    
    # 5. Convert to CoreML
    print("Converting to CoreML...")
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, 128), dtype=int),
            ct.TensorType(name="attention_mask", shape=(1, 128), dtype=int),
        ],
        outputs=[
            ct.TensorType(name="topic_probs"),
            ct.TensorType(name="sentiment_probs"),
        ],
        minimum_deployment_target=ct.target.iOS16,
    )
    
    # 6. 添加元数据
    mlmodel.author = "Morning Goal Team"
    mlmodel.license = "MIT"
    mlmodel.short_description = "Multitask BERT for Topic and Sentiment Analysis"
    mlmodel.output_description["topic_probs"] = "Probability distribution over 16 topics"
    mlmodel.output_description["sentiment_probs"] = "Probability distribution over 3 sentiments (Negative, Neutral, Positive)"
    
    # 7. 保存
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving to {output_path}...")
    mlmodel.save(output_path)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to PyTorch model")
    parser.add_argument("--output_path", default="models/coreml/multitask_model.mlpackage", help="Path to save .mlpackage")
    args = parser.parse_args()
    
    export_to_coreml(args.model_path, args.output_path)
