"""
导出扇出架构 CoreML 模型 (Fan-out Architecture):
1. BertFeatureExtractor.mlpackage - 静态 MLProgram，用于提取特征
2. TopicClassifier_Updatable.mlmodel - 可更新 NeuralNetwork，Topic 分类
3. SentimentClassifier_Updatable.mlmodel - 可更新 NeuralNetwork，Sentiment 分类

这种架构解决了 CoreML 仅支持单 Loss 层的限制，每个分类器独立更新。

用法:
    python src/export/export_fanout_coreml.py --model_dir models/trained/distill_multitask_student
"""
import argparse
from pathlib import Path
import sys
import torch
import torch.nn as nn
import coremltools as ct
from coremltools.models import neural_network
from coremltools.models.neural_network import SgdParams
from coremltools.models import MLModel
from coremltools.proto import FeatureTypes_pb2 as ft
from transformers import AutoConfig, AutoTokenizer

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.multitask_bert import MultitaskBertForClassification


class BertFeatureExtractor(nn.Module):
    """只包含 BERT 主干，输出 pooled_output"""
    def __init__(self, original_model):
        super().__init__()
        self.bert = original_model.bert
        self.dropout = original_model.dropout
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        pooled = outputs.pooler_output
        pooled = self.dropout(pooled)
        return pooled


class SingleClassifier(nn.Module):
    """单个分类头，用于独立导出"""
    def __init__(self, classifier_layer):
        super().__init__()
        self.classifier = classifier_layer
        
    def forward(self, embedding):
        logits = self.classifier(embedding)
        return logits


def make_single_classifier_updatable(
    mlmodel_path: Path, 
    output_path: Path, 
    task_name: str,
    num_classes: int,
    class_labels: list = None
):
    """
    将单个分类器转换为可更新模型
    
    Args:
        mlmodel_path: 基础模型路径
        output_path: 输出路径
        task_name: 任务名称 (topic/sentiment)
        num_classes: 类别数
        class_labels: 类别标签列表
    """
    print(f"\n[Updatable] Converting {task_name} classifier to updatable model...")
    
    # 1. 加载 CoreML 模型规范
    mlmodel = ct.models.MLModel(str(mlmodel_path))
    spec = mlmodel.get_spec()
    
    # 2. 获取 NeuralNetworkBuilder
    builder = neural_network.NeuralNetworkBuilder(spec=spec)
    
    # 3. 找到 innerProduct 层
    layers = spec.neuralNetwork.layers
    fc_layer = None
    fc_output_name = None
    
    for layer in layers:
        layer_type = layer.WhichOneof('layer')
        if layer_type == "innerProduct":
            fc_layer = layer.name
            fc_output_name = list(layer.output)[0] if layer.output else layer.name
            print(f"  Found FC layer: {fc_layer}, output: {fc_output_name}")
            break
    
    if not fc_layer:
        raise ValueError(f"Could not find innerProduct layer in {task_name} classifier")
    
    # 4. 标记层为 Updatable
    builder.make_updatable([fc_layer])
    
    # 5. 添加 Softmax 层
    probs_output_name = f"{task_name}_probs"
    builder.add_softmax(
        name=f"{task_name}_softmax", 
        input_name=fc_output_name, 
        output_name=probs_output_name
    )
    
    # 6. 设置 Loss Function
    builder.set_categorical_cross_entropy_loss(
        name=f"{task_name}_loss",
        input=probs_output_name
    )
    
    # 7. 设置 Optimizer (SGD)
    builder.set_sgd_optimizer(
        SgdParams(lr=0.01, batch=1, momentum=0.9)
    )
    
    # 8. 设置 Epochs
    builder.set_epochs(10)
    
    # 9. 确保 isUpdatable 标记
    spec.isUpdatable = True
    
    # 10. 更新模型输出描述 - 只保留 probs 输出
    while len(spec.description.output) > 0:
        spec.description.output.pop()
    
    probs_output = spec.description.output.add()
    probs_output.name = probs_output_name
    probs_output.type.multiArrayType.dataType = ft.ArrayFeatureType.FLOAT32
    
    # 11. 验证
    updatable_layers = [l.name for l in spec.neuralNetwork.layers if l.isUpdatable]
    print(f"  Updatable layers: {updatable_layers}")
    print(f"  Output: {probs_output_name}")
    
    # 12. 保存
    updatable_model = MLModel(spec)
    updatable_model.author = "Morning Goal Team"
    updatable_model.short_description = f"Updatable {task_name.title()} Classifier"
    
    # 使用 .mlpackage 扩展名
    updatable_model.save(str(output_path))
    print(f"  Saved to {output_path}")
    
    return True


def export_feature_extractor(original_model, tokenizer, output_dir: Path, reuse_existing: Path = None):
    """导出静态特征提取器
    
    Args:
        original_model: 原始 PyTorch 模型
        tokenizer: Tokenizer
        output_dir: 输出目录
        reuse_existing: 如果提供，复用已存在的模型路径
    """
    print("\n" + "="*60)
    print("Step 1: BertFeatureExtractor (MLProgram, Static)")
    print("="*60)
    
    fe_path = output_dir / "BertFeatureExtractor.mlpackage"
    
    # 如果指定复用已有模型
    if reuse_existing and reuse_existing.exists():
        import shutil
        print(f"  Reusing existing model from: {reuse_existing}")
        if fe_path.exists():
            shutil.rmtree(fe_path)
        shutil.copytree(reuse_existing, fe_path)
        print(f"✅ Copied to: {fe_path}")
        return fe_path
    
    # 如果目标路径已存在
    if fe_path.exists():
        print(f"  Feature extractor already exists at: {fe_path}")
        print(f"  Skipping export. Use --force to re-export.")
        print(f"✅ Using existing: {fe_path}")
        return fe_path
    
    print("  Exporting new feature extractor...")
    feature_extractor = BertFeatureExtractor(original_model)
    
    # 创建 dummy input
    dummy_text = "测试输入"
    inputs = tokenizer(
        dummy_text, 
        return_tensors="pt", 
        max_length=128, 
        padding="max_length", 
        truncation=True
    )
    
    # Trace
    traced_fe = torch.jit.trace(
        feature_extractor, 
        (inputs["input_ids"], inputs["attention_mask"])
    )
    
    # 转换为 CoreML (MLProgram 格式，性能更优)
    fe_model = ct.convert(
        traced_fe,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, 128), dtype=int),
            ct.TensorType(name="attention_mask", shape=(1, 128), dtype=int)
        ],
        outputs=[ct.TensorType(name="embedding")],
        minimum_deployment_target=ct.target.iOS16,
        convert_to="mlprogram"
    )
    
    fe_model.author = "Morning Goal Team"
    fe_model.short_description = "BERT Feature Extractor - Static"
    
    fe_model.save(str(fe_path))
    print(f"✅ Saved: {fe_path}")
    
    return fe_path


def export_topic_classifier(original_model, config, output_dir: Path, topic_labels: list = None):
    """导出 Topic 分类器"""
    print("\n" + "="*60)
    print("Step 2: Exporting TopicClassifier (NeuralNetwork, Updatable)")
    print("="*60)
    
    # 创建单独的分类器
    topic_classifier = SingleClassifier(original_model.topic_classifier)
    
    # Trace
    dummy_embedding = torch.randn(1, config.hidden_size)
    traced_classifier = torch.jit.trace(topic_classifier, dummy_embedding)
    
    # 转换为 NeuralNetwork (支持端侧更新)
    num_topic_labels = getattr(config, 'num_topic_labels', 16)
    base_model = ct.convert(
        traced_classifier,
        inputs=[ct.TensorType(name="embedding", shape=(1, config.hidden_size))],
        outputs=[ct.TensorType(name="topic_logits")],
        minimum_deployment_target=ct.target.iOS14,
        convert_to="neuralnetwork"
    )
    
    # 保存基础模型
    base_path = output_dir / "TopicClassifier_Base.mlmodel"
    base_model.save(str(base_path))
    print(f"  Base model saved: {base_path}")
    
    # 转换为可更新模型
    final_path = output_dir / "TopicClassifier_Updatable.mlpackage"
    make_single_classifier_updatable(
        base_path, 
        final_path, 
        task_name="topic",
        num_classes=num_topic_labels,
        class_labels=topic_labels
    )
    
    # 删除基础模型
    base_path.unlink()
    print(f"✅ Saved: {final_path}")
    
    return final_path


def export_sentiment_classifier(original_model, config, output_dir: Path, sentiment_labels: list = None):
    """导出 Sentiment 分类器"""
    print("\n" + "="*60)
    print("Step 3: Exporting SentimentClassifier (NeuralNetwork, Updatable)")
    print("="*60)
    
    # 创建单独的分类器
    sentiment_classifier = SingleClassifier(original_model.sentiment_classifier)
    
    # Trace
    dummy_embedding = torch.randn(1, config.hidden_size)
    traced_classifier = torch.jit.trace(sentiment_classifier, dummy_embedding)
    
    # 转换为 NeuralNetwork
    num_sentiment_labels = getattr(config, 'num_sentiment_labels', 3)
    base_model = ct.convert(
        traced_classifier,
        inputs=[ct.TensorType(name="embedding", shape=(1, config.hidden_size))],
        outputs=[ct.TensorType(name="sentiment_logits")],
        minimum_deployment_target=ct.target.iOS14,
        convert_to="neuralnetwork"
    )
    
    # 保存基础模型
    base_path = output_dir / "SentimentClassifier_Base.mlmodel"
    base_model.save(str(base_path))
    print(f"  Base model saved: {base_path}")
    
    # 转换为可更新模型
    final_path = output_dir / "SentimentClassifier_Updatable.mlpackage"
    make_single_classifier_updatable(
        base_path, 
        final_path, 
        task_name="sentiment",
        num_classes=num_sentiment_labels,
        class_labels=sentiment_labels or ["Negative", "Neutral", "Positive"]
    )
    
    # 删除基础模型
    base_path.unlink()
    print(f"✅ Saved: {final_path}")
    
    return final_path


def main():
    parser = argparse.ArgumentParser(description="Export Fan-out CoreML Models")
    parser.add_argument(
        "--model_dir", 
        type=str, 
        required=True, 
        help="Path to trained PyTorch model directory"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="models/exported_coreml", 
        help="Output directory for CoreML models"
    )
    parser.add_argument(
        "--reuse-fe",
        type=str,
        default=None,
        help="Path to existing BertFeatureExtractor.mlpackage to reuse (skip conversion)"
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    reuse_fe_path = Path(args.reuse_fe) if args.reuse_fe else None
    
    print("\n" + "="*60)
    print("Fan-out CoreML Export Script")
    print("="*60)
    print(f"Input model: {args.model_dir}")
    print(f"Output directory: {output_dir}")
    if reuse_fe_path:
        print(f"Reuse Feature Extractor: {reuse_fe_path}")
    
    # 加载模型
    print(f"\nLoading model from {args.model_dir}...")
    config = AutoConfig.from_pretrained(args.model_dir)
    original_model = MultitaskBertForClassification.from_pretrained(args.model_dir)
    original_model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    
    # 获取标签信息
    topic_labels = None
    sentiment_labels = ["Negative", "Neutral", "Positive"]
    
    if hasattr(config, 'id2label'):
        topic_labels = list(config.id2label.values())
    
    # 导出三个模型
    fe_path = export_feature_extractor(original_model, tokenizer, output_dir, reuse_existing=reuse_fe_path)
    topic_path = export_topic_classifier(original_model, config, output_dir, topic_labels)
    sentiment_path = export_sentiment_classifier(original_model, config, output_dir, sentiment_labels)
    
    # 打印总结
    print("\n" + "="*60)
    print("Export Complete! 🎉")
    print("="*60)
    print("\nGenerated Files:")
    print(f"  1. {fe_path}")
    print(f"     └── Format: MLProgram (Static, iOS 16+)")
    print(f"     └── Size: ~50MB")
    print(f"  2. {topic_path}")
    print(f"     └── Format: NeuralNetwork (Updatable, iOS 14+)")
    print(f"     └── Size: ~50KB")
    print(f"  3. {sentiment_path}")
    print(f"     └── Format: NeuralNetwork (Updatable, iOS 14+)")
    print(f"     └── Size: ~10KB")
    
    print("\nArchitecture:")
    print("""
    ┌─────────────────────────────────────────────────────────┐
    │                    User Input Text                      │
    └──────────────────────────┬──────────────────────────────┘
                               │
                               ▼
    ┌─────────────────────────────────────────────────────────┐
    │        BertFeatureExtractor.mlpackage (Static)          │
    │                  → 768-dim Embedding                    │
    └──────────────────────────┬──────────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                                 │
              ▼                                 ▼
    ┌─────────────────────┐           ┌─────────────────────┐
    │ TopicClassifier     │           │ SentimentClassifier │
    │ _Updatable.mlpackage│           │ _Updatable.mlpackage│
    │   (16 classes)      │           │   (3 classes)       │
    │   ✏️ On-device       │           │   ✏️ On-device       │
    │      trainable      │           │      trainable      │
    └─────────────────────┘           └─────────────────────┘
    """)
    
    print("Next Steps:")
    print("  1. Copy these files to your iOS app's Resources folder")
    print("  2. Use InsightModelManager to load and manage these models")
    print("  3. Call MLUpdateTask separately for each classifier when user corrects")


if __name__ == "__main__":
    main()
