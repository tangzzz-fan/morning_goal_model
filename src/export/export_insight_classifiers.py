"""
导出洞察分类器为 CoreML 可更新模型 (Updatable CoreML Models)

将训练好的5个分类器导出为 CoreML 格式:
1. UrgencyClassifier_Updatable.mlpackage
2. TimeFrameClassifier_Updatable.mlpackage
3. ActionTypeClassifier_Updatable.mlpackage
4. DifficultyClassifier_Updatable.mlpackage
5. SpecificityClassifier_Updatable.mlpackage

这些模型与现有的 TopicClassifier 和 SentimentClassifier 保持相同的格式，
都是可更新的 NeuralNetwork 模型，支持端侧训练。

用法:
    python src/export/export_insight_classifiers.py \
        --input_dir models/trained/insight_classifiers \
        --output_dir models/exported_coreml
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


# 分类器配置 (与训练脚本保持一致)
CLASSIFIER_CONFIGS = {
    "urgency": {
        "num_classes": 3,
        "labels": ["low", "medium", "high"],
    },
    "timeFrame": {
        "num_classes": 4,
        "labels": ["today", "this_week", "this_month", "long_term"],
    },
    "actionType": {
        "num_classes": 6,
        "labels": ["learning", "exercise", "work", "lifestyle", "social", "creative"],
    },
    "difficulty": {
        "num_classes": 3,
        "labels": ["easy", "moderate", "hard"],
    },
    "specificity": {
        "num_classes": 3,
        "labels": ["vague", "moderate", "specific"],
    },
}


class SingleClassifier(nn.Module):
    """单个分类头，用于导出"""
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, embedding):
        logits = self.classifier(embedding)
        return logits


def make_classifier_updatable(
    mlmodel_path: Path, 
    output_path: Path, 
    task_name: str,
    num_classes: int,
    class_labels: list = None
):
    """
    将分类器转换为可更新模型
    
    与 export_fanout_coreml.py 中的 make_single_classifier_updatable 保持一致
    
    Args:
        mlmodel_path: 基础模型路径
        output_path: 输出路径
        task_name: 任务名称
        num_classes: 类别数
        class_labels: 类别标签列表
    """
    print(f"\n  [Updatable] Converting {task_name} classifier to updatable model...")
    
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
            print(f"    Found FC layer: {fc_layer}, output: {fc_output_name}")
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
    print(f"    Updatable layers: {updatable_layers}")
    print(f"    Output: {probs_output_name}")
    
    # 12. 保存
    updatable_model = MLModel(spec)
    updatable_model.author = "Morning Goal Team"
    updatable_model.short_description = f"Updatable {task_name.title()} Classifier for Goal Insights"
    
    # 使用 .mlpackage 扩展名
    updatable_model.save(str(output_path))
    print(f"    Saved to {output_path}")
    
    return True


def export_classifier(
    name: str,
    config: dict,
    checkpoint_path: Path,
    output_dir: Path
):
    """
    导出单个分类器为 CoreML 可更新模型
    
    Args:
        name: 分类器名称
        config: 分类器配置
        checkpoint_path: PyTorch checkpoint 路径
        output_dir: 输出目录
    
    Returns:
        Path: 导出的模型路径
    """
    print(f"\n{'='*60}")
    print(f"Exporting {name.title()}Classifier (NeuralNetwork, Updatable)")
    print(f"  Classes: {config['num_classes']} ({', '.join(config['labels'])})")
    print(f"{'='*60}")
    
    # 1. 加载 PyTorch checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    hidden_size = checkpoint["hidden_size"]
    num_classes = checkpoint["num_classes"]
    
    print(f"  Hidden size: {hidden_size}")
    print(f"  Num classes: {num_classes}")
    
    # 2. 创建 PyTorch 模型并加载权重
    classifier = SingleClassifier(hidden_size, num_classes)
    classifier.classifier.load_state_dict({
        "weight": checkpoint["state_dict"]["classifier.weight"],
        "bias": checkpoint["state_dict"]["classifier.bias"]
    })
    classifier.eval()
    
    # 3. Trace
    dummy_embedding = torch.randn(1, hidden_size)
    traced_classifier = torch.jit.trace(classifier, dummy_embedding)
    
    # 4. 转换为 NeuralNetwork (支持端侧更新)
    base_model = ct.convert(
        traced_classifier,
        inputs=[ct.TensorType(name="embedding", shape=(1, hidden_size))],
        outputs=[ct.TensorType(name=f"{name}_logits")],
        minimum_deployment_target=ct.target.iOS14,
        convert_to="neuralnetwork"
    )
    
    # 5. 保存基础模型
    base_path = output_dir / f"{name.title()}Classifier_Base.mlmodel"
    base_model.save(str(base_path))
    print(f"  Base model saved: {base_path}")
    
    # 6. 转换为可更新模型
    final_path = output_dir / f"{name.title()}Classifier_Updatable.mlpackage"
    make_classifier_updatable(
        base_path, 
        final_path, 
        task_name=name,
        num_classes=num_classes,
        class_labels=config["labels"]
    )
    
    # 7. 删除基础模型
    base_path.unlink()
    print(f"✅ Exported: {final_path}")
    
    return final_path


def main():
    parser = argparse.ArgumentParser(description="Export Insight Classifiers to CoreML")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="models/trained/insight_classifiers",
        help="Directory containing trained classifier checkpoints"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/exported_coreml",
        help="Output directory for CoreML models"
    )
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("Insight Classifiers CoreML Export Script")
    print("="*60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # 导出每个分类器
    exported_models = []
    
    for name, config in CLASSIFIER_CONFIGS.items():
        checkpoint_path = input_dir / f"{name}_classifier.pt"
        
        if not checkpoint_path.exists():
            print(f"\n⚠️  Skipping {name}: checkpoint not found at {checkpoint_path}")
            continue
        
        try:
            model_path = export_classifier(
                name=name,
                config=config,
                checkpoint_path=checkpoint_path,
                output_dir=output_dir
            )
            exported_models.append((name, model_path))
        except Exception as e:
            print(f"\n❌ Error exporting {name}: {e}")
            import traceback
            traceback.print_exc()
    
    # 打印总结
    print("\n" + "="*60)
    print("Export Complete!")
    print("="*60)
    print("\nExported Models:")
    
    for name, path in exported_models:
        size_kb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024
        print(f"  {name.title()}Classifier_Updatable.mlpackage")
        print(f"    └── Format: NeuralNetwork (Updatable, iOS 14+)")
        print(f"    └── Size: ~{size_kb:.1f}KB")
    
    print("\nArchitecture (with new classifiers):")
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
        ┌──────────┬───────────┼───────────┬──────────┬──────────┐
        │          │           │           │          │          │
        ▼          ▼           ▼           ▼          ▼          ▼
    ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
    │ Topic  │ │Senti-  │ │Urgency │ │Time-   │ │Action- │ │Diffi-  │ ...
    │Classi- │ │ment    │ │Classi- │ │Frame   │ │Type    │ │culty   │
    │fier    │ │Classi- │ │fier    │ │Classi- │ │Classi- │ │Classi- │
    │(16cls) │ │fier    │ │(3cls)  │ │fier    │ │fier    │ │fier    │
    │  ✏️    │ │(3cls)  │ │  ✏️    │ │(4cls)  │ │(6cls)  │ │(3cls)  │
    │        │ │  ✏️    │ │        │ │  ✏️    │ │  ✏️    │ │  ✏️    │
    └────────┘ └────────┘ └────────┘ └────────┘ └────────┘ └────────┘
    
    ✏️ = On-device trainable (Updatable)
    """)
    
    print("\nAll 7 classifiers in exported_coreml:")
    print("  - BertFeatureExtractor.mlpackage (Static, shared)")
    print("  - TopicClassifier_Updatable.mlpackage (16 classes)")
    print("  - SentimentClassifier_Updatable.mlpackage (3 classes)")
    for name, _ in exported_models:
        config = CLASSIFIER_CONFIGS[name]
        print(f"  - {name.title()}Classifier_Updatable.mlpackage ({config['num_classes']} classes)")
    
    print("\nNext Steps:")
    print("  1. Copy these files to your iOS app's Resources folder")
    print("  2. Update InsightModelManager to load all 7 classifiers")
    print("  3. Implement GoalDataAggregator for data collection")
    print("  4. Build InsightGenerator using rule engine")


if __name__ == "__main__":
    main()
