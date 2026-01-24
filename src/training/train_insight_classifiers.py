"""
璁�缁冩礊瀵熷垎绫诲櫒 (Insight Classifiers)

璁�缁�5涓�鏂扮殑鍒嗙被鍣ㄥご:
1. UrgencyClassifier (3绫�): low, medium, high
2. TimeFrameClassifier (4绫�): today, this_week, this_month, long_term
3. ActionTypeClassifier (6绫�): learning, exercise, work, lifestyle, social, creative
4. DifficultyClassifier (3绫�): easy, moderate, hard
5. SpecificityClassifier (3绫�): vague, moderate, specific

杩欎簺鍒嗙被鍣ㄤ娇鐢ㄥ凡璁�缁冨ソ鐨凚ERT鐗瑰緛鎻愬彇鍣ㄧ殑embedding浣滀负杈撳叆锛�
杈撳嚭鍚勮嚜缁村害鐨勫垎绫荤粨鏋溿€�

鐢ㄦ硶:
    python src/training/train_insight_classifiers.py \
        --model_dir models/trained/distill_multitask_student \
        --data_dir data/processed/insight \
        --output_dir models/trained/insight_classifiers
"""
import argparse
from pathlib import Path
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score, classification_report
import sys

# 娣诲姞椤圭洰鏍圭洰褰曞埌璺�寰�
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# 鍒嗙被鍣ㄩ厤缃�
CLASSIFIER_CONFIGS = {
    "urgency": {
        "num_classes": 3,
        "labels": ["low", "medium", "high"],
        "column": "urgency_label",
    },
    "timeFrame": {
        "num_classes": 4,
        "labels": ["today", "this_week", "this_month", "long_term"],
        "column": "timeFrame_label",
    },
    "actionType": {
        "num_classes": 6,
        "labels": ["learning", "exercise", "work", "lifestyle", "social", "creative"],
        "column": "actionType_label",
    },
    "difficulty": {
        "num_classes": 3,
        "labels": ["easy", "moderate", "hard"],
        "column": "difficulty_label",
    },
    "specificity": {
        "num_classes": 3,
        "labels": ["vague", "moderate", "specific"],
        "column": "specificity_label",
    },
}


def mean_pool(last_hidden_state, attention_mask):
    """Mean pooling over token embeddings"""
    m = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    masked = last_hidden_state * m
    summed = masked.sum(dim=1)
    lengths = m.sum(dim=1).clamp(min=1)
    return summed / lengths


def encode_texts(model, tokenizer, texts, max_length, batch_size=32, device="cpu"):
    """
    灏嗘枃鏈�鎵归噺缂栫爜涓篹mbedding
    
    Args:
        model: BERT妯″瀷
        tokenizer: 鍒嗚瘝鍣�
        texts: 鏂囨湰鍒楄〃
        max_length: 鏈€澶у簭鍒楅暱搴�
        batch_size: 鎵瑰ぇ灏�
        device: 璁惧��
    
    Returns:
        numpy array: (N, hidden_size) embedding鐭╅樀
    """
    model = model.to(device)
    model.eval()
    
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        enc = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        
        enc = {k: v.to(device) for k, v in enc.items()}
        
        with torch.no_grad():
            out = model(**enc)
        
        # 浣跨敤pooler_output (CLS token鐨勮〃绀�)
        if hasattr(out, 'pooler_output') and out.pooler_output is not None:
            emb = out.pooler_output.cpu().numpy()
        else:
            # fallback: mean pooling
            emb = mean_pool(out.last_hidden_state, enc["attention_mask"]).cpu().numpy()
        
        all_embeddings.append(emb)
        
        if (i // batch_size) % 10 == 0:
            print(f"  Encoded {min(i + batch_size, len(texts))}/{len(texts)} texts...")
    
    return np.vstack(all_embeddings)


class SimpleClassifier(nn.Module):
    """绠€鍗曠殑绾挎€у垎绫诲櫒"""
    
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        return self.classifier(x)


def train_classifier(
    name: str,
    config: dict,
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    val_embeddings: np.ndarray,
    val_labels: np.ndarray,
    hidden_size: int,
    output_dir: Path,
    epochs: int = 20,
    lr: float = 0.01,
    batch_size: int = 64,
    device: str = "cpu"
):
    """
    璁�缁冨崟涓�鍒嗙被鍣�
    
    Args:
        name: 鍒嗙被鍣ㄥ悕绉�
        config: 鍒嗙被鍣ㄩ厤缃�
        train_embeddings: 璁�缁冮泦embedding
        train_labels: 璁�缁冮泦鏍囩��
        val_embeddings: 楠岃瘉闆唀mbedding
        val_labels: 楠岃瘉闆嗘爣绛�
        hidden_size: embedding缁村害
        output_dir: 杈撳嚭鐩�褰�
        epochs: 璁�缁冭疆鏁�
        lr: 瀛︿範鐜�
        batch_size: 鎵瑰ぇ灏�
        device: 璁�缁冭�惧��
    
    Returns:
        dict: 璁�缁冩寚鏍�
    """
    print(f"\n{'='*60}")
    print(f"Training {name.title()} Classifier")
    print(f"  Classes: {config['num_classes']} ({', '.join(config['labels'])})")
    print(f"  Train samples: {len(train_labels)}")
    print(f"  Val samples: {len(val_labels)}")
    print(f"{'='*60}")
    
    # 杞�鎹�涓篜yTorch tensors
    train_X = torch.FloatTensor(train_embeddings)
    train_y = torch.LongTensor(train_labels)
    val_X = torch.FloatTensor(val_embeddings)
    val_y = torch.LongTensor(val_labels)
    
    # 鍒涘缓DataLoader
    train_dataset = TensorDataset(train_X, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 鍒涘缓妯″瀷
    model = SimpleClassifier(hidden_size, config["num_classes"])
    model = model.to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    best_epoch = 0
    best_state = None
    
    for epoch in range(epochs):
        # 璁�缁�
        model.train()
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # 楠岃瘉
        model.eval()
        with torch.no_grad():
            val_logits = model(val_X.to(device))
            val_preds = val_logits.argmax(dim=1).cpu().numpy()
            val_acc = accuracy_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds, average="macro")
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_state = model.state_dict().copy()
    
    print(f"  Best epoch: {best_epoch}, Best val acc: {best_val_acc:.4f}")
    
    # 鎭㈠�嶆渶浣虫ā鍨�
    model.load_state_dict(best_state)
    
    # 淇濆瓨妯″瀷
    save_path = output_dir / f"{name}_classifier.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "hidden_size": hidden_size,
        "num_classes": config["num_classes"],
        "labels": config["labels"],
        "name": name,
    }, save_path)
    print(f"  Saved: {save_path}")
    
    # 璁＄畻鏈€缁堟寚鏍�
    model.eval()
    with torch.no_grad():
        val_logits = model(val_X.to(device))
        val_preds = val_logits.argmax(dim=1).cpu().numpy()
    
    metrics = {
        "name": name,
        "num_classes": config["num_classes"],
        "labels": config["labels"],
        "best_epoch": best_epoch,
        "val_accuracy": float(accuracy_score(val_labels, val_preds)),
        "val_f1_macro": float(f1_score(val_labels, val_preds, average="macro")),
    }
    
    return model, metrics


def evaluate_on_test(
    model: nn.Module,
    name: str,
    config: dict,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    device: str = "cpu"
):
    """鍦ㄦ祴璇曢泦涓婅瘎浼版ā鍨�"""
    model.eval()
    test_X = torch.FloatTensor(test_embeddings).to(device)
    
    with torch.no_grad():
        logits = model(test_X)
        preds = logits.argmax(dim=1).cpu().numpy()
    
    acc = accuracy_score(test_labels, preds)
    f1 = f1_score(test_labels, preds, average="macro")
    
    print(f"\n{name.title()} Classifier - Test Results:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 (macro): {f1:.4f}")
    print("\nClassification Report:")
    # 使用 labels 参数确保所有类别都被包含
    labels = list(range(config["num_classes"]))
    print(classification_report(
        test_labels, preds, 
        target_names=config["labels"],
        labels=labels,
        zero_division=0
    ))
    
    return {
        "test_accuracy": float(acc),
        "test_f1_macro": float(f1),
    }


def main():
    parser = argparse.ArgumentParser(description="Train Insight Classifiers")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models/trained/distill_multitask_student",
        help="Path to BERT model for feature extraction"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed/insight",
        help="Path to insight data directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/trained/insight_classifiers",
        help="Output directory for trained classifiers"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Training batch size"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps" if torch.backends.mps.is_available() else "cpu",
        help="Device to use for training"
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("Insight Classifiers Training Script")
    print("="*60)
    print(f"Model directory: {args.model_dir}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {args.device}")
    
    # 鍔犺浇鍒嗚瘝鍣ㄥ拰妯″瀷
    print(f"\nLoading model from {args.model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    encoder = AutoModel.from_pretrained(args.model_dir)
    encoder.eval()
    
    hidden_size = encoder.config.hidden_size
    print(f"  Hidden size: {hidden_size}")
    
    # 鍔犺浇鏁版嵁
    print(f"\nLoading data from {args.data_dir}...")
    data_dir = Path(args.data_dir)
    train_df = pd.read_csv(data_dir / "train_insight.csv")
    val_df = pd.read_csv(data_dir / "val_insight.csv")
    test_df = pd.read_csv(data_dir / "test_insight.csv")
    
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    
    # 缂栫爜鏂囨湰涓篹mbedding
    print(f"\nEncoding texts to embeddings...")
    print("  Encoding training set...")
    train_embeddings = encode_texts(
        encoder, tokenizer, 
        train_df["text"].tolist(), 
        args.max_length, 
        batch_size=64,
        device=args.device
    )
    
    print("  Encoding validation set...")
    val_embeddings = encode_texts(
        encoder, tokenizer, 
        val_df["text"].tolist(), 
        args.max_length, 
        batch_size=64,
        device=args.device
    )
    
    print("  Encoding test set...")
    test_embeddings = encode_texts(
        encoder, tokenizer, 
        test_df["text"].tolist(), 
        args.max_length, 
        batch_size=64,
        device=args.device
    )
    
    # 淇濆瓨embeddings浠ヤ究澶嶇敤
    embeddings_dir = output_dir / "embeddings"
    embeddings_dir.mkdir(exist_ok=True)
    np.save(embeddings_dir / "train_embeddings.npy", train_embeddings)
    np.save(embeddings_dir / "val_embeddings.npy", val_embeddings)
    np.save(embeddings_dir / "test_embeddings.npy", test_embeddings)
    print(f"  Saved embeddings to {embeddings_dir}")
    
    # 璁�缁冩瘡涓�鍒嗙被鍣�
    all_metrics = {}
    trained_models = {}
    
    for name, config in CLASSIFIER_CONFIGS.items():
        train_labels = train_df[config["column"]].values
        val_labels = val_df[config["column"]].values
        test_labels = test_df[config["column"]].values
        
        model, metrics = train_classifier(
            name=name,
            config=config,
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            val_embeddings=val_embeddings,
            val_labels=val_labels,
            hidden_size=hidden_size,
            output_dir=output_dir,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=args.device
        )
        
        # 娴嬭瘯闆嗚瘎浼�
        test_metrics = evaluate_on_test(
            model, name, config,
            test_embeddings, test_labels,
            device=args.device
        )
        
        metrics.update(test_metrics)
        all_metrics[name] = metrics
        trained_models[name] = model
    
    # 淇濆瓨鎵€鏈夋寚鏍�
    metrics_path = output_dir / "training_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"\nSaved metrics to {metrics_path}")
    
    # 鎵撳嵃鎬荤粨
    print("\n" + "="*60)
    print("Training Complete! Summary:")
    print("="*60)
    print(f"\n{'Classifier':<15} {'Val Acc':>10} {'Val F1':>10} {'Test Acc':>10} {'Test F1':>10}")
    print("-"*60)
    for name, metrics in all_metrics.items():
        print(f"{name:<15} {metrics['val_accuracy']:>10.4f} {metrics['val_f1_macro']:>10.4f} {metrics['test_accuracy']:>10.4f} {metrics['test_f1_macro']:>10.4f}")
    
    print(f"\nTrained models saved to: {output_dir}")
    print("\nNext step: Export to CoreML using:")
    print(f"  python src/export/export_insight_classifiers.py --input_dir {output_dir}")


if __name__ == "__main__":
    main()
