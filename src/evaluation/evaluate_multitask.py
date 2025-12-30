"""
多任务模型详细评估脚本
生成包含分类报告和混淆矩阵的详细评估结果
"""
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from transformers import AutoTokenizer, AutoConfig
import sys
import json

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.multitask_bert import MultitaskBertForClassification

def evaluate_model(model_path, data_path, output_dir=None):
    """
    评估模型性能
    """
    print(f"Loading model from {model_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    model = MultitaskBertForClassification.from_pretrained(
        model_path,
        config=config,
        num_topic_labels=config.num_topic_labels if hasattr(config, 'num_topic_labels') else 16,
        num_sentiment_labels=config.num_sentiment_labels if hasattr(config, 'num_sentiment_labels') else 3
    )
    model.to(device)
    model.eval()
    
    # 加载测试数据
    print(f"Loading test data from {data_path}...")
    df = pd.read_csv(data_path)
    
    texts = df['text'].tolist()
    true_topics = df['topic_label'].tolist()
    true_sentiments = df['sentiment_label'].tolist()
    
    # 预测
    print("Running predictions...")
    pred_topics = []
    pred_sentiments = []
    
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        topic_preds = torch.argmax(outputs['topic_logits'], dim=1).cpu().numpy()
        sentiment_preds = torch.argmax(outputs['sentiment_logits'], dim=1).cpu().numpy()
        
        pred_topics.extend(topic_preds)
        pred_sentiments.extend(sentiment_preds)
        
        if (i + batch_size) % 100 == 0:
            print(f"Processed {i + batch_size}/{len(texts)} samples")
            
    # 计算指标
    print("\nComputing metrics...")
    
    # 定义标签名称 (根据实际项目定义修改)
    topic_names = [f"Topic {i}" for i in range(16)] # 建议替换为真实类别名
    sentiment_names = ["Negative", "Neutral", "Positive"]
    
    # 生成报告
    report = {
        "overall_metrics": {
            "topic_accuracy": accuracy_score(true_topics, pred_topics),
            "topic_macro_f1": f1_score(true_topics, pred_topics, average='macro'),
            "sentiment_accuracy": accuracy_score(true_sentiments, pred_sentiments),
            "sentiment_macro_f1": f1_score(true_sentiments, pred_sentiments, average='macro')
        },
        "topic_report": classification_report(true_topics, pred_topics, target_names=topic_names, output_dict=True),
        "sentiment_report": classification_report(true_sentiments, pred_sentiments, target_names=sentiment_names, output_dict=True)
    }
    
    # 打印简要报告
    print("\n" + "="*50)
    print("Evaluation Report")
    print("="*50)
    print(f"Topic Accuracy: {report['overall_metrics']['topic_accuracy']:.4f}")
    print(f"Topic Macro F1: {report['overall_metrics']['topic_macro_f1']:.4f}")
    print(f"Sentiment Accuracy: {report['overall_metrics']['sentiment_accuracy']:.4f}")
    print(f"Sentiment Macro F1: {report['overall_metrics']['sentiment_macro_f1']:.4f}")
    
    print("\nSentiment Classification Report:")
    print(classification_report(true_sentiments, pred_sentiments, target_names=sentiment_names))
    
    # 保存结果
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        # 保存JSON详细报告
        with open(out_path / "detailed_evaluation.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        # 保存Markdown报告
        md_report = f"""# Model Evaluation Report
        
## Overall Metrics
- **Topic Accuracy**: {report['overall_metrics']['topic_accuracy']:.4f}
- **Topic Macro F1**: {report['overall_metrics']['topic_macro_f1']:.4f}
- **Sentiment Accuracy**: {report['overall_metrics']['sentiment_accuracy']:.4f}
- **Sentiment Macro F1**: {report['overall_metrics']['sentiment_macro_f1']:.4f}

## Sentiment Analysis Details
```
{classification_report(true_sentiments, pred_sentiments, target_names=sentiment_names)}
```

## Topic Classification Details
```
{classification_report(true_topics, pred_topics, target_names=topic_names)}
```
"""
        with open(out_path / "evaluation_report.md", "w", encoding="utf-8") as f:
            f.write(md_report)
            
        print(f"\nDetailed report saved to {output_dir}")

    return report

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--data_path", required=True, help="Path to test CSV")
    parser.add_argument("--output_dir", default="evaluation_results", help="Directory to save results")
    args = parser.parse_args()
    
    evaluate_model(args.model_path, args.data_path, args.output_dir)
