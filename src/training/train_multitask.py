"""
多任务模型训练脚本 (Multitask Training)
同时训练主题分类和情感分析
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.metrics import f1_score, accuracy_score
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.multitask_bert import MultitaskBertForClassification


def read_multitask_csvs(data_dir):
    """读取多任务数据集"""
    p = Path(data_dir)
    train = pd.read_csv(p / "train_multitask.csv")
    val = pd.read_csv(p / "val_multitask.csv")
    test = pd.read_csv(p / "test_multitask.csv")
    return train, val, test


class MultitaskDataset(torch.utils.data.Dataset):
    """多任务数据集"""
    
    def __init__(self, texts, topic_labels, sentiment_labels, tokenizer, max_length=128):
        self.texts = texts
        self.topic_labels = topic_labels
        self.sentiment_labels = sentiment_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
        )
        
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'topic_labels': torch.tensor(self.topic_labels[idx], dtype=torch.long),
            'sentiment_labels': torch.tensor(self.sentiment_labels[idx], dtype=torch.long),
        }
        
        if 'token_type_ids' in encoding:
            item['token_type_ids'] = encoding['token_type_ids'].squeeze(0)
        
        return item


def compute_multitask_metrics(eval_pred):
    """计算多任务指标"""
    predictions, labels = eval_pred
    
    # predictions 是一个元组: (topic_logits, sentiment_logits)
    # labels 也是一个元组: (topic_labels, sentiment_labels)
    topic_logits, sentiment_logits = predictions
    topic_labels, sentiment_labels = labels[:, 0], labels[:, 1]
    
    # 计算预测
    topic_preds = np.argmax(topic_logits, axis=-1)
    sentiment_preds = np.argmax(sentiment_logits, axis=-1)
    
    # 主题分类指标
    topic_f1 = f1_score(topic_labels, topic_preds, average='macro')
    topic_acc = accuracy_score(topic_labels, topic_preds)
    
    # 情感分析指标
    sentiment_f1 = f1_score(sentiment_labels, sentiment_preds, average='macro')
    sentiment_acc = accuracy_score(sentiment_labels, sentiment_preds)
    
    return {
        'topic_f1': topic_f1,
        'topic_accuracy': topic_acc,
        'sentiment_f1': sentiment_f1,
        'sentiment_accuracy': sentiment_acc,
        'avg_f1': (topic_f1 + sentiment_f1) / 2,
        'avg_accuracy': (topic_acc + sentiment_acc) / 2,
    }


class MultitaskTrainer(Trainer):
    """自定义Trainer以处理多任务输出"""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """计算多任务损失"""
        outputs = model(**inputs)
        loss = outputs['loss']
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """预测步骤"""
        inputs = self._prepare_inputs(inputs)
        
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs['loss']
            
            # 返回两个任务的logits
            topic_logits = outputs['topic_logits']
            sentiment_logits = outputs['sentiment_logits']
            
            # 组合logits
            logits = (topic_logits.cpu(), sentiment_logits.cpu())
        
        if prediction_loss_only:
            return (loss, None, None)
        
        # 组合labels
        topic_labels = inputs['topic_labels'].cpu()
        sentiment_labels = inputs['sentiment_labels'].cpu()
        labels = torch.stack([topic_labels, sentiment_labels], dim=1)
        
        return (loss, logits, labels)


def run_training(args):
    torch.manual_seed(args.seed)
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        torch.backends.cudnn.benchmark = True
    
    # 读取数据
    print("Loading multitask datasets...")
    train_df, val_df, test_df = read_multitask_csvs(args.data_dir)
    
    # 限制数据量（可选）
    if args.limit_train and args.limit_train > 0:
        train_df = train_df.sample(
            n=min(args.limit_train, len(train_df)), random_state=args.seed
        )
    if args.limit_eval and args.limit_eval > 0:
        val_df = val_df.sample(
            n=min(args.limit_eval, len(val_df)), random_state=args.seed
        )
        test_df = test_df.sample(
            n=min(args.limit_eval, len(test_df)), random_state=args.seed
        )
    
    print(f"Train size: {len(train_df)}")
    print(f"Val size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")
    
    # 加载tokenizer
    print(f"\nLoading tokenizer from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    # 创建多任务模型
    print(f"\nInitializing multitask model from {args.base_model}...")
    config = AutoConfig.from_pretrained(args.base_model)
    
    # 创建多任务模型（会自动加载BERT权重）
    model = MultitaskBertForClassification.from_pretrained(
        args.base_model,
        config=config,
        num_topic_labels=args.num_topic_labels,
        num_sentiment_labels=args.num_sentiment_labels,
        ignore_mismatched_sizes=True,  # 忽略分类头大小不匹配
    )
    
    print(f"Model architecture:")
    print(f"  - Shared encoder: {config.num_hidden_layers} layers, {config.hidden_size} hidden size")
    print(f"  - Topic classifier: {args.num_topic_labels} classes")
    print(f"  - Sentiment classifier: {args.num_sentiment_labels} classes")
    
    # 创建数据集
    print("\nCreating datasets...")
    train_ds = MultitaskDataset(
        train_df['text'].tolist(),
        train_df['topic_label'].tolist(),
        train_df['sentiment_label'].tolist(),
        tokenizer,
        args.max_length,
    )
    val_ds = MultitaskDataset(
        val_df['text'].tolist(),
        val_df['topic_label'].tolist(),
        val_df['sentiment_label'].tolist(),
        tokenizer,
        args.max_length,
    )
    test_ds = MultitaskDataset(
        test_df['text'].tolist(),
        test_df['topic_label'].tolist(),
        test_df['sentiment_label'].tolist(),
        tokenizer,
        args.max_length,
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="avg_f1",
        greater_is_better=True,
        save_total_limit=2,
        seed=args.seed,
        fp16=use_gpu,
        gradient_accumulation_steps=args.grad_accum,
        dataloader_num_workers=0,
        logging_steps=50,
        report_to="none",
    )
    
    # 创建Trainer
    trainer = MultitaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_multitask_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    
    # 训练
    print("\n" + "="*50)
    print("Starting multitask training...")
    print("="*50 + "\n")
    trainer.train()
    
    # 评估
    print("\n" + "="*50)
    print("Evaluating on validation set...")
    print("="*50)
    metrics_val = trainer.evaluate()
    print(f"\nValidation metrics:")
    for key, value in metrics_val.items():
        if not key.startswith('eval_'):
            continue
        print(f"  {key}: {value:.4f}")
    
    print("\n" + "="*50)
    print("Evaluating on test set...")
    print("="*50)
    metrics_test = trainer.evaluate(test_ds)
    print(f"\nTest metrics:")
    for key, value in metrics_test.items():
        if not key.startswith('eval_'):
            continue
        print(f"  {key}: {value:.4f}")
    
    # 保存模型
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving model to {out_path}...")
    model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)
    
    # 保存指标
    with open(out_path / "metrics_val.json", "w", encoding="utf-8") as f:
        f.write(pd.Series(metrics_val).to_json())
    with open(out_path / "metrics_test.json", "w", encoding="utf-8") as f:
        f.write(pd.Series(metrics_test).to_json())
    
    print("\n" + "="*50)
    print("Training completed successfully!")
    print("="*50)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument(
        "--base_model",
        default="models/trained/distill_student",
        help="Base model to initialize from (will add multitask heads)",
    )
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument(
        "--output_dir",
        default="models/trained/multitask_model",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit_train", type=int, default=20000)
    parser.add_argument("--limit_eval", type=int, default=4000)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--num_topic_labels", type=int, default=16)
    parser.add_argument("--num_sentiment_labels", type=int, default=3)
    args = parser.parse_args()
    
    run_training(args)


if __name__ == "__main__":
    main()
