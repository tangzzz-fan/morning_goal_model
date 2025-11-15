import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.metrics import f1_score, accuracy_score


def read_csvs(data_dir):
    p = Path(data_dir)
    train = pd.read_csv(p / "train.csv")
    val = pd.read_csv(p / "val.csv")
    test = pd.read_csv(p / "test.csv")
    return train, val, test


def to_hf_dataset(df, tokenizer, max_length):
    texts = df["text"].tolist()
    labels = df["label"].tolist()
    enc = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    enc["labels"] = labels
    return enc


def freeze_layers(model, keep_last_n):
    for name, param in model.named_parameters():
        param.requires_grad = False
    total = len(model.bert.encoder.layer)
    for i in range(total - keep_last_n, total):
        for p in model.bert.encoder.layer[i].parameters():
            p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True
    for p in model.bert.pooler.parameters():
        p.requires_grad = True


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    return {"f1": f1, "accuracy": acc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--model_name", default="bert-base-chinese")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--keep_last_n", type=int, default=3)
    parser.add_argument(
        "--output_dir",
        default="models/trained/bert_base_chinese",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit_train", type=int, default=10000)
    parser.add_argument("--limit_eval", type=int, default=2000)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    train_df, val_df, test_df = read_csvs(args.data_dir)
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
    num_labels = len(sorted(set(train_df["label"].tolist())))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=num_labels
    )
    freeze_layers(model, args.keep_last_n)
    train_enc = to_hf_dataset(train_df, tokenizer, args.max_length)
    val_enc = to_hf_dataset(val_df, tokenizer, args.max_length)
    test_enc = to_hf_dataset(test_df, tokenizer, args.max_length)

    class DictDataset(torch.utils.data.Dataset):
        def __init__(self, enc):
            self.enc = enc

        def __len__(self):
            return len(self.enc["labels"])

        def __getitem__(self, idx):
            return {k: torch.tensor(v[idx]) for k, v in self.enc.items()}

    train_ds = DictDataset(train_enc)
    val_ds = DictDataset(val_enc)
    test_ds = DictDataset(test_enc)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
    metrics_val = trainer.evaluate()
    metrics_test = trainer.evaluate(test_ds)
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "metrics_val.json", "w", encoding="utf-8") as f:
        f.write(pd.Series(metrics_val).to_json())
    with open(out_path / "metrics_test.json", "w", encoding="utf-8") as f:
        f.write(pd.Series(metrics_test).to_json())


if __name__ == "__main__":
    main()
