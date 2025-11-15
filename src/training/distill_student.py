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
import torch.nn.functional as F


def read_csvs(data_dir):
    p = Path(data_dir)
    train = pd.read_csv(p / "train.csv")
    val = pd.read_csv(p / "val.csv")
    test = pd.read_csv(p / "test.csv")
    return train, val, test


def to_enc(df, tokenizer, max_length):
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


class DictDataset(torch.utils.data.Dataset):
    def __init__(self, enc):
        self.enc = enc

    def __len__(self):
        return len(self.enc["labels"])

    def __getitem__(self, idx):
        return {k: torch.tensor(v[idx]) for k, v in self.enc.items()}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    return {"f1": f1, "accuracy": acc}


class DistillTrainer(Trainer):
    def __init__(self, teacher, temperature, alpha, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()
        self.teacher.to(self.model.device)
        self.temperature = temperature
        self.alpha = alpha

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        labels = inputs.get("labels")
        outputs_s = model(**inputs)
        logits_s = outputs_s.get("logits")
        with torch.no_grad():
            outputs_t = self.teacher(**inputs)
            logits_t = outputs_t.get("logits")
        ce = torch.nn.functional.cross_entropy(logits_s, labels)
        t = self.temperature
        p_s = F.log_softmax(logits_s / t, dim=-1)
        p_t = F.softmax(logits_t / t, dim=-1)
        kl = F.kl_div(p_s, p_t, reduction="batchmean") * (t * t)
        loss = self.alpha * kl + (1 - self.alpha) * ce
        return (loss, outputs_s) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--teacher_name", default="bert-base-chinese")
    parser.add_argument(
        "--student_name",
        default="uer/chinese_roberta_L-4_H-512",
    )
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument(
        "--output_dir",
        default="models/trained/distill_student",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit_train", type=int, default=20000)
    parser.add_argument("--limit_eval", type=int, default=4000)
    parser.add_argument("--grad_accum", type=int, default=1)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        torch.backends.cudnn.benchmark = True

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
    tokenizer_s = AutoTokenizer.from_pretrained(args.student_name)
    model_s = AutoModelForSequenceClassification.from_pretrained(
        args.student_name,
        num_labels=num_labels,
        use_safetensors=True,
    )
    model_t = AutoModelForSequenceClassification.from_pretrained(
        args.teacher_name,
        num_labels=num_labels,
        use_safetensors=True,
    )
    enc_train = to_enc(train_df, tokenizer_s, args.max_length)
    enc_val = to_enc(val_df, tokenizer_s, args.max_length)
    enc_test = to_enc(test_df, tokenizer_s, args.max_length)
    train_ds = DictDataset(enc_train)
    val_ds = DictDataset(enc_val)
    test_ds = DictDataset(enc_test)
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
        fp16=use_gpu,
        gradient_accumulation_steps=args.grad_accum,
        dataloader_num_workers=0,
    )
    trainer = DistillTrainer(
        teacher=model_t,
        temperature=args.temperature,
        alpha=args.alpha,
        model=model_s,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer_s,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
    metrics_val = trainer.evaluate()
    metrics_test = trainer.evaluate(test_ds)
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model_s.save_pretrained(out_path)
    tokenizer_s.save_pretrained(out_path)
    with open(out_path / "metrics_val.json", "w", encoding="utf-8") as f:
        f.write(pd.Series(metrics_val).to_json())
    with open(out_path / "metrics_test.json", "w", encoding="utf-8") as f:
        f.write(pd.Series(metrics_test).to_json())


if __name__ == "__main__":
    main()
