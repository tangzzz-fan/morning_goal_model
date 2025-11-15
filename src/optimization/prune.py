import argparse
from pathlib import Path
import pandas as pd
import torch
from torch.nn.utils import prune
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score


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
        return_tensors="pt",
    )
    return enc, torch.tensor(labels)


def evaluate(model, tokenizer, df, max_length):
    model.eval()
    enc, labels = to_enc(df, tokenizer, max_length)
    with torch.no_grad():
        logits = model(**enc).logits
    preds = torch.argmax(logits, dim=-1)
    acc = (preds == labels).float().mean().item()
    f1 = f1_score(labels.numpy(), preds.numpy(), average="macro")
    return {"accuracy": acc, "f1": f1}


def prune_model(model, amount):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--output_dir", default="models/optimized/pruned")
    parser.add_argument("--amount", type=float, default=0.2)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    train_df, val_df, test_df = read_csvs(args.data_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    metrics_val_fp = evaluate(model, tokenizer, val_df, args.max_length)
    metrics_test_fp = evaluate(model, tokenizer, test_df, args.max_length)
    pmodel = prune_model(model, args.amount)
    metrics_val_p = evaluate(pmodel, tokenizer, val_df, args.max_length)
    metrics_test_p = evaluate(pmodel, tokenizer, test_df, args.max_length)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(pmodel.state_dict(), out / "pruned_state_dict.pt")
    report = {
        "fp32_val": metrics_val_fp,
        "fp32_test": metrics_test_fp,
        "pruned_val": metrics_val_p,
        "pruned_test": metrics_test_p,
        "amount": args.amount,
    }
    with open(out / "prune_metrics.json", "w", encoding="utf-8") as f:
        f.write(pd.Series(report).to_json())


if __name__ == "__main__":
    main()
