import argparse
from pathlib import Path
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib


def mean_pool(last_hidden_state, attention_mask):
    m = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    masked = last_hidden_state * m
    summed = masked.sum(dim=1)
    lengths = m.sum(dim=1).clamp(min=1)
    return summed / lengths


def encode_embed(model, tokenizer, texts, max_length):
    enc = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    with torch.no_grad():
        out = model(**enc)
    emb = mean_pool(out.last_hidden_state, enc["attention_mask"]).cpu().numpy()
    return emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="models/trained/distill_student")
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument(
        "--output_dir",
        default="models/optimized/classifier_baseline",
    )
    ap.add_argument("--max_length", type=int, default=128)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    base = AutoModel.from_pretrained(args.model_dir)
    base.eval()

    train = pd.read_csv(Path(args.data_dir) / "train.csv")
    val = pd.read_csv(Path(args.data_dir) / "val.csv")
    test = pd.read_csv(Path(args.data_dir) / "test.csv")

    X_train = encode_embed(base, tok, train["text"].tolist(), args.max_length)
    y_train = train["label"].to_numpy()
    X_val = encode_embed(base, tok, val["text"].tolist(), args.max_length)
    y_val = val["label"].to_numpy()
    X_test = encode_embed(base, tok, test["text"].tolist(), args.max_length)
    y_test = test["label"].to_numpy()

    clf = LogisticRegression(max_iter=1000, n_jobs=-1, multi_class="auto")
    clf.fit(X_train, y_train)

    def eval_set(X, y):
        pred = clf.predict(X)
        acc = accuracy_score(y, pred)
        f1 = f1_score(y, pred, average="macro")
        return {"accuracy": acc, "f1": f1}

    metrics = {
        "train": eval_set(X_train, y_train),
        "val": eval_set(X_val, y_val),
        "test": eval_set(X_test, y_test),
    }

    joblib.dump(clf, out / "classifier.joblib")
    Path(out / "classifier_baseline_report.json").write_text(
        json.dumps(metrics), encoding="utf-8"
    )

    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
