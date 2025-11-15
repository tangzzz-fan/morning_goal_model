import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset


def load_from_source(source, text_column, label_column):
    if source.startswith("hf:"):
        spec = source.split("hf:")[1]
        if "/" in spec:
            name, config = spec.split("/", 1)
            ds = load_dataset(name, config)
        else:
            ds = load_dataset(spec)
        df_train = pd.DataFrame(ds["train"])
        if text_column not in df_train.columns:
            raise ValueError("text_column not found")
        if label_column not in df_train.columns:
            raise ValueError("label_column not found")
        df = df_train[[text_column, label_column]].rename(
            columns={text_column: "text", label_column: "label"}
        )
        return df
    p = Path(source)
    if not p.exists():
        raise FileNotFoundError("source not found")
    df = pd.read_csv(p)
    if text_column not in df.columns or label_column not in df.columns:
        raise ValueError("missing columns")
    df = df[[text_column, label_column]].rename(
        columns={text_column: "text", label_column: "label"}
    )
    return df


def split_and_save(df, output_dir, seed):
    y = df["label"]
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        df, y, test_size=0.3, stratify=y, random_state=seed
    )
    val_size = 2 / 3
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=1 - val_size, stratify=y_tmp, random_state=seed
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(out / "train.csv", index=False)
    X_val.to_csv(out / "val.csv", index=False)
    X_test.to_csv(out / "test.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--label_column", default="label")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    df = load_from_source(args.source, args.text_column, args.label_column)
    split_and_save(df, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
