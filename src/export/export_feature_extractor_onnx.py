import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModel


class FeatureExtractor(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base

    def mean_pool(self, hidden, mask):
        m = mask.unsqueeze(-1).type_as(hidden)
        masked = hidden * m
        summed = masked.sum(dim=1)
        lengths = m.sum(dim=1).clamp(min=1)
        return summed / lengths

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden = out.last_hidden_state
        emb = self.mean_pool(hidden, attention_mask)
        return emb


def encode(tokenizer, texts, max_length):
    return tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
    )


def export(model, enc, path, opset):
    inputs = {}
    for k in ["input_ids", "attention_mask", "token_type_ids"]:
        if k in enc:
            inputs[k] = torch.tensor(enc[k][:1])
    dyn = {k: {0: "batch", 1: "seq"} for k in inputs.keys()}
    torch.onnx.export(
        model,
        (
            inputs["input_ids"],
            inputs.get("attention_mask"),
            inputs.get("token_type_ids"),
        ),
        str(path),
        input_names=list(inputs.keys()),
        output_names=["embedding"],
        dynamic_axes=dyn | {"embedding": {0: "batch", 1: "feat"}},
        opset_version=opset,
        do_constant_folding=True,
    )


def run_pt(model, enc):
    tens = {k: torch.tensor(v) for k, v in enc.items()}
    with torch.no_grad():
        emb = (
            model(
                tens["input_ids"],
                tens.get("attention_mask"),
                tens.get("token_type_ids"),
            )
            .cpu()
            .numpy()
        )
    return emb


def run_onnx(sess, enc):
    feeds = {}
    names = {i.name for i in sess.get_inputs()}
    for k in ["input_ids", "attention_mask", "token_type_ids"]:
        if k in enc and k in names:
            feeds[k] = np.asarray(enc[k])
    out = sess.run([sess.get_outputs()[0].name], feeds)[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="models/trained/distill_student")
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--output_dir", default="models/onnx")
    ap.add_argument("--opset", type=int, default=14)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--samples", type=int, default=64)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    base = AutoModel.from_pretrained(args.model_dir)
    base.eval()
    fx = FeatureExtractor(base)

    val_df = pd.read_csv(Path(args.data_dir) / "val.csv")
    texts = val_df["text"].tolist()[: args.samples]
    enc = encode(tokenizer, texts, args.max_length)

    onnx_path = out / "feature_extractor.onnx"
    export(fx, enc, onnx_path, args.opset)

    sess = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    pt = run_pt(fx, enc)
    ox = run_onnx(sess, enc)
    mae = np.abs(pt - ox).mean().item()
    mse = ((pt - ox) ** 2).mean().item()
    if pt.ndim == 2 and ox.ndim == 2:
        match = (pt.argmax(axis=-1) == ox.argmax(axis=-1)).mean().item()
    else:
        match = 1.0

    md = []
    md.append("# ONNX 特征提取一致性评估")
    md.append("")
    md.append(f"模型目录: {args.model_dir}")
    md.append(f"ONNX 路径: {onnx_path.name}")
    md.append(f"样本数: {args.samples}")
    md.append(f"opset: {args.opset}")
    md.append("")
    md.append(f"MAE: {mae:.6f}")
    md.append(f"MSE: {mse:.6f}")
    md.append(f"伪匹配率: {match:.6f}")
    Path(out / "feature_onnx_evaluation_report.md").write_text(
        "\n".join(md), encoding="utf-8"
    )

    print(json.dumps({"mae": mae, "mse": mse}, ensure_ascii=False))


if __name__ == "__main__":
    main()
