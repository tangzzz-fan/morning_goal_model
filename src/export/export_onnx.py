import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_student(model_dir):
    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_dir)
    mdl.eval()
    return tok, mdl


def export_onnx(model, sample_enc, output_path, opset):
    inputs = {}
    for k in ["input_ids", "attention_mask", "token_type_ids"]:
        if k in sample_enc:
            t = torch.tensor(sample_enc[k][:1])
            inputs[k] = t
    dynamic_axes = {k: {0: "batch", 1: "seq"} for k in inputs.keys()}
    torch.onnx.export(
        model,
        (
            inputs["input_ids"],
            inputs.get("attention_mask"),
            inputs.get("token_type_ids"),
        ),
        str(output_path),
        input_names=list(inputs.keys()),
        output_names=["logits"],
        dynamic_axes=dynamic_axes | {"logits": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
    )


def run_pt(model, enc_batch):
    tens = {k: torch.tensor(v) for k, v in enc_batch.items()}
    with torch.no_grad():
        logits = model(**tens).logits.cpu().numpy()
    return logits


def run_onnx(session, enc_batch):
    feeds = {}
    for k in ["input_ids", "attention_mask", "token_type_ids"]:
        if k in enc_batch and k in {i.name for i in session.get_inputs()}:
            feeds[k] = np.asarray(enc_batch[k])
    outputs = session.run([session.get_outputs()[0].name], feeds)
    return outputs[0]


def encode_texts(tokenizer, texts, max_length):
    enc = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    return enc


def evaluate_consistency(model, tokenizer, session, texts, max_length):
    enc = encode_texts(tokenizer, texts, max_length)
    logits_pt = run_pt(model, enc)
    logits_onnx = run_onnx(session, enc)
    preds_pt = logits_pt.argmax(axis=-1)
    preds_onnx = logits_onnx.argmax(axis=-1)
    match = (preds_pt == preds_onnx).mean().item()
    mae = np.abs(logits_pt - logits_onnx).mean().item()
    mse = ((logits_pt - logits_onnx) ** 2).mean().item()
    return {"match_rate": match, "mae": mae, "mse": mse}


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

    tokenizer, model = load_student(args.model_dir)
    val_df = pd.read_csv(Path(args.data_dir) / "val.csv")
    sample_texts = val_df["text"].tolist()[: args.samples]
    sample_enc = encode_texts(tokenizer, sample_texts, args.max_length)

    onnx_path = out / "student_sequence_classification.onnx"
    export_onnx(model, sample_enc, onnx_path, args.opset)

    sess = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    metrics = evaluate_consistency(
        model,
        tokenizer,
        sess,
        sample_texts,
        args.max_length,
    )

    report_md = out / "onnx_evaluation_report.md"
    md = []
    md.append("# ONNX 一致性评估报告")
    md.append("")
    md.append(f"模型目录: {args.model_dir}")
    md.append(f"ONNX 路径: {onnx_path.name}")
    md.append(f"样本数: {args.samples}")
    md.append(f"opset: {args.opset}")
    md.append("")
    md.append(f"匹配率: {metrics['match_rate']:.6f}")
    md.append(f"MAE: {metrics['mae']:.6f}")
    md.append(f"MSE: {metrics['mse']:.6f}")
    md.append("")
    Path(report_md).write_text("\n".join(md), encoding="utf-8")

    summary = {"report": str(report_md), "metrics": metrics}
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
