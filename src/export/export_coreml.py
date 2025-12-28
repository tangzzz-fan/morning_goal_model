import argparse
from pathlib import Path
import torch
import coremltools as ct
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Path to the PyTorch model directory")
    ap.add_argument("--output_dir", default="models/coreml", help="Directory to save the CoreML model")
    ap.add_argument("--seq_len", type=int, default=128, help="Sequence length for tracing")
    args = ap.parse_args()

    model_path = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    # Wrapper to return tuple (logits,) instead of SequenceClassifierOutput
    class WrapperModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, input_ids, attention_mask):
            return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    wrapper_model = WrapperModel(model)

    # Create dummy input for tracing
    print("Tracing model...")
    text = "这是一个测试句子"
    inputs = tokenizer(
        text, 
        max_length=args.seq_len, 
        padding="max_length", 
        truncation=True, 
        return_tensors="pt"
    )
    
    # Trace the model
    # We trace with (input_ids, attention_mask)
    dummy_input = (inputs["input_ids"], inputs["attention_mask"])
    traced_model = torch.jit.trace(wrapper_model, dummy_input)

    # Define input types for CoreML
    # Using RangeDim for dynamic batch size, fixed sequence length for mobile efficiency
    input_tensors = [
        ct.TensorType(name="input_ids", shape=(1, args.seq_len), dtype=int),
        ct.TensorType(name="attention_mask", shape=(1, args.seq_len), dtype=int)
    ]

    print("Converting to CoreML...")
    # Get class labels from model config
    id2label = model.config.id2label
    if id2label:
        labels = [id2label[i] for i in sorted(id2label.keys())]
    else:
        labels = [f"label_{i}" for i in range(model.config.num_labels)]
    
    print(f"Using class labels: {labels}")

    mlmodel = ct.convert(
        traced_model,
        inputs=input_tensors,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS15,
        compute_units=ct.ComputeUnit.ALL,
        classifier_config=ct.ClassifierConfig(class_labels=labels)
    )

    output_path = output_dir / "student_sequence_classification.mlpackage"
    print(f"Saving to {output_path}...")
    mlmodel.save(str(output_path))
    print("Done!")

if __name__ == "__main__":
    main()
