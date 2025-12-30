import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse
import sys

def verify_model(model_path):
    print(f"Loading model from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()
    print("\nModel loaded successfully!")
    print("Type sentences to test (type 'q' or 'quit' to exit):")
    print("-" * 50)

    while True:
        try:
            text = input("\nInput Text > ")
            if text.lower() in ('q', 'quit', 'exit'):
                break
            
            if not text.strip():
                continue

            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
            
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][predicted_class].item()

            print(f"Prediction:")
            print(f"  Topic Label ID: {predicted_class}")
            print(f"  Confidence:     {confidence:.4f}")
            print(f"  Top 3 Probs:    {torch.topk(probabilities[0], 3).values.tolist()}")
            print(f"  Top 3 Indices:  {torch.topk(probabilities[0], 3).indices.tolist()}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error during inference: {e}")

    print("\nExiting...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify trained MobileBERT model")
    parser.add_argument("--model_path", default="models/trained/distill_student", help="Path to the model")
    args = parser.parse_args()
    
    verify_model(args.model_path)
