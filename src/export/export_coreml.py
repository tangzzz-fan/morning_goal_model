import argparse
from pathlib import Path
import coremltools as ct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx_path", required=True)
    ap.add_argument("--output_dir", default="models/coreml")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    mlmodel = ct.converters.onnx.convert(args.onnx_path)
    mlpackage = out / "student_sequence_classification.mlpackage"
    mlmodel.save(str(mlpackage))


if __name__ == "__main__":
    main()
