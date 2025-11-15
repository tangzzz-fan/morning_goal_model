import argparse
from pathlib import Path
import coremltools as ct
import onnx
from onnx_coreml import convert


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx_path", required=True)
    ap.add_argument("--output_dir", default="models/coreml")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    onnx_model = onnx.load(args.onnx_path)
    mlmodel = convert(onnx_model)
    mlmodel_path = out / "student_sequence_classification.mlmodel"
    ct.utils.save_spec(mlmodel.get_spec(), str(mlmodel_path))


if __name__ == "__main__":
    main()
