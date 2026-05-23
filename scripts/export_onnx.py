"""
Export the trained PyTorch model to ONNX format for cross-platform inference.

Usage:
    python scripts/export_onnx.py
    python scripts/export_onnx.py --model models/best_model.pth --out models/digit_recognizer.onnx
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.predict import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument(
        "--model", type=Path,
        default=Path(__file__).resolve().parents[1] / "models" / "best_model.pth",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).resolve().parents[1] / "models" / "digit_recognizer.onnx",
    )
    args = parser.parse_args()

    if not args.model.exists():
        print(f"[ERROR] Checkpoint not found: {args.model}")
        sys.exit(1)

    print(f"Loading model from {args.model} …")
    model = load_model(args.model, device=torch.device("cpu"))
    model.eval()

    dummy = torch.zeros(1, 1, 28, 28)

    print(f"Exporting to {args.out} …")
    torch.onnx.export(
        model, dummy, str(args.out),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"✓ ONNX model saved to {args.out}")


if __name__ == "__main__":
    main()
