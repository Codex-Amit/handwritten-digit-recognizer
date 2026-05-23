"""
Predict the digit in one or more image files.

Usage:
    python scripts/predict_image.py path/to/digit.png
    python scripts/predict_image.py img1.png img2.png img3.png
    python scripts/predict_image.py --model models/best_model.pth img.png
"""

import argparse
import sys
from pathlib import Path

# Allow running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.predict import load_model, predict_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict digits from image files")
    parser.add_argument("images", nargs="+", type=Path, help="Image file(s) to classify")
    parser.add_argument(
        "--model", type=Path,
        default=Path(__file__).resolve().parents[1] / "models" / "best_model.pth",
        help="Path to model checkpoint",
    )
    args = parser.parse_args()

    if not args.model.exists():
        print(f"[ERROR] Model checkpoint not found: {args.model}")
        print("  Run `python main.py` first to train and save the model.")
        sys.exit(1)

    print(f"Loading model from {args.model} …")
    model = load_model(args.model)

    print(f"\nPredicting {len(args.images)} image(s):\n{'─'*40}")
    results = predict_batch(model, args.images)

    print(f"\n{'─'*40}")
    print(f"Done. {len(results)} image(s) processed.")


if __name__ == "__main__":
    main()
