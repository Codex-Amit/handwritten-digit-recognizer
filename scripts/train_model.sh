#!/usr/bin/env bash
# Run model training from project root.
set -e
echo "==> Starting training..."
python -m src.train "$@"
echo "==> Training complete. Model saved to models/digit_cnn.h5"
