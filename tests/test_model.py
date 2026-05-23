"""Unit tests for model architecture."""
import pytest
import numpy as np
import tensorflow as tf
from src.model import build_model


def test_model_output_shape():
    model = build_model()
    dummy = np.zeros((4, 28, 28, 1), dtype="float32")
    preds = model.predict(dummy, verbose=0)
    assert preds.shape == (4, 10), "Output should be (batch, 10)"

def test_model_probabilities_sum_to_one():
    model = build_model()
    dummy = np.random.rand(1, 28, 28, 1).astype("float32")
    preds = model.predict(dummy, verbose=0)
    assert abs(preds[0].sum() - 1.0) < 1e-5

def test_model_has_layers():
    model = build_model()
    layer_types = [type(l).__name__ for l in model.layers]
    assert "Conv2D" in layer_types
    assert "Dense" in layer_types
    assert "Dropout" in layer_types
