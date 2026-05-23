"""Unit tests for data loading pipeline."""
import numpy as np
from src.data_loader import load_mnist, preprocess_image


def test_load_mnist_shapes():
    (x_train, y_train), (x_test, y_test) = load_mnist()
    assert x_train.shape == (60000, 28, 28, 1)
    assert x_test.shape  == (10000, 28, 28, 1)
    assert y_train.shape == (60000, 10)
    assert y_test.shape  == (10000, 10)

def test_pixel_range():
    (x_train, _), _ = load_mnist()
    assert x_train.max() <= 1.0
    assert x_train.min() >= 0.0

def test_preprocess_image():
    raw = np.random.randint(0, 256, (28, 28), dtype=np.uint8)
    out = preprocess_image(raw)
    assert out.shape == (1, 28, 28, 1)
    assert out.dtype == np.float32
    assert out.max() <= 1.0
