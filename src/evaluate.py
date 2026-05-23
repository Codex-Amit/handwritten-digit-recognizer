"""Evaluate the trained model on the MNIST test set."""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from src.data_loader import load_mnist

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "digit_cnn.h5")
MODELS_DIR = os.path.dirname(MODEL_PATH)


def evaluate():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run training first.")
    _, (x_test, y_test) = load_mnist()
    y_true = np.argmax(y_test, axis=1)
    model = tf.keras.models.load_model(MODEL_PATH)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n  Test Loss:     {loss:.4f}")
    print(f"  Test Accuracy: {acc * 100:.2f}%\n")
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    print(classification_report(y_true, y_pred, target_names=[str(i) for i in range(10)]))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, "confusion_matrix.png"))
    print(f"[INFO] Confusion matrix saved.")
    plt.close()


if __name__ == "__main__":
    evaluate()
