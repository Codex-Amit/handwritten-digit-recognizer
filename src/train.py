"""Train the CNN on MNIST and save the best checkpoint."""
import os, argparse
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from src.data_loader import load_mnist
from src.model import build_model

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=15)
    p.add_argument("--batch_size", type=int,   default=128)
    return p.parse_args()


def train(epochs=15, batch_size=128):
    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path = os.path.join(MODELS_DIR, "digit_cnn.h5")
    print("[INFO] Loading MNIST...")
    (x_train, y_train), (x_test, y_test) = load_mnist()
    model = build_model()
    model.summary()
    callbacks = [
        ModelCheckpoint(save_path, save_best_only=True, monitor="val_accuracy", verbose=1),
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
        TensorBoard(log_dir=os.path.join(MODELS_DIR, "logs")),
    ]
    history = model.fit(x_train, y_train, validation_split=0.1,
                        epochs=epochs, batch_size=batch_size, callbacks=callbacks)
    _plot(history)
    print(f"[INFO] Model saved to {save_path}")
    return model, history


def _plot(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, title in zip(axes, ["accuracy", "loss"], ["Accuracy", "Loss"]):
        ax.plot(history.history[key],         label="Train")
        ax.plot(history.history[f"val_{key}"], label="Val")
        ax.set_title(title); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, "training_curves.png"))
    plt.close()


if __name__ == "__main__":
    args = parse_args()
    train(args.epochs, args.batch_size)
