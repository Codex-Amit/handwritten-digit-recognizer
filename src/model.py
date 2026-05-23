"""CNN architecture for handwritten digit recognition."""
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
)


def build_model(input_shape=(28, 28, 1), num_classes=10) -> Sequential:
    model = Sequential([
        Input(shape=input_shape),
        # Block 1
        Conv2D(32, 3, padding="same", activation="relu"),
        BatchNormalization(),
        Conv2D(32, 3, padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling2D(2),
        Dropout(0.25),
        # Block 2
        Conv2D(64, 3, padding="same", activation="relu"),
        BatchNormalization(),
        Conv2D(64, 3, padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling2D(2),
        Dropout(0.25),
        # Head
        Flatten(),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ], name="digit_cnn")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


if __name__ == "__main__":
    build_model().summary()
