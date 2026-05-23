"""
export_model.py
Export the trained Keras model to TFLite format.
"""
import os
import tensorflow as tf

MODEL_PATH  = os.path.join("models", "digit_cnn.h5")
EXPORT_PATH = os.path.join("models", "digit_cnn.tflite")

def export():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(EXPORT_PATH, "wb") as f:
        f.write(tflite_model)
    size_kb = os.path.getsize(EXPORT_PATH) / 1024
    print(f"[INFO] TFLite model saved to {EXPORT_PATH} ({size_kb:.1f} KB)")

if __name__ == "__main__":
    export()
