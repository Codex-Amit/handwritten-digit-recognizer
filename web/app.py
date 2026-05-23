"""
app.py - Flask web server for the digit recognizer demo.
"""
import os, sys, base64
import numpy as np
from io import BytesIO
from PIL import Image, ImageOps
try:
    import tensorflow as tf
except ImportError:
    tf = None
from flask import Flask, render_template, request, jsonify

RESAMPLE_LANCZOS = getattr(Image, "Resampling", Image).LANCZOS
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

app = Flask(__name__)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "digit_cnn.h5")
model = None


def load_model():
    global model
    if tf is None:
        print("[ERROR] TensorFlow is not installed.")
        return
    if not os.path.exists(MODEL_PATH):
        print(f"[WARNING] Model not found at {MODEL_PATH}. Run training first.")
        return
    model = tf.keras.models.load_model(MODEL_PATH)
    print("[INFO] Model loaded.")


def preprocess(img: Image.Image) -> np.ndarray:
    # Convert to grayscale
    img = img.convert("L")

    # Canvas is now white background + black digit
    # MNIST is white digit on black — so invert
    img = ImageOps.invert(img)

    # Crop tightly to the digit bounding box
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    else:
        # Empty canvas — return blank
        return np.zeros((1, 28, 28, 1), dtype="float32")

    # Add padding so digit isn't edge-to-edge (mimics MNIST centering)
    pad = max(img.size) // 4
    img = ImageOps.expand(img, border=pad, fill=0)

    # Resize to 28x28
    img = img.resize((28, 28), RESAMPLE_LANCZOS)

    arr = np.array(img).astype("float32") / 255.0
    return arr.reshape(1, 28, 28, 1)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if tf is None:
        return jsonify({"error": "TensorFlow is not available."}), 503
    if model is None:
        return jsonify({"error": "Model not loaded. Train first."}), 503

    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "No image data."}), 400

    try:
        image_data = data["image"]
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]
        img_bytes = base64.b64decode(image_data)
        img = Image.open(BytesIO(img_bytes))
    except Exception:
        return jsonify({"error": "Invalid image data."}), 400

    try:
        arr = preprocess(img)
        preds = model.predict(arr, verbose=0)[0]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    digit = int(np.argmax(preds))
    return jsonify({
        "digit": digit,
        "confidence": round(float(np.max(preds)) * 100, 2),
        "probabilities": {str(i): round(float(preds[i]) * 100, 2) for i in range(10)},
    })


if __name__ == "__main__":
    load_model()
    app.run(debug=True, port=5000)