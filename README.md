# 🔢 Handwritten Digit Recognizer

A Convolutional Neural Network (CNN) trained on the MNIST dataset to recognize handwritten digits (0–9) with a live web demo.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📸 Demo

Draw a digit on the canvas and watch the model predict it in real time!

> Run `python web/app.py` and open `http://localhost:5000`

---

## 🗂️ Project Structure

```
handwritten-digit-recognizer/
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Load & preprocess MNIST data
│   ├── model.py             # CNN architecture definition
│   ├── train.py             # Training loop & callbacks
│   └── evaluate.py          # Evaluation & metrics
│
├── models/
│   └── .gitkeep             # Saved model weights stored here
│
├── notebooks/
│   └── exploration.ipynb    # EDA & training walkthrough
│
├── tests/
│   ├── __init__.py
│   ├── test_model.py        # Unit tests for model
│   └── test_data_loader.py  # Unit tests for data pipeline
│
├── web/
│   ├── app.py               # Flask web app
│   ├── templates/
│   │   └── index.html       # Draw & predict UI
│   └── static/
│       ├── css/style.css
│       └── js/canvas.js
│
├── scripts/
│   ├── train_model.sh       # Shell script to train
│   └── export_model.py      # Export to TFLite / ONNX
│
├── docs/
│   └── architecture.md      # CNN architecture details
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your_username/handwritten-digit-recognizer.git
cd handwritten-digit-recognizer
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python -m src.train
# or use the shell script:
bash scripts/train_model.sh
```

The trained model will be saved to `models/digit_cnn.h5`.

### 3. Evaluate

```bash
python -m src.evaluate
```

### 4. Run the Web Demo

```bash
python web/app.py
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

---

## 🧠 Model Architecture

| Layer        | Details                        |
|-------------|-------------------------------|
| Input        | 28×28 grayscale image          |
| Conv2D       | 32 filters, 3×3, ReLU          |
| MaxPooling2D | 2×2                            |
| Conv2D       | 64 filters, 3×3, ReLU          |
| MaxPooling2D | 2×2                            |
| Flatten      | —                              |
| Dense        | 128 units, ReLU, Dropout 0.5   |
| Output       | 10 units, Softmax              |

**Test Accuracy: ~99.2%** on MNIST test set.

---

## 📊 Results

| Metric     | Value  |
|-----------|--------|
| Train Acc  | 99.6%  |
| Test Acc   | 99.2%  |
| Parameters | ~93K   |
| Epochs     | 10     |

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **TensorFlow / Keras** — model training
- **NumPy / Matplotlib** — data processing & visualization
- **Flask** — web demo backend
- **HTML5 Canvas** — frontend drawing interface

---

## 📓 Notebook

See `notebooks/exploration.ipynb` for a full walkthrough including:
- Dataset exploration
- Training curves
- Confusion matrix
- Sample predictions

---

## 🧪 Tests

```bash
pytest tests/
```

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) — Yann LeCun et al.
- [TensorFlow](https://www.tensorflow.org/) team
