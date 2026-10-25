from pathlib import Path

import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent

# The same network as in train.py: 784 -> 128 -> 64 -> 10.
# Its weights were exported from PyTorch to a plain NumPy file, so the server
# does not need PyTorch (which is far too large for Vercel's 500 MB limit).
weights = np.load(BASE_DIR / "models" / "mnist_weights.npz")
W1, b1 = weights["W1"], weights["b1"]
W2, b2 = weights["W2"], weights["b2"]
W3, b3 = weights["W3"], weights["b3"]


def predict_digit(img):
    img = img.convert("L")
    img = img.resize((28, 28))

    pixels = np.array(img, dtype=np.float32) / 255.0
    x = pixels.flatten()                      # shape (784,)

    # Same maths as nn.Linear + nn.ReLU: y = x @ W.T + b
    h1 = np.maximum(x @ W1.T + b1, 0)         # Linear(784, 128) + ReLU
    h2 = np.maximum(h1 @ W2.T + b2, 0)        # Linear(128, 64)  + ReLU
    logits = h2 @ W3.T + b3                   # Linear(64, 10)

    return int(np.argmax(logits))
