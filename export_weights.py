

import numpy as np
import torch
import torch.nn as nn

# Same architecture as in train.py
net = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
)
net.load_state_dict(torch.load("models/mnist_model.pth", map_location="cpu"))
net.eval()

state = net.state_dict()
np.savez(
    "models/mnist_weights.npz",
    W1=state["0.weight"].numpy(), b1=state["0.bias"].numpy(),
    W2=state["2.weight"].numpy(), b2=state["2.bias"].numpy(),
    W3=state["4.weight"].numpy(), b3=state["4.bias"].numpy(),
)
print("Saved models/mnist_weights.npz")

# ---- Check: NumPy must match PyTorch ----
import model as numpy_model   # loads the file we just saved

x = torch.rand(200, 784)
with torch.no_grad():
    torch_logits = net(x).numpy()

a = x.numpy()
h1 = np.maximum(a @ numpy_model.W1.T + numpy_model.b1, 0)
h2 = np.maximum(h1 @ numpy_model.W2.T + numpy_model.b2, 0)
numpy_logits = h2 @ numpy_model.W3.T + numpy_model.b3

print("Largest difference between PyTorch and NumPy:", float(np.abs(torch_logits - numpy_logits).max()))
print("Same predictions on 200 random inputs:", bool((torch_logits.argmax(1) == numpy_logits.argmax(1)).all()))
