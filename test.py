import torch
import torch.nn as nn
from model import predict_digit
from PIL import Image

img=Image.open("six.png")

predict=predict_digit(img)

model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

print(f"PREDICTED DIGIT IS: {predict}")

