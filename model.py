import torch
import torch.nn as nn
from PIL import Image
import numpy as np


model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)


model.load_state_dict(
    torch.load("models/mnist_model.pth")
)

model.eval()


def predict_digit(img):
    
    print(img.size)
    print(img.mode)
    img=img.convert('L')
    img=img.resize((28,28))

    pixels=np.array(img)

    # ch=int(input("Does the image contain black background? 1-YES    0-NO    "))
    # if(ch==0):
    #     pixels = 255 - pixels
        
    print(pixels.shape)
    print(pixels)

    img_tensor=torch.tensor(pixels,dtype=torch.float32)

    img_tensor = img_tensor / 255.0
    img_tensor = img_tensor.flatten().unsqueeze(0)

    with torch.no_grad():
        logits_newimg=model(img_tensor)

    predictions=torch.argmax(logits_newimg,dim=1)
    
    return predictions.item()