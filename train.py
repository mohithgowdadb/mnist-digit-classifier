import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from PIL import Image
import numpy as np

df=pd.read_csv("train.csv")

print(df.shape)
print(df.info())
print(df.head())

label_counts = df["label"].value_counts().sort_index()
print(label_counts)

label_counts.plot(kind="bar")
plt.xlabel("Digit")
plt.ylabel("Number of images")
plt.title("Number of Images per Digit")
plt.show()

X=df.drop("label",axis=1)
y=df["label"]

# print(X[:5])
# print(y[:5])
# print(X.shape)
# print(y.shape)

X=X/255.0

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42) 

X_train= torch.tensor(X_train.values,dtype=torch.float32)
X_test= torch.tensor(X_test.values,dtype=torch.float32)
y_train= torch.tensor(y_train.values,dtype=torch.long)
y_test= torch.tensor(y_test.values,dtype=torch.long)

# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

model=nn.Sequential(
    nn.Linear(784,128),
    nn.ReLU(),
    nn.Linear(128,64),
    nn.ReLU(),
    nn.Linear(64,10),   
)

loss=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

itr=50

for epoch in range(itr):
    p=model(X_train)
    l=loss(p,y_train)
    optimizer.zero_grad()
    l.backward()
    optimizer.step()
    
    
    print(f"Epoch {epoch+1}, Loss: {l.item():.4f}")
    
model.eval()

with torch.no_grad():
    logits=model(X_test)
    
probabilites=torch.softmax(logits,dim=1)
predictions=torch.argmax(logits,dim=1)

print(probabilites)
print(predictions)

correct=(predictions==y_test).sum()
acc=correct/len(y_test)
print(f"Test Accuracy: {acc.item()*100:.2f}%")


cm = confusion_matrix(y_test, predictions)

plt.imshow(cm)  
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.colorbar()
plt.show()


torch.save(model.state_dict(), "models/mnist_model.pth")
print("Model saved!")




