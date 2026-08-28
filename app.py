from fastapi import FastAPI, UploadFile, File
from PIL import Image
from model import predict_digit
from fastapi.responses import FileResponse


app = FastAPI()

@app.get("/")  
def home():
    return FileResponse("index.html")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file)
    prediction=predict_digit(image)
    return {"digit":prediction}



