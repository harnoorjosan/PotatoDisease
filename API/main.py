from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
from keras import models

app = FastAPI()
MODEL = models.load_model("potato_trained.keras")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello World!"


def read_file_as_img(data):
    img = np.array(Image.open(BytesIO(data)))
    return img

''' 
Synchronous:
Second request can be processed only after first is finished.
This causes lag.

Asynchronous (await and async):
Second request can be processed even before finishing the first request.
This doesn't cause lag
'''

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = read_file_as_img(await file.read()) 
    img_batch = np.expand_dims(img,0)
    prediction = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    return {'class': predicted_class, 'confidence': float(confidence)}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port = 8000)