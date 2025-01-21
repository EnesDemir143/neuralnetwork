from typing import Annotated
import numpy as np
from tensorflow.keras.preprocessing import image
from io import BytesIO
from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware


model = load_model("/Users/enesdemir/Desktop/neuralnetwork/model.keras")

app = FastAPI()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):

    img_bytes = await file.read()

    img = image.load_img(BytesIO(img_bytes), target_size=(32, 32,3)) 
    img_array = image.img_to_array(img) 
    img_array = img_array.astype('float32') / 255.0  
    
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)

    predicted_class = np.argmax(predictions)
    predicted_class_name = class_names[predicted_class]

    return {"filename": file.filename, "predicted_class": predicted_class_name}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)