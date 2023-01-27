from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import os

from db import Database
db = Database()

app = FastAPI()

# origins = [
#     "http://localhost",
#     "http://localhost:3000",
# ]

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_A = tf.keras.models.load_model(os.path.join("models", "InceptionV3", "daun-solo", "best_model"))
MODEL_B = tf.keras.models.load_model(os.path.join("models", "InceptionV3", "daun-nempel", "best_model"))
MODEL_C = tf.keras.models.load_model(os.path.join("models", "InceptionV3", "keseluruhan", "5"))
MODEL_D = tf.keras.models.load_model(os.path.join("models", "InceptionV3", "batang", "5"))

CLASS_NAMES = ['Adas', 'Andong', 'Beluntas', 'Bidara', 'Brotowali', 'Ciplukan', 'Ginseng Jawa', 'Jahe Merah', 'Jinten', 'Kecubung', 'Kejibeling', 'Kelor', 'Kenanga', 'Kumis Kucing', 'Laos', 'Legundi', 'Lemon', 'Mangkok', 'Patikim', 'Pegagan', 'Pepaya', 'Salam', 'Sambiloto', 'Sambung Nyawa', 'Sepatu', 'Sirih', 'Sirih Cina', 'Telang', 'Ungu', 'Yodium']

@app.get('/ping')
async def ping():
  return "FastAPI is Healthy"

@app.get('/getPlantByName')
async def getPlantByName(nama_umum: str):
  real_nama_umum = nama_umum.replace("%20", " ")
  return db.query(query=f"SELECT * FROM herbals WHERE nama_umum='{real_nama_umum}';")

@app.post('/predict-daun-solo')
async def predict(
  file: UploadFile = File(...) # We're also setting the default value
):
  return await predict(file, "A")

@app.post('/predict-daun-nempel')
async def predict(
  file: UploadFile = File(...) # We're also setting the default value
):
  return await predict(file, "B")

@app.post('/predict-keseluruhan')
async def predict(
  file: UploadFile = File(...) # We're also setting the default value
):
  return await predict(file, "C")

@app.post('/predict-batang')
async def predict(
  file: UploadFile = File(...) # We're also setting the default value
):
  return await predict(file, "D")



async def predict(file, datatype):
  # Convert the uploaded file read from bytes to numpy format
  image = read_file_as_image(await file.read())
  img_batch = np.expand_dims(image, 0)
  
  if datatype == "A":
    predictions = MODEL_A.predict(img_batch)
  elif datatype == "B":
    predictions = MODEL_B.predict(img_batch)
  elif datatype == "C":
    predictions = MODEL_C.predict(img_batch)
  else:
    predictions = MODEL_D.predict(img_batch)
  arg_max = np.argmax(predictions[0])
  predicted_class = CLASS_NAMES[arg_max]
  confidence = np.max(predictions[0])
  return {
      'class': predicted_class,
      'confidence': float(confidence)
  }

def read_file_as_image(data) -> np.ndarray:
  image_1 = Image.open(BytesIO(data)).convert('RGB')
  image_2 = image_1.resize((299, 299))
  image_3 = np.array(image_2)
  # image_4 = np.true_divide(image_3, 255)
  
  image = np.array(image_3)
  return image

if __name__ == "__main__":
  uvicorn.run(app, host='0.0.0.0', port=8000)