from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import os

# For YOLO
import uuid
import base64
import torch
import cv2
from collections import Counter


app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DRONE = tf.keras.models.load_model(os.path.join("models", "drone", "1"))
CLASS_NAMES_DRONE = ['Green algae', 'Blue Green', 'Chrysophyta algae', 'Euglenophyta', 'Dinoflagellata', 'Protozoa']

MODEL_KAMERA_MIKROSKOP = tf.keras.models.load_model(os.path.join("models", "kamera-mikroskop", "6"))
CLASS_NAMES_KAMERA_MIKROSKOP = ['Corethron', 'bead', 'Dictyocha', 'clusterflagellate', 'Ciliate_mix', 'pennate']

MODEL_MIKROSKOP_DIGITAL = tf.keras.models.load_model(os.path.join("models", "mikroskop-digital", "model"))
YOLO_MIKROSKOP_DIGITAL = torch.hub.load('ultralytics/yolov5', 'custom', path=os.path.join("models", "mikroskop-digital", 'yolov5','runs', 'train', 'exp2', 'weights', 'best.pt'), force_reload=True)
# YOLO_MIKROSKOP_DIGITAL = torch.hub.load('ultralytics/yolov5', 'yolov5s')
CLASS_NAMES_MIKROSKOP_DIGITAL = ['Chaestoceros', 'Thallasiosira', 'Skeletonema', 'Nanochloropsis']

@app.get('/ping')
async def ping():
  return "FastAPI is Healthy"

@app.post('/predict/drone')
async def predict_drone(
  file: UploadFile = File(...) # We're also setting the default value
):
  # Convert the uploaded file read from bytes to numpy format
  image = read_file_as_image_and_normalization(await file.read())
  img_batch = np.expand_dims(image, 0)
  prob_result = MODEL_DRONE.predict(img_batch)
  sum = np.sum(prob_result)

  result = {}
  result["Green Algae"] = str(round(float(prob_result[0][0]/sum) * 100, 2)) + '%'
  result["BlueGreen Algae"] = str(round(float(prob_result[0][1]/sum) * 100, 2)) + '%'
  result["Chrysophyta"] = str(round(float(prob_result[0][2]/sum) * 100, 2)) + '%'
  result["Euglenophyta"] = str(round(float(prob_result[0][3]/sum) * 100, 2)) + '%'
  result["Dinoflagellata"] = str(round(float(prob_result[0][4]/sum) * 100, 2)) + '%'
  result["Protozoa"] = str(round(float(prob_result[0][5]/sum) * 100, 2)) + '%'
  return result

@app.post('/predict/mikroskop-digital')
async def predict_mikroskop_digital(
  file: UploadFile = File(...) # We're also setting the default value
):
  return_HTTP = {}

  # Object Detection
  image = read_file_as_image(await file.read())

  results = YOLO_MIKROSKOP_DIGITAL(image)
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  image_result = image.copy()

  for index, obj in enumerate(results.xyxy[0]): # Loop sejumlah objek yang dideteksi
      xMin = int(obj[0])
      yMin = int(obj[1])
      xMax = int(obj[2])
      yMax = int(obj[3])

      cv2.rectangle(image_result, (xMin, yMin), (xMax, yMax), (0, 0, 255), 1)

      crop = image[yMin:yMax, xMin:xMax]
      if not os.path.exists('mikroskop-digital-objects-result'):
          os.makedirs('mikroskop-digital-objects-result')
      imgname = os.path.join('mikroskop-digital-objects-result', '{}.jpg'.format(str(uuid.uuid1())))

      # Save every cropped object
      cv2.imwrite(imgname, crop)

  _, buffer = cv2.imencode('.jpg', image_result)
  # To see, add : data:image/jpeg;base64,
  return_HTTP['BYTE64IMAGE'] = 'data:image/jpeg;base64,' + str(base64.b64encode(buffer).decode("utf-8"))

  # Predict File
  images = []
  for result_file in os.listdir(os.path.join('mikroskop-digital-objects-result')):
    image = tf.keras.utils.load_img(os.path.join('mikroskop-digital-objects-result', result_file), target_size= (160, 160, 3))
    x = tf.keras.utils.img_to_array(image)
    images.append(x)
  prediction = MODEL_MIKROSKOP_DIGITAL.predict(np.array(images))
  result = [CLASS_NAMES_MIKROSKOP_DIGITAL[i] for i in np.argmax(prediction, axis=1)]
  result = dict(Counter(result))

  for k, v in result.items():
    # return_HTTP[str(list(result.keys()).index(k) + 1) + '.'] = str(v) + ' alga jenis ' + str(k)
    return_HTTP[str(k)] = v
  # return_HTTP['BYTE64IMAGE'] = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAApgAAAKYB3X3/OAAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAANCSURBVEiJtZZPbBtFFMZ/M7ubXdtdb1xSFyeilBapySVU8h8OoFaooFSqiihIVIpQBKci6KEg9Q6H9kovIHoCIVQJJCKE1ENFjnAgcaSGC6rEnxBwA04Tx43t2FnvDAfjkNibxgHxnWb2e/u992bee7tCa00YFsffekFY+nUzFtjW0LrvjRXrCDIAaPLlW0nHL0SsZtVoaF98mLrx3pdhOqLtYPHChahZcYYO7KvPFxvRl5XPp1sN3adWiD1ZAqD6XYK1b/dvE5IWryTt2udLFedwc1+9kLp+vbbpoDh+6TklxBeAi9TL0taeWpdmZzQDry0AcO+jQ12RyohqqoYoo8RDwJrU+qXkjWtfi8Xxt58BdQuwQs9qC/afLwCw8tnQbqYAPsgxE1S6F3EAIXux2oQFKm0ihMsOF71dHYx+f3NND68ghCu1YIoePPQN1pGRABkJ6Bus96CutRZMydTl+TvuiRW1m3n0eDl0vRPcEysqdXn+jsQPsrHMquGeXEaY4Yk4wxWcY5V/9scqOMOVUFthatyTy8QyqwZ+kDURKoMWxNKr2EeqVKcTNOajqKoBgOE28U4tdQl5p5bwCw7BWquaZSzAPlwjlithJtp3pTImSqQRrb2Z8PHGigD4RZuNX6JYj6wj7O4TFLbCO/Mn/m8R+h6rYSUb3ekokRY6f/YukArN979jcW+V/S8g0eT/N3VN3kTqWbQ428m9/8k0P/1aIhF36PccEl6EhOcAUCrXKZXXWS3XKd2vc/TRBG9O5ELC17MmWubD2nKhUKZa26Ba2+D3P+4/MNCFwg59oWVeYhkzgN/JDR8deKBoD7Y+ljEjGZ0sosXVTvbc6RHirr2reNy1OXd6pJsQ+gqjk8VWFYmHrwBzW/n+uMPFiRwHB2I7ih8ciHFxIkd/3Omk5tCDV1t+2nNu5sxxpDFNx+huNhVT3/zMDz8usXC3ddaHBj1GHj/As08fwTS7Kt1HBTmyN29vdwAw+/wbwLVOJ3uAD1wi/dUH7Qei66PfyuRj4Ik9is+hglfbkbfR3cnZm7chlUWLdwmprtCohX4HUtlOcQjLYCu+fzGJH2QRKvP3UNz8bWk1qMxjGTOMThZ3kvgLI5AzFfo379UAAAAASUVORK5CYII="

  # Delete images after predict
  for file in os.listdir(os.path.join('mikroskop-digital-objects-result')):
    if(os.path.isfile(os.path.join('mikroskop-digital-objects-result', file))):
      os.remove(os.path.join('mikroskop-digital-objects-result', file))

  return return_HTTP


def read_file_as_image_and_normalization(data) -> np.ndarray:
  image_1 = Image.open(BytesIO(data)).convert('RGB')
  image_2 = image_1.resize((160, 160))
  image_3 = np.array(image_2)
  image = np.true_divide(image_3, 255)
  return image

def read_file_as_image(data) -> np.ndarray:
  image_1 = Image.open(BytesIO(data)).convert('RGB')
  image_2 = image_1.resize((160, 160))
  image_3 = np.array(image_2)
  # image = np.true_divide(image_3, 255)
  return image_3

if __name__ == "__main__":
  uvicorn.run(app, host='0.0.0.0', port=8000)