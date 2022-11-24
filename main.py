from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import os

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

MODEL = tf.keras.models.load_model(os.path.join("models", "daun-solo", "1"))
CLASS_NAMES = ['Adas', 'Andong', 'Beluntas', 'Bidara', 'Brotowali', 'Ciplukan', 'Ginseng Jawa', 'Jahe Merah', 'Jinten', 'Kecubung', 'Kejibeling', 'Kelor', 'Kenanga', 'Kumis Kucing', 'Laos', 'Legundi', 'Lemon', 'Mangkok', 'Patikim', 'Pegagan', 'Pepaya', 'Salam', 'Sambiloto', 'Sambung Nyawa', 'Sepatu', 'Sirih', 'Sirih Cina', 'Telang', 'Ungu', 'Yodium']

# 66 Jenis
# CLASS_NAMES = ['Amaranthus viridis (Bayam hijau)', 'Andrographis paniculata (Sambiloto)', 'Anredera cordifolia (Binahong)', 'Artocarpus heterophyllus (Nangka)', 'Averrhoa bilimbi (Belimbing sayur)', 'Azadirachta indica (Mimba)', 'Basella alba (Bayam malabar)', 'Boesenbergia rotunda (Kunci)', 'Brassica juncea (Sawi india)', 'Cananga odorata (Kenanga)', 'Carica Papaya (Pepaya)', 'Carissa carandas (Buah samarinda)', 'Citrus limon (Lemon)', 'Citrus sinensis (Jeruk Manis)', 'Citrus xamblycarpa (Jeruk Limau)', 'Citrus × aurantiifolia (Jeruk nipis)', 'Clinacanthus nutans (Dandang gendis)', 'Clitoria ternatea (Telang)', 'Cnidoscolus aconitifolius (Pepaya Jepang)', 'Datura metel (Kecubung)', 'Ficus auriculata (Pohon ara)', 'Ficus religiosa (Pohon bodhi)', 'Garcinia mangostana (Manggis)', 'Graptophyllum pictum (Ungu)', 'Gynura procumbens (Sambung nyowo)', 'Hibiscus rosa-sinensis (Bunga sepatu)', 'Jasminum (Melati)', 'Jatropha curcas (Jarak Pagar)', 'Jatropha multifida L. (Yudium)', 'Kaempferia galanga (Kencur)', 'Mangifera indica (Mangga)', 'Melaleuca leucadendra (Kayu Putih)', 'Mentha (Mint)', 'Muntingia calabura (Kersen)', 'Murraya koenigii (Salam koja)', 'Nerium oleander (Bunga jepun)', 'Nyctanthes arbor-tristis (Srigading)', 'Ocimum tenuiflorum (Ruku-ruku)', 'Orthosiphon aristatus (Kumis kucing)', 'Panax (Ginseng)', 'Phaleria macrocarpa (Mahkota dewa)', 'Philodendron Burle-marx (Philo Brekele)', 'Physalis angulata L. (Ciplukan)', 'Piper betle (Sirih)', 'Plectranthus amboinicus (Daun jintan)', 'Pluchea indica (Beluntas)', 'Polyscias scutellaria (Mangkokan)', 'Pongamia pinnata (Malapari)', 'Premna serratifolia (Waung)', 'Psidium guajava (Jambu biji)', 'Punica granatum (Delima)', 'Ruellia napifera (Gempur Batu)', 'Santalum album (Cendana)', 'Stachytarpheta jamaicensis (Pecut kuda)', 'Strobilanthes crispa (Keji beling)', 'Syzygium cumini (Jamblang)', 'Syzygium jambos (Jambu mawar)', 'Syzygium polyanthum (Salam)', 'Tabernaemontana divaricata (Mondokaki)', 'Talinum paniculatum (Ginseng jawa)', 'Tinospora cordifolia (Brotowali)', 'Trigonella foenum-graecum (Kelabat)', 'Vitex trifolia (Legundi)', 'Ziziphus jujuba (Apel india)', 'Ziziphus mauritiana (Bidara)', 'lsotoma longiflora (Ki Tolod)']

@app.get('/ping')
async def ping():
  return "FastAPI is Healthy"

@app.post('/predict')
async def predict(
  file: UploadFile = File(...) # We're also setting the default value
):
  # Convert the uploaded file read from bytes to numpy format
  image = read_file_as_image(await file.read())
  img_batch = np.expand_dims(image, 0)
  predictions = MODEL.predict(img_batch)
  arg_max = np.argmax(predictions[0])
  predicted_class = CLASS_NAMES[arg_max]
  confidence = np.max(predictions[0])
  return {
      'class': predicted_class,
      'confidence': float(confidence)
  }

def read_file_as_image(data) -> np.ndarray:
  image_1 = Image.open(BytesIO(data)).convert('RGB')
  image_2 = image_1.resize((224, 224))
  image_3 = np.array(image_2)
  # image_4 = np.true_divide(image_3, 255)
  
  image = np.array(image_3)
  return image

if __name__ == "__main__":
  uvicorn.run(app, host='0.0.0.0', port=8000)