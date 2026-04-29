import os
import json
import pickle
import traceback
from datetime import datetime, timezone
from collections import deque

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from ultralytics import YOLO

# ------------------------------------------------------------
# Configurações
# ------------------------------------------------------------
MODEL_PATH = "petzy_model.keras"          # Classificador de dor
CLASS_INDICES_PATH = "class_indices.json"
MODEL_INFO_PATH = "model_info.pkl"

YOLO_MODEL_PATH = "yolov8n.pt"            # YOLO COCO (cão/gato)
HISTORY_LEN = 5
THRESHOLD = 0.7

# ------------------------------------------------------------
# Inicialização do FastAPI
# ------------------------------------------------------------
app = FastAPI(title="Petzy Pain Analyzer")

# 🔧 HABILITAR CORS - necessário para o frontend rodar em origem diferente
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, restrinja para o domínio do seu frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globais
model = None          # classificador de dor
yolo = None           # detector YOLO
input_size = 224
class_indices = {}
score_history = deque(maxlen=HISTORY_LEN)

# ------------------------------------------------------------
# Carregamento dos modelos
# ------------------------------------------------------------
def load_artifacts():
    global model, yolo, input_size, class_indices

    # 1. Carregar YOLO
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"⚠️ YOLO model não encontrado em {YOLO_MODEL_PATH}. Baixando...")
        yolo = YOLO(YOLO_MODEL_PATH)  # baixa automaticamente
    else:
        yolo = YOLO(YOLO_MODEL_PATH)
    print("✅ YOLO carregado")

    # 2. Carregar classificador de dor
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modelo não encontrado: {MODEL_PATH}")
    if not os.path.exists(CLASS_INDICES_PATH):
        raise FileNotFoundError(f"Arquivo de classes não encontrado: {CLASS_INDICES_PATH}")
    if not os.path.exists(MODEL_INFO_PATH):
        raise FileNotFoundError(f"Arquivo de metadados não encontrado: {MODEL_INFO_PATH}")

    model = load_model(MODEL_PATH)
    print(f"✅ Classificador carregado de {MODEL_PATH}")

    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)   # ex: {"dor":0, "sem_dor":1}
    print(f"✅ Classes: {class_indices}")

    with open(MODEL_INFO_PATH, "rb") as f:
        info = pickle.load(f)
    input_size = info.get("input_size", 224)
    print(f"✅ input_size = {input_size}")

try:
    load_artifacts()
except Exception as e:
    print(f"FALHA CRÍTICA: {e}")
    # Em produção, você pode querer encerrar ou usar mock

# ------------------------------------------------------------
# Funções auxiliares
# ------------------------------------------------------------
def preprocess_image_crop(image_bytes: bytes, bbox):
    """
    Recorta a região da bounding box (x, y, w, h), redimensiona e pré-processa.
    Retorna tensor pronto para o classificador e as dimensões originais do recorte.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Imagem inválida")

    x, y, w, h = bbox
    # Garantir que a bbox esteja dentro da imagem
    x = max(0, x)
    y = max(0, y)
    w = min(w, img.shape[1] - x)
    h = min(h, img.shape[0] - y)
    if w <= 0 or h <= 0:
        raise HTTPException(status_code=400, detail="Bbox inválida")

    crop = img[y:y+h, x:x+w]
    if crop.size == 0:
        raise HTTPException(status_code=400, detail="Recorte vazio")

    # Redimensionar e pré-processar
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(crop_rgb, (input_size, input_size))
    preprocessed = preprocess_input(resized.astype(np.float32))
    batch = np.expand_dims(preprocessed, axis=0)
    return batch, (crop.shape[1], crop.shape[0])  # (largura, altura)

def get_risk_level(score: float) -> str:
    if score < 0.35:
        return "normal"
    elif score < 0.65:
        return "atencao"
    else:
        return "critico"

# ------------------------------------------------------------
# Endpoint principal
# ------------------------------------------------------------
@app.post("/analyze-frame")
async def analyze_frame(file: UploadFile = File(...)):
    if model is None or yolo is None:
        raise HTTPException(status_code=503, detail="Modelos não carregados")

    try:
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Arquivo vazio")

        # 1. YOLO detecta cães (classe 16) e gatos (classe 15) do COCO
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        results = yolo(img_cv)
        detections = results[0].boxes
        if detections is None:
            animals = []
        else:
            animals = []
            for box in detections:
                cls = int(box.cls[0])
                if cls in [15, 16]:  # 15: cat, 16: dog
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    w = x2 - x1
                    h = y2 - y1
                    bbox = [x1, y1, w, h]
                    species = "gato" if cls == 15 else "cao"

                    # Classificar dor nesta região
                    try:
                        img_tensor, (crop_w, crop_h) = preprocess_image_crop(image_bytes, bbox)
                        proba_sem_dor = float(model.predict(img_tensor)[0][0])
                        proba_dor = 1 - proba_sem_dor

                        score_history.append(proba_dor)
                        smoothed = sum(score_history) / len(score_history)

                        pred_class = "dor" if smoothed >= THRESHOLD else "sem_dor"
                        risk_score = smoothed if pred_class == "dor" else 1 - smoothed
                        risk_level = get_risk_level(risk_score)

                        animals.append({
                            "trackId": 1,
                            "species": species,
                            "bbox": bbox,           # [x, y, w, h] em pixels (relativo à imagem original)
                            "faceScore": risk_score,
                            "riskScore": risk_score,
                            "riskLevel": risk_level,
                            "confidence": 0.85      # valor fictício, pode ser extraído do modelo se disponível
                        })
                    except Exception as e:
                        print(f"Erro ao classificar detecção: {e}")
                        continue

        response = {
            "animals": animals,
            "processedAt": datetime.now(timezone.utc).isoformat()
        }
        print(f"✅ Análise concluída: {len(animals)} animal(is) detectado(s)")
        return response

    except HTTPException:
        raise
    except Exception as e:
        print("❌ Erro interno:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)