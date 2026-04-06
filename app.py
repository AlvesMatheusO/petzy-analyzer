# app.py - Serviço de análise de dor para cães/gatos
import os
import json
import pickle
import traceback
from datetime import datetime, timezone
from typing import List, Dict, Any

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# ------------------------------------------------------------
# Configurações
# ------------------------------------------------------------
MODEL_PATH = "petzy_model.keras"          # Nome do arquivo do modelo
CLASS_INDICES_PATH = "class_indices.json" # Mapeamento classes -> índices
MODEL_INFO_PATH = "model_info.pkl"        # Metadados (input_size, etc.)

# ------------------------------------------------------------
# Inicialização do FastAPI e carregamento do modelo
# ------------------------------------------------------------
app = FastAPI(title="Petzy Pain Analyzer")

# Variáveis globais
model = None
input_size = 224
idx_to_class = {}
class_indices = {}

def load_artifacts():
    """Carrega modelo, mapeamento de classes e metadados."""
    global model, input_size, idx_to_class, class_indices

    # Verifica se os arquivos existem
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modelo não encontrado: {MODEL_PATH}")
    if not os.path.exists(CLASS_INDICES_PATH):
        raise FileNotFoundError(f"Arquivo de classes não encontrado: {CLASS_INDICES_PATH}")
    if not os.path.exists(MODEL_INFO_PATH):
        raise FileNotFoundError(f"Arquivo de metadados não encontrado: {MODEL_INFO_PATH}")

    # Carrega o modelo Keras
    try:
        model = load_model(MODEL_PATH)
        print(f"✅ Modelo carregado com sucesso de {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Erro ao carregar o modelo: {e}")
        raise

    # Carrega mapeamento de classes
    try:
        with open(CLASS_INDICES_PATH, "r") as f:
            class_indices = json.load(f)  # ex: {"dor": 0, "sem_dor": 1}
        idx_to_class = {v: k for k, v in class_indices.items()}
        print(f"✅ Classes carregadas: {class_indices}")
    except Exception as e:
        print(f"❌ Erro ao carregar class_indices.json: {e}")
        raise

    # Carrega metadados
    try:
        with open(MODEL_INFO_PATH, "rb") as f:
            info = pickle.load(f)
        input_size = info.get("input_size", 224)
        print(f"✅ Metadados carregados: input_size={input_size}")
    except Exception as e:
        print(f"⚠️ Aviso: não foi possível carregar metadados, usando input_size=224. Erro: {e}")
        input_size = 224

# Tenta carregar os artefatos na inicialização
try:
    load_artifacts()
except Exception as e:
    print(f"FALHA CRÍTICA: {e}")
    print("O serviço não funcionará corretamente sem os arquivos do modelo.")
    # Se preferir, pode deixar o serviço rodando com mock:
    # model = None

# ------------------------------------------------------------
# Funções auxiliares
# ------------------------------------------------------------
def preprocess_image(image_bytes: bytes):
    """
    Recebe bytes da imagem, retorna tensor pré-processado e dimensões originais.
    """
    # Decodifica a imagem com OpenCV
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Imagem inválida ou corrompida")

    h, w = img.shape[:2]  # altura e largura originais

    # Converte BGR (OpenCV) para RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Redimensiona para o tamanho esperado pelo modelo
    img_resized = cv2.resize(img_rgb, (input_size, input_size))

    # Aplica pré-processamento específico do EfficientNetV2
    img_preprocessed = preprocess_input(img_resized.astype(np.float32))

    # Adiciona dimensão de batch (1, H, W, C)
    img_batch = np.expand_dims(img_preprocessed, axis=0)
    return img_batch, (h, w)

def get_risk_level(score: float) -> str:
    """Converte score numérico (0 a 1) em nível textual."""
    if score < 0.3:
        return "normal"
    elif score < 0.7:
        return "atencao"
    else:
        return "critico"

# ------------------------------------------------------------
# Endpoint de análise
# ------------------------------------------------------------
@app.post("/analyze-frame")
async def analyze_frame(file: UploadFile = File(...)):
    """
    Recebe uma imagem (multipart/form-data) e retorna análise de dor.
    Formato de resposta compatível com o backend C#.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo não carregado. Verifique os arquivos do modelo."
        )

    try:
        # 1. Ler a imagem
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Arquivo vazio")

        # 2. Pré-processar
        img_tensor, (h, w) = preprocess_image(image_bytes)

        # 3. Inferência
        # Saída do modelo: probabilidade da classe positiva (dor)
        proba = float(model.predict(img_tensor)[0][0])   # sigmoid output
        # Se o modelo tiver duas saídas softmax, use:
        # proba = float(model.predict(img_tensor)[0][class_indices.get("dor", 1)])

        # 4. Determinar classe e score de risco
        predicted_class = "dor" if proba >= 0.5 else "sem_dor"
        risk_score = proba if predicted_class == "dor" else 1 - proba
        risk_level = get_risk_level(risk_score)

        # 5. Montar resposta no formato esperado pelo backend
        response = {
            "animals": [
                {
                    "trackId": 1,                     # fixo, sem tracking
                    "species": "desconhecido",        # você pode melhorar depois
                    "bbox": [0, 0, w, h],            # imagem inteira
                    "faceScore": risk_score,          # mesmo score por enquanto
                    "riskScore": risk_score,
                    "riskLevel": risk_level
                }
            ],
            "processedAt": datetime.now(timezone.utc).isoformat()
        }

        # Log opcional para debug
        print(f"✅ Análise concluída: classe={predicted_class}, riskScore={risk_score:.3f}")

        return response

    except HTTPException:
        raise
    except Exception as e:
        # Log detalhado do erro no console do servidor
        print("❌ Erro durante o processamento:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

# ------------------------------------------------------------
# Ponto de entrada (se executar diretamente)
# ------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)