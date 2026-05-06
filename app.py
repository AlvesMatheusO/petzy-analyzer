import os
import json
import pickle
import traceback
import asyncio

from datetime import datetime, timezone
from collections import deque

import cv2
import numpy as np

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
)

from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import (
    FileResponse,
    StreamingResponse,
)

from tensorflow.keras.models import load_model

from tensorflow.keras.applications.efficientnet_v2 import (
    preprocess_input as efficient_preprocess,
)

from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input as mobilenet_preprocess,
)

from ultralytics import YOLO

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)

PAIN_MODEL_DOGS = os.path.join(
    MODELS_DIR,
    "petzy_pain_dogs.keras"
)

PAIN_MODEL_CATS = os.path.join(
    MODELS_DIR,
    "petzy_pain_cats.keras"
)

INFO_DOGS = os.path.join(
    MODELS_DIR,
    "model_info_dogs.pkl"
)

INFO_CATS = os.path.join(
    MODELS_DIR,
    "model_info_cats.pkl"
)

SPECIES_MODEL = os.path.join(
    MODELS_DIR,
    "species_classifier_mobilenetv2.keras"
)

SPECIES_INDICES = os.path.join(
    MODELS_DIR,
    "species_class_indices.json"
)

YOLO_PATH = os.path.join(
    BASE_DIR,
    "yolov8n.pt"
)

LATEST_FRAME_PATH = os.path.join(
    BASE_DIR,
    "latest-frame.jpg"
)

HISTORY_LEN = 5
THRESHOLD = 0.7

# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Petzy AI"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# GLOBAIS
# ============================================================

yolo = None

pain_models = {}
model_infos = {}

species_model = None
species_idx_map = None

score_history = deque(
    maxlen=HISTORY_LEN
)

latest_frame_bytes = None

latest_analysis = {
    "animals": [],
    "processedAt": None
}

# ============================================================
# LOAD MODELS
# ============================================================

def load_artifacts():

    global yolo
    global species_model
    global species_idx_map

    print(
        "\n🔍 Carregando modelos...\n"
    )

    yolo = YOLO(YOLO_PATH)

    print("✅ YOLO carregado")

    models_data = [
        (
            "cao",
            PAIN_MODEL_DOGS,
            INFO_DOGS
        ),
        (
            "gato",
            PAIN_MODEL_CATS,
            INFO_CATS
        )
    ]

    for species, model_path, info_path in models_data:

        if not os.path.exists(model_path):

            raise FileNotFoundError(
                f"Modelo não encontrado: {model_path}"
            )

        if not os.path.exists(info_path):

            raise FileNotFoundError(
                f"Info não encontrada: {info_path}"
            )

        pain_models[species] = load_model(
            model_path
        )

        with open(info_path, "rb") as f:

            model_infos[species] = pickle.load(f)

        print(
            f"✅ Modelo {species} carregado"
        )

    if (
        os.path.exists(SPECIES_MODEL)
        and
        os.path.exists(SPECIES_INDICES)
    ):

        species_model = load_model(
            SPECIES_MODEL
        )

        with open(SPECIES_INDICES, "r") as f:

            species_idx_map = json.load(f)

        print(
            "✅ Classificador de espécie carregado"
        )

    print(
        "\n🎉 Todos os modelos carregados!"
    )

# ============================================================
# HELPERS
# ============================================================

def get_species_from_yolo(cls: int):

    if cls == 15:
        return "gato"

    return "cao"


def classify_species(
    image_bytes,
    bbox
):

    if species_model is None:
        raise RuntimeError()

    nparr = np.frombuffer(
        image_bytes,
        np.uint8
    )

    img = cv2.imdecode(
        nparr,
        cv2.IMREAD_COLOR
    )

    x, y, w, h = bbox

    crop = img[
        y:y + h,
        x:x + w
    ]

    crop_rgb = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB
    )

    resized = cv2.resize(
        crop_rgb,
        (224, 224)
    )

    preprocessed = mobilenet_preprocess(
        resized.astype(np.float32)
    )

    batch = np.expand_dims(
        preprocessed,
        axis=0
    )

    preds = species_model.predict(
        batch,
        verbose=0
    )[0]

    class_idx = int(
        np.argmax(preds)
    )

    for species_name, idx in species_idx_map.items():

        if idx == class_idx:

            if species_name == "dogs":
                return "cao"

            return "gato"

    return "desconhecido"


def preprocess_for_pain(
    image_bytes,
    bbox,
    input_size
):

    nparr = np.frombuffer(
        image_bytes,
        np.uint8
    )

    img = cv2.imdecode(
        nparr,
        cv2.IMREAD_COLOR
    )

    x, y, w, h = bbox

    crop = img[
        y:y + h,
        x:x + w
    ]

    crop_rgb = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB
    )

    resized = cv2.resize(
        crop_rgb,
        (input_size, input_size)
    )

    preprocessed = efficient_preprocess(
        resized.astype(np.float32)
    )

    return np.expand_dims(
        preprocessed,
        axis=0
    )


def get_risk_level(score):

    if score < 0.35:
        return "normal"

    if score < 0.65:
        return "atencao"

    return "critico"

# ============================================================
# MJPEG STREAM
# ============================================================

async def mjpeg_generator():

    global latest_frame_bytes

    while True:

        if latest_frame_bytes is not None:

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + latest_frame_bytes +
                b"\r\n"
            )

        await asyncio.sleep(0.03)

# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
async def startup_event():

    try:

        load_artifacts()

        print(
            "✅ Startup concluído"
        )

    except Exception:

        traceback.print_exc()

# ============================================================
# ROUTES
# ============================================================

@app.get("/")
async def health():

    return {
        "status": "online",
        "models_loaded": (
            yolo is not None
            and
            len(pain_models) > 0
        )
    }


@app.get("/video-feed")
async def video_feed():

    return StreamingResponse(
        mjpeg_generator(),
        media_type=(
            "multipart/x-mixed-replace; boundary=frame"
        )
    )


@app.get("/latest-analysis")
async def get_latest_analysis():

    return latest_analysis


@app.post("/analyze-frame")
async def analyze_frame(
    file: UploadFile = File(...)
):

    global latest_analysis
    global latest_frame_bytes

    if yolo is None:

        raise HTTPException(
            status_code=503,
            detail="Modelos não carregados"
        )

    try:

        image_bytes = await file.read()

        if not image_bytes:

            raise HTTPException(
                status_code=400,
                detail="Imagem vazia"
            )

        # ============================================
        # SALVA FRAME
        # ============================================

        latest_frame_bytes = image_bytes

        with open(
            LATEST_FRAME_PATH,
            "wb"
        ) as f:

            f.write(image_bytes)

        # ============================================
        # OPENCV
        # ============================================

        nparr = np.frombuffer(
            image_bytes,
            np.uint8
        )

        img_cv = cv2.imdecode(
            nparr,
            cv2.IMREAD_COLOR
        )

        # ============================================
        # YOLO
        # ============================================

        results = yolo(img_cv)

        detections = results[0].boxes

        animals = []

        for box in detections if detections else []:

            cls = int(box.cls[0])

            if cls not in (15, 16):
                continue

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            bbox = [
                x1,
                y1,
                x2 - x1,
                y2 - y1
            ]

            conf_yolo = float(
                box.conf[0]
            )

            yolo_species = get_species_from_yolo(
                cls
            )

            if species_model is not None:

                try:

                    species = classify_species(
                        image_bytes,
                        bbox
                    )

                except Exception:

                    species = yolo_species

            else:

                species = yolo_species

            pain_model = pain_models.get(
                species
            )

            if pain_model is None:
                continue

            info = model_infos[species]

            input_size = info.get(
                "input_size",
                384
            )

            img_tensor = preprocess_for_pain(
                image_bytes,
                bbox,
                input_size
            )

            proba_sem_dor = float(
                pain_model.predict(
                    img_tensor,
                    verbose=0
                )[0][0]
            )

            proba_dor = 1 - proba_sem_dor

            score_history.append(
                proba_dor
            )

            smoothed = (
                sum(score_history)
                /
                len(score_history)
            )

            pred_class = (
                "dor"
                if smoothed >= THRESHOLD
                else "sem_dor"
            )

            risk_score = (
                smoothed
                if pred_class == "dor"
                else 1 - smoothed
            )

            risk_level = get_risk_level(
                risk_score
            )

            animals.append({
                "trackId": 1,
                "species": species,
                "bbox": bbox,
                "faceScore": risk_score,
                "riskScore": risk_score,
                "riskLevel": risk_level,
                "confidence": conf_yolo
            })

        latest_analysis = {
            "animals": animals,
            "processedAt": datetime.now(
                timezone.utc
            ).isoformat()
        }

        return latest_analysis

    except HTTPException:
        raise

    except Exception as e:

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )