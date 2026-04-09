import cv2
import requests
import time
from datetime import datetime

API_URL = "http://localhost:5214/frames"

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao acessar a câmera")
    exit()

print("enviando frames")

FPS = 5
INTERVAL = 1 / FPS

try:
    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            print("erro capturar frame")
            break

        frame = cv2.resize(frame, (640, 480))

        success, buffer = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        )

        if not success:
            print("erro converter frame")
            continue

        files = {
            "Frame": ("frame.jpg", buffer.tobytes(), "image/jpeg")   # ← "Frame" com F maiúsculo
        }

        data = {
            "timestamp": datetime.utcnow().isoformat()
        }

        try:
            response = requests.post(API_URL, files=files, data=data, timeout=5)
        except Exception as e:
            print(f"erro ao enviar frame: {e}")

        elapsed = time.time() - start_time
        sleep_time = max(0, INTERVAL - elapsed)
        time.sleep(sleep_time)

except KeyboardInterrupt:
    print("interrompido pelo usuário")

finally:
    cap.release()
    cv2.destroyAllWindows()
