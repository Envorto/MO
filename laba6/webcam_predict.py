# webcam_predict.py
import cv2
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = 'models/gesture_rps.h5'
LABELS_PATH = 'models/labels.json'
IMG_SIZE = (224, 224)

model = load_model(MODEL_PATH)
with open(LABELS_PATH, 'r', encoding='utf-8') as f:
    class_indices = json.load(f)
inv_map = {v: k for k, v in class_indices.items()}

cap = cv2.VideoCapture(0)  # 0 — первая камера
if not cap.isOpened():
    print("Cannot open camera"); exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Преобразование для модели
    img = cv2.resize(frame, IMG_SIZE)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = np.expand_dims(img_rgb.astype('float32'), axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    idx = int(np.argmax(preds, axis=1)[0])
    prob = float(np.max(preds))
    label = inv_map[idx]

    # Отрисовка результата
    cv2.putText(frame, f'{label} {prob:.2f}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow('RPS - press q to quit', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
