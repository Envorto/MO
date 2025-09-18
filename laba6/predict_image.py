# predict_image.py
import sys, os, json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = 'models/gesture_rps.h5'
LABELS_PATH = 'models/labels.json'
IMG_SIZE = (224, 224)

if len(sys.argv) < 2:
    print("Usage: python predict_image.py path/to/image.jpg")
    sys.exit(1)

img_path = sys.argv[1]
if not os.path.exists(img_path):
    print("File not found:", img_path); sys.exit(1)

model = load_model(MODEL_PATH)
with open(LABELS_PATH, 'r', encoding='utf-8') as f:
    class_indices = json.load(f)
inv_map = {v: k for k, v in class_indices.items()}

img = image.load_img(img_path, target_size=IMG_SIZE)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
idx = int(np.argmax(preds, axis=1)[0])
prob = float(np.max(preds))
label = inv_map[idx]
print(f'Predicted: {label} ({prob*100:.2f}%)')
