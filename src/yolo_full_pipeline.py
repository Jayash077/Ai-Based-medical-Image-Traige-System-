# =========================
# yolo_detection.py (4 CLASS VERSION)
# =========================

import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# =========================
# PATHS
# =========================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")

TRANSFER_MODEL_PATH = os.path.join(
    MODELS_DIR, "classification", "transfer_model_resnet50.h5"
)

YOLO_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "models", "detection", "best.pt"
)

# =========================
# CLASS NAMES (4 CLASS)
# =========================

CLASS_NAMES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]

# =========================
# RANDOM TEST IMAGE (4 CLASS)
# =========================

test_class = random.choice(CLASS_NAMES)

img_name = random.choice(
    os.listdir(
        os.path.join(PROJECT_ROOT, "data", "processed", "test", test_class)
    )
)

TEST_IMAGE = os.path.join(
    PROJECT_ROOT, "data", "processed", "test", test_class, img_name
)

print("Using image:", TEST_IMAGE)

# =========================
# LOAD MODELS
# =========================

if not os.path.exists(TRANSFER_MODEL_PATH):
    raise FileNotFoundError(f"CNN model nahi mila:\n{TRANSFER_MODEL_PATH}")

print("Loading CNN model...")
transfer_model = load_model(TRANSFER_MODEL_PATH)

if not os.path.exists(YOLO_MODEL_PATH):
    raise FileNotFoundError(f"YOLO model nahi mila:\n{YOLO_MODEL_PATH}")

print("Loading YOLO model...")
yolo_model = YOLO(YOLO_MODEL_PATH)

# =========================
# PRIORITY LOGIC (MULTI-CLASS)
# =========================

def get_priority_label(class_name, conf):

    if class_name == "COVID" and conf >= 0.6:
        return "Critical"

    elif class_name in ["Lung_Opacity", "Viral Pneumonia"] and conf >= 0.5:
        return "Moderate"

    else:
        return "Normal"

# =========================
# CNN PREDICTION (4 CLASS)
# =========================

def cnn_predict(img_path, model=transfer_model):

    img  = image.load_img(img_path, target_size=(224, 224))
    x    = image.img_to_array(img) / 255.0
    x    = np.expand_dims(x, axis=0)

    preds = model.predict(x, verbose=0)[0]

    class_index = np.argmax(preds)
    confidence  = preds[class_index]

    class_name = CLASS_NAMES[class_index]
    priority   = get_priority_label(class_name, confidence)

    return class_name, priority, confidence

# =========================
# YOLO DETECTION (4 CLASS)
# =========================

def yolo_predict(img_path, model=yolo_model):

    results = model.predict(source=img_path, imgsz=640, conf=0.4)
    img     = cv2.imread(img_path)

    for result in results:

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        cls   = result.boxes.cls.cpu().numpy()

        for (x1, y1, x2, y2), c, cl in zip(boxes, confs, cls):

            class_name = CLASS_NAMES[int(cl)]
            priority   = get_priority_label(class_name, c)

            label_text = f"{class_name} | {priority} | {c:.2f}"

            cv2.rectangle(
                img,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )

            cv2.putText(
                img,
                label_text,
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    return img

# =========================
# RUN PREDICTION
# =========================

cnn_class, cnn_priority, cnn_conf = cnn_predict(TEST_IMAGE)

print(f"CNN Prediction : {cnn_class}")
print(f"Priority       : {cnn_priority}")
print(f"Confidence     : {cnn_conf:.2f}")

yolo_img     = yolo_predict(TEST_IMAGE)
yolo_img_rgb = cv2.cvtColor(yolo_img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 10))
plt.imshow(yolo_img_rgb)
plt.axis("off")
plt.title(f"YOLO + CNN Triage → {cnn_class} ({cnn_priority})")
plt.show()
