import os
import shutil
import yaml
from ultralytics import YOLO

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DATA  = os.path.join(PROJECT_ROOT, "data", "processed")
YOLO_DATA_DIR   = os.path.join(PROJECT_ROOT, "data", "yolo_dataset")
YAML_PATH       = os.path.join(PROJECT_ROOT, "data", "yolo_dataset.yaml")

FINAL_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "detection")
os.makedirs(FINAL_MODEL_DIR, exist_ok=True)

CLASSES = {
    "COVID": 0,
    "Lung_Opacity": 1,
    "Normal": 2,
    "Viral Pneumonia": 3
}

# ============================================================
# STEP 1: Prepare YOLO Format Data
# ============================================================
def prepare_yolo_data():

    if os.path.exists(YOLO_DATA_DIR):
        shutil.rmtree(YOLO_DATA_DIR)

    for split in ["train", "val", "test"]:
        for class_name, class_id in CLASSES.items():

            src = os.path.join(PROCESSED_DATA, split, class_name)
            if not os.path.exists(src):
                continue

            img_dst = os.path.join(YOLO_DATA_DIR, "images", split)
            lbl_dst = os.path.join(YOLO_DATA_DIR, "labels", split)

            os.makedirs(img_dst, exist_ok=True)
            os.makedirs(lbl_dst, exist_ok=True)

            for img in os.listdir(src):
                if img.lower().endswith((".jpg", ".jpeg", ".png")):

                    # Copy image
                    shutil.copy(
                        os.path.join(src, img),
                        os.path.join(img_dst, img)
                    )

                    # Create full-image bounding box
                    with open(os.path.join(lbl_dst, os.path.splitext(img)[0] + ".txt"), "w") as f:
                        f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

    print("✅ Data Prepared Successfully.")


# ============================================================
# STEP 2: Create YAML File
# ============================================================
def create_yaml():

    yaml_data = {
        "path": YOLO_DATA_DIR,
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(CLASSES),
        "names": list(CLASSES.keys())
    }

    with open(YAML_PATH, "w") as f:
        yaml.dump(yaml_data, f)

    print("✅ YAML Config Ready.")


# ============================================================
# STEP 3: Optimized Training (RTX 2050 Safe)
# ============================================================
def train_yolo():

    print("\n🚀 Training Start Ho Rahi Hai...\n")

    model = YOLO("yolov8n.pt")

    results = model.train(
        data=YAML_PATH,
        epochs=30,
        imgsz=640,        # ✅ Safe size
        batch=8,          # ✅ 4GB GPU optimized
        device=0,         # ✅ GPU
        workers=0,        # ✅ Windows stable
        mosaic=0.0,       # ✅ Disable memory-heavy mosaic
        amp=True,         # ✅ Mixed precision (faster + less memory)
        project=os.path.join(PROJECT_ROOT, "runs", "detect"),
        name="train_medical",
        exist_ok=True
    )

    # Move best model automatically
    temp_best = os.path.join(
        PROJECT_ROOT,
        "runs",
        "detect",
        "train_medical",
        "weights",
        "best.pt"
    )

    final_best = os.path.join(FINAL_MODEL_DIR, "best.pt")

    if os.path.exists(temp_best):
        shutil.copy(temp_best, final_best)
        print(f"\n✨ SUCCESS! Model saved at:\n{final_best}")
    else:
        print("\n⚠️ Warning: Training complete but best.pt not found!")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    prepare_yolo_data()
    create_yaml()
    train_yolo()
