# ==================================================
# preprocessing.py
# 4-Class Lung Disease Detection
# ==================================================
import os
import shutil
import random
import tensorflow as tf
import matplotlib.pyplot as plt

# =========================
# 1 - Paths
# =========================
RAW_DIR = r"C:/Users/jayas/Downloads/archive/COVID-19_Radiography_Dataset"

# FIXED: OneDrive path
PROCESSED_DIR = r"C:/Users/jayas/OneDrive/Desktop/AI Based Medical Image Triage System/data/processed"

# FIXED: Kaggle folder name → Code class name mapping
KAGGLE_FOLDERS = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
CLASSES        = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]

SPLITS = ["train", "val", "test"]
RATIOS = [0.7, 0.15, 0.15]  # train, val, test

# =========================
# 2 - Create processed folders
# =========================
for split in SPLITS:
    for cls in CLASSES:
        os.makedirs(os.path.join(PROCESSED_DIR, split, cls), exist_ok=True)

print("✅ Folders created!")

# =========================
# 3 - Split and copy images
# =========================
for kaggle_name, class_name in zip(KAGGLE_FOLDERS, CLASSES):
    # Kaggle folder (space ho sakta hai)
    src = os.path.join(RAW_DIR, kaggle_name, "images")
    
    if not os.path.exists(src):
        print(f"❌ Folder nahi mila: {src}")
        continue
    
    images = [f for f in os.listdir(src) 
              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(images)

    n_total = len(images)
    n_train = int(RATIOS[0] * n_total)
    n_val   = int(RATIOS[1] * n_total)

    train_imgs = images[:n_train]
    val_imgs   = images[n_train:n_train + n_val]
    test_imgs  = images[n_train + n_val:]

    print(f"\n  {class_name}: {n_total} images")
    print(f"    Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test: {len(test_imgs)}")

    for img_list, split in zip([train_imgs, val_imgs, test_imgs], SPLITS):
        dst_dir = os.path.join(PROCESSED_DIR, split, class_name)
        for img_file in img_list:
            shutil.copy(
                os.path.join(src, img_file), 
                os.path.join(dst_dir, img_file)
            )

print("\n✅ Dataset split into train/val/test completed!")

# =========================
# 4 - TensorFlow Dataset Creation
# =========================
IMG_SIZE   = (224, 224)
BATCH_SIZE = 8

train_dataset = tf.keras.utils.image_dataset_from_directory(
    os.path.join(PROCESSED_DIR, "train"),
    image_size  = IMG_SIZE,
    batch_size  = BATCH_SIZE,
    label_mode  = 'categorical',
    class_names = CLASSES,
    shuffle     = True
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    os.path.join(PROCESSED_DIR, "val"),
    image_size  = IMG_SIZE,
    batch_size  = BATCH_SIZE,
    label_mode  = 'categorical',
    class_names = CLASSES,
    shuffle     = False
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    os.path.join(PROCESSED_DIR, "test"),
    image_size  = IMG_SIZE,
    batch_size  = BATCH_SIZE,
    label_mode  = 'categorical',
    class_names = CLASSES,
    shuffle     = False
)

# =========================
# 5 -  Data augmentation 
# =========================
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
])

train_dataset = train_dataset.map(
    lambda x, y: (data_augmentation(x, training=True), y)
)

# =========================
# 6 - Normalize images
# =========================
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset   = val_dataset.map(lambda x, y: (normalization_layer(x), y))
test_dataset  = test_dataset.map(lambda x, y: (normalization_layer(x), y))

# =========================
# 7 - Prefetch for performance
# =========================
AUTOTUNE      = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset   = val_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset  = test_dataset.prefetch(buffer_size=AUTOTUNE)

# =========================
# 8 - Preview some images
# =========================
for images_batch, labels_batch in train_dataset.take(1):
    plt.figure(figsize=(12, 6))
    for i in range(min(8, len(images_batch))):
        plt.subplot(2, 4, i + 1)
        plt.imshow(images_batch[i].numpy())
        class_idx = tf.argmax(labels_batch[i]).numpy()
        plt.title(CLASSES[class_idx], fontsize=8)
        plt.axis('off')
    plt.suptitle("Sample Training Images — 4 Classes")
    plt.tight_layout()
    plt.show()

print("✅ Preprocessing complete: datasets ready with augmentation + normalization")