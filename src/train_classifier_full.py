# ==================================================
# train_classifier_full.py (CORRECTED FOR 4 CLASSES)
# ==================================================
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# =========================
# 0️⃣ Suppress TF info logs
# =========================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# =========================
# 1️⃣ Paths & Parameters
# =========================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
CLASSIFICATION_DIR = os.path.join(PROJECT_ROOT, "models", "classification")
os.makedirs(CLASSIFICATION_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS_BASELINE = 10
EPOCHS_TRANSFER = 5
CLASSES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]

# =========================
# 2️⃣ Load preprocessed datasets
# =========================
# Note: label_mode='categorical' creates one-hot labels [1,0,0,0]
train_dataset = tf.keras.utils.image_dataset_from_directory(
    os.path.join(PROCESSED_DIR, "train"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=True
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    os.path.join(PROCESSED_DIR, "val"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=False
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    os.path.join(PROCESSED_DIR, "test"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=False
)

# =========================
# 3️⃣ Data augmentation
# =========================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])

train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))

# =========================
# 4️⃣ Normalize images
# =========================
normalization_layer = layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset   = val_dataset.map(lambda x, y: (normalization_layer(x), y))
test_dataset  = test_dataset.map(lambda x, y: (normalization_layer(x), y))

# =========================
# 5️⃣ Prefetch for performance
# =========================
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset   = val_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset  = test_dataset.prefetch(buffer_size=AUTOTUNE)

# =========================
# 6️⃣ Preview some images (FIXED ERROR)
# =========================
for images, labels in train_dataset.take(1):
    plt.figure(figsize=(12, 6))
    for i in range(min(len(images), 8)): 
        plt.subplot(2, 4, i+1)
        # Display image
        plt.imshow(images[i].numpy())
        
        # FIX: One-hot vector se index nikal kar name match karna
        label_idx = np.argmax(labels[i].numpy())
        plt.title(f"Label: {CLASSES[label_idx]}")
        plt.axis('off')
    plt.show()

# =========================
# 7️⃣ Baseline CNN Model (FIXED FOR 4 CLASSES)
# =========================
baseline_model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax') # FIX: 4 classes + softmax
])

# FIX: Categorical Crossentropy for multiclass
baseline_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("\n✅ Training Baseline CNN Model...")
history_baseline = baseline_model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS_BASELINE)

# Plot baseline accuracy
plt.figure(figsize=(8,5))
plt.plot(history_baseline.history['accuracy'], label='Train Accuracy')
plt.plot(history_baseline.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Baseline CNN Accuracy')
plt.legend()
plt.show()

# Save baseline model
BASELINE_SAVE_PATH = os.path.join(CLASSIFICATION_DIR, "baseline_cnn.h5")
baseline_model.save(BASELINE_SAVE_PATH)
print(f"✅ Baseline CNN model saved at {BASELINE_SAVE_PATH}")

# =========================
# 8️⃣ Evaluate Baseline Model (FIXED)
# =========================
# Extract indices from one-hot labels
y_true = np.concatenate([np.argmax(y.numpy(), axis=1) for x, y in test_dataset], axis=0)
y_pred = np.concatenate([np.argmax(baseline_model.predict(x), axis=1) for x, y in test_dataset], axis=0)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
plt.title("Baseline CNN Confusion Matrix")
plt.show()

# =========================
# 9️⃣ Transfer Learning Model (ResNet50) (FIXED)
# =========================
resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
resnet_base.trainable = False

transfer_model = models.Sequential([
    resnet_base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax') # FIX: 4 classes + softmax
])

transfer_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("\n✅ Training Transfer Learning Model (ResNet50)...")
history_transfer = transfer_model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS_TRANSFER)

# Save Transfer Model
MODEL_SAVE_PATH = os.path.join(CLASSIFICATION_DIR, "transfer_model_resnet50.h5")
transfer_model.save(MODEL_SAVE_PATH)
print(f"✅ Transfer model saved at {MODEL_SAVE_PATH}")

# =========================
# 10️⃣ Prediction Function (FIXED FOR 4 CLASSES)
# =========================
def predict_image(img_path, model=transfer_model):
    img = image.load_img(img_path, target_size=(224,224))
    x = image.img_to_array(img)/255.0
    x = np.expand_dims(x, axis=0)
    
    preds = model.predict(x)[0]
    label_idx = np.argmax(preds)
    confidence = preds[label_idx]
    
    label_name = CLASSES[label_idx]
    
    print(f"Prediction: {label_name}, Confidence: {confidence:.2f}")

# =========================
# 11️⃣ Example Prediction
# =========================
test_class = random.choice(CLASSES)
test_class_path = os.path.join(PROCESSED_DIR, "test", test_class)

if os.path.exists(test_class_path) and len(os.listdir(test_class_path)) > 0:
    example_img_name = random.choice(os.listdir(test_class_path))
    example_img_path = os.path.join(test_class_path, example_img_name)
    print(f"\nPredicting a random test image from {test_class} class:")
    predict_image(example_img_path)
else:
    print("\nNo test images found for prediction.")