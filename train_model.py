import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

# ==========================
# CONFIG
# ==========================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

TRAIN_DIR = "dataset_split/train"
VAL_DIR = "dataset_split/val"

# ==========================
# LOAD DATASET
# ==========================
train_ds = image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

val_ds = image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("\nKelas:")
for i, c in enumerate(class_names):
    print(i, c)

# ==========================
# CLASS WEIGHT
# ==========================
labels = []

for _, y in train_ds:
    labels.extend(y.numpy())

labels = np.array(labels)

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)

class_weights = dict(enumerate(weights))

print("\nClass Weights:")
print(class_weights)

# ==========================
# OPTIMASI DATASET
# ==========================
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ==========================
# DATA AUGMENTATION
# ==========================
augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# ==========================
# MODEL CNN
# ==========================
model = models.Sequential([

    layers.Input(shape=(224, 224, 3)),

    augmentation,

    layers.Rescaling(1./255),

    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(256, 3, activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4),

    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==========================
# TRAINING
# ==========================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights
)

# ==========================
# SAVE MODEL
# ==========================
os.makedirs("models", exist_ok=True)

model.save("models/batik_defect_model.keras")

print("\nModel berhasil disimpan:")
print("models/batik_defect_model.keras")