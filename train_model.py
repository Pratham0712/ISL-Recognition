import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os

# -------- YOUR DATASET PATH --------
DATASET_PATH = r"C:\Users\prath\OneDrive\Desktop\Sign language detection\Data"

# -------- IMAGE + TRAIN SETTINGS -----
img_size = 224
batch_size = 16
epochs = 30

# -------- DATA GENERATOR --------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_gen = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

num_classes = len(train_gen.class_indices)
print("Class Mapping:", train_gen.class_indices)

# -------- BUILD MOBILENETV2 MODEL ----------
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(img_size, img_size, 3)
)

base_model.trainable = False   # Freeze MobileNet

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------- TRAIN THE MODEL -------------------
history = model.fit(
    train_gen,
    epochs=epochs,
    validation_data=val_gen
)

# -------- PLOT Accuracy vs Epoch -------------
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Epoch')
plt.legend()
plt.show()

# -------- PLOT Loss vs Epoch -----------------
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.legend()
plt.show()

# -------- SAVE MODEL -------------------------
model.save("keras_model.h5")
print("Model saved as keras_model.h5")

# -------- SAVE LABELS.TXT ---------------------
labels = list(train_gen.class_indices.keys())
with open("labels.txt", "w") as f:
    for i, label in enumerate(labels):
        f.write(f"{i} {label}\n")

print("labels.txt saved!")
