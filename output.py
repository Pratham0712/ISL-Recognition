# =========================
#  SIGN LANGUAGE ANALYSIS
#  Pratham – Ready-to-run
# =========================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2

# ---------- PATHS (edit if needed) ----------
Y_TRUE_PATH = "y_true.npy"
Y_PRED_PATH = "y_pred.npy"
Y_PROB_PATH = "y_prob.npy"
FRAMES_PATH = "frames.npy"    # not strictly needed, but good to have
HISTORY_PATH = "history.npy"  # optional, only if you saved training history

MODEL_PATH = "C:/Users/prath/OneDrive/Desktop/MODEL/keras_model.h5"

# Example image path for Grad-CAM (EDIT THIS to a real image)
GRADCAM_IMAGE_PATH = "C:/Users/prath/OneDrive/Desktop/Sign language detection/Data/Hello/Image_1.jpg" 
# make sure this exists, or Grad-CAM section will be skipped


# ---------- CLASS NAMES (indices 0..6) ----------
class_names = [
    "Hello",       # 0
    "I love you",  # 1
    "No",          # 2
    "Okay",        # 3
    "Please",      # 4
    "Thank you",   # 5
    "Yes"          # 6
]

# =========================================================
# 1. LOAD DATA
# =========================================================

y_true = np.load(Y_TRUE_PATH)
y_pred = np.load(Y_PRED_PATH)
y_prob = np.load(Y_PROB_PATH)

print("y_true shape:", y_true.shape)
print("y_pred shape:", y_pred.shape)
print("y_prob shape:", y_prob.shape)

n_classes = len(class_names)

# =========================================================
# 2. CLASS DISTRIBUTION BAR CHART
# =========================================================

counts = pd.Series(y_true).value_counts().sort_index()
counts.index = class_names  # map 0..6 to class names

plt.figure(figsize=(8, 5))
sns.barplot(x=counts.index, y=counts.values)
plt.xlabel("Class")
plt.ylabel("Number of samples")
plt.title("Class Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# =========================================================
# 3. CONFUSION MATRIX (raw + normalized)
# =========================================================

cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))

plt.figure(figsize=(7, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
plt.figure(figsize=(7, 6))
sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Normalized Confusion Matrix")
plt.tight_layout()
plt.show()


# =========================================================
# 4. CLASS-WISE PRECISION, RECALL, F1 (bar charts)
# =========================================================

precision, recall, f1, support = precision_recall_fscore_support(
    y_true, y_pred, labels=range(n_classes), zero_division=0
)

metrics_df = pd.DataFrame({
    "class": class_names,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "support": support
})

print("\nClass-wise metrics:")
print(metrics_df)

# Precision
plt.figure(figsize=(8, 5))
sns.barplot(data=metrics_df, x="class", y="precision")
plt.ylim(0, 1.05)
plt.title("Class-wise Precision")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Recall
plt.figure(figsize=(8, 5))
sns.barplot(data=metrics_df, x="class", y="recall")
plt.ylim(0, 1.05)
plt.title("Class-wise Recall")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# F1-score
plt.figure(figsize=(8, 5))
sns.barplot(data=metrics_df, x="class", y="f1")
plt.ylim(0, 1.05)
plt.title("Class-wise F1-score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# =========================================================
# 5. ROC CURVE (One-vs-Rest)
# =========================================================

# Binarize true labels
y_true_bin = label_binarize(y_true, classes=range(n_classes))  # shape: (n_samples, n_classes)

fpr = {}
tpr = {}
roc_auc = {}

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8, 6))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multi-class ROC Curve (One-vs-Rest)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# =========================================================
# 6. TRAINING CURVES (Loss & Accuracy vs Epoch) – OPTIONAL
# =========================================================

if os.path.exists(HISTORY_PATH):
    history_dict = np.load(HISTORY_PATH, allow_pickle=True).item()
    print("\nHistory keys:", history_dict.keys())

    epochs = range(1, len(history_dict["loss"]) + 1)

    # Try to infer accuracy key
    if "accuracy" in history_dict:
        acc_key = "accuracy"
    elif "categorical_accuracy" in history_dict:
        acc_key = "categorical_accuracy"
    else:
        acc_key = None

    plt.figure(figsize=(8, 5))

    # Loss
    plt.plot(epochs, history_dict["loss"], label="Train Loss")
    if "val_loss" in history_dict:
        plt.plot(epochs, history_dict["val_loss"], linestyle="--", label="Val Loss")

    # Accuracy
    if acc_key is not None:
        plt.plot(epochs, history_dict[acc_key], label="Train Accuracy")
        if f"val_{acc_key}" in history_dict:
            plt.plot(epochs, history_dict[f"val_{acc_key}"], linestyle="--", label="Val Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Loss & Accuracy vs Epochs")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    print("\n⚠️ HISTORY_FILE not found – skipping training curves.")
    print("If you want this, save Keras History like:")
    print("  np.save('history.npy', history.history, allow_pickle=True)")


# =========================================================
# 7. GRAD-CAM HEATMAP VISUALISATION
# =========================================================

def get_last_conv_layer(model):
    """
    Find the last convolutional layer in the model.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None

def get_img_array(img_path, size):
    img = image.load_img(img_path, target_size=size)
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = arr / 255.0
    return arr

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [last_conv_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_gradcam_on_image(img_path, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read image: {img_path}")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_resized = np.uint8(255 * heatmap_resized)

    heatmap_color = cv2.applyColorMap(heatmap_resized, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    superimposed = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)

    plt.figure(figsize=(5, 5))
    plt.imshow(superimposed)
    plt.axis("off")
    plt.title("Grad-CAM")
    plt.tight_layout()
    plt.show()

    return superimposed

# ---- Run Grad-CAM for one example image ----
if os.path.exists(MODEL_PATH) and os.path.exists(GRADCAM_IMAGE_PATH):
    print("\nLoading model for Grad-CAM...")
    model = tf.keras.models.load_model(MODEL_PATH)

    input_size = model.input_shape[1:3]  # e.g., (224, 224)
    last_conv_layer_name = get_last_conv_layer(model)

    if last_conv_layer_name is None:
        print("⚠️ No Conv2D layer found in model – cannot compute Grad-CAM.")
    else:
        print("Last conv layer:", last_conv_layer_name)
        img_array = get_img_array(GRADCAM_IMAGE_PATH, size=input_size)

        # Predicted class for this image
        preds = model.predict(img_array)
        pred_index = np.argmax(preds[0])
        print("Predicted class index:", pred_index, "->", class_names[pred_index])

        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=pred_index)
        _ = overlay_gradcam_on_image(GRADCAM_IMAGE_PATH, heatmap)
else:
    print("\n⚠️ MODEL_PATH or GRADCAM_IMAGE_PATH not valid – skipping Grad-CAM.")
    print("Edit MODEL_PATH and GRADCAM_IMAGE_PATH at the top and rerun this cell.")
