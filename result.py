import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import os

# ---------------- SETTINGS ----------------
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier(
    "C:/Users/prath/OneDrive/Desktop/MODEL/keras_model.h5",
    "C:/Users/prath/OneDrive/Desktop/MODEL/labels.txt"
)

offset = 20
imgSize = 300

# Load labels
labels = []
with open("C:/Users/prath/OneDrive/Desktop/MODEL/labels.txt", "r") as f:
    for line in f:
        name = line.strip()
        if name:
            parts = name.split(" ", 1)
            if len(parts) == 2 and parts[0].isdigit():
                labels.append(parts[1])
            else:
                labels.append(name)

# ----------- DATA STORAGE -----------
y_true = []
y_pred = []
y_prob = []
frame_numbers = []

sample_id = 0

print("\n------------------------------")
print("DATA COLLECTION MODE")
print("Press 1-7 to set TRUE LABEL")
print("Press S to SAVE this prediction")
print("Press Q to quit and export data")
print("------------------------------\n")

current_true_label = None

while True:
    success, img = cap.read()
    if not success:
        continue

    imgOutput = img.copy()

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Safe crop
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)

        imgCrop = img[y1:y2, x1:x2]
        if imgCrop.size == 0:
            cv2.imshow("Image", imgOutput)
            cv2.waitKey(1)
            continue

        aspectRatio = imgCrop.shape[0] / imgCrop.shape[1]

        if aspectRatio > 1:
            k = imgSize / imgCrop.shape[0]
            wCal = math.ceil(imgCrop.shape[1] * k)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = (imgSize - wCal) // 2
            imgWhite[:, wGap:wGap + wCal] = imgResize
        else:
            k = imgSize / imgCrop.shape[1]
            hCal = math.ceil(imgCrop.shape[0] * k)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = (imgSize - hCal) // 2
            imgWhite[hGap:hGap + hCal, :] = imgResize

        # Prediction
        prediction, index = classifier.getPrediction(imgWhite, draw=False)
        predicted_label = labels[index]

        # Display predicted label
        cv2.putText(imgOutput, f"PRED: {predicted_label}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(imgOutput, f"TRUE: {current_true_label}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

        cv2.imshow("ImageWhite", imgWhite)

    # Show main image
    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1) & 0xFF

    # ------------------ TRUE LABEL SETTING ------------------
    if key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7')]:
        current_true_label = int(chr(key)) - 1  # convert 1..7 → 0..6
        print(f"True label set to: {labels[current_true_label]}")

    # ------------------ SAVE SAMPLE -------------------------
    if key == ord('s') and current_true_label is not None:
        y_true.append(current_true_label)
        y_pred.append(index)
        y_prob.append(prediction)
        frame_numbers.append(sample_id)
        sample_id += 1
        print(f"Saved sample {sample_id}: TRUE={labels[current_true_label]} PRED={predicted_label}")

    # ------------------ EXIT ------------------
    if key == ord('q'):
        print("Exporting data...")

        np.save("y_true.npy", np.array(y_true))
        np.save("y_pred.npy", np.array(y_pred))
        np.save("y_prob.npy", np.array(y_prob))
        np.save("frames.npy", np.array(frame_numbers))

        print("Saved: y_true.npy, y_pred.npy, y_prob.npy, frames.npy")
        break

cap.release()
cv2.destroyAllWindows()
