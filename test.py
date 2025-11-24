import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

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

while True:
    success, img = cap.read()
    imgOutput = img.copy()

    hands, img = detector.findHands(img, draw=True)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), dtype=np.uint8) * 255

        # Safe crop region
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)

        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size == 0:
            cv2.imshow("Image", imgOutput)
            cv2.waitKey(1)
            continue

        # CORRECT aspect ratio
        aspectRatio = imgCrop.shape[0] / imgCrop.shape[1]

        if aspectRatio > 1:
            # Tall image
            k = imgSize / imgCrop.shape[0]
            wCal = math.ceil(imgCrop.shape[1] * k)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = (imgSize - wCal) // 2
            imgWhite[:, wGap:wGap + wCal] = imgResize
        else:
            # Wide image
            k = imgSize / imgCrop.shape[1]
            hCal = math.ceil(imgCrop.shape[0] * k)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = (imgSize - hCal) // 2
            imgWhite[hGap:hGap + hCal, :] = imgResize

        # ---------- Prediction ----------
        prediction, index = classifier.getPrediction(imgWhite, draw=False)

        # Draw label background
        cv2.rectangle(imgOutput, (x - offset, y - offset - 60),
                      (x - offset + 250, y - offset),
                      (0, 255, 0), cv2.FILLED)

        # Label text
        cv2.putText(imgOutput, labels[index], (x - offset + 10, y - offset - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Bounding box
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset),
                      (0, 255, 0), 3)

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)
