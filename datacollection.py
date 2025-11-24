import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
counter = 0

folder = "C:/Users/prath/OneDrive/Desktop/Sign language detection/Data/Please"

# Create folder if it does not exist
if not os.path.exists(folder):
    os.makedirs(folder)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image.")
        break

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # ---- SAFE CROP (fixes empty imgCrop) ----
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)

        imgCrop = img[y1:y2, x1:x2]

        # Skip frame if crop is empty
        if imgCrop.size == 0:
            cv2.imshow('Image', img)
            if cv2.waitKey(1) == ord("q"):
                break
            continue

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            try:
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            except:
                continue
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wGap + wCal] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            try:
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            except:
                continue
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hGap + hCal, :] = imgResize

        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', img)

    key = cv2.waitKey(1)

    # Save image
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f"{folder}/Image_{time.time()}.jpg", imgWhite)
        print("Saved:", counter)

    # Quit program
    elif key == ord("q"):
        print("Stopping capture...")
        break

# ----- CLEAN EXIT -----
cap.release()
cv2.destroyAllWindows()
