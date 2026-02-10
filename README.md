# 🤟 ISL Recognition System (Indian Sign Language)

An AI-powered computer vision project that detects and recognizes Indian Sign Language gestures using deep learning and real-time hand tracking.

🚀 **Project Goal:**  
To enable real-time recognition of hand signs using a webcam and classify them using a trained deep learning model.

---

## 📖 Overview

This project builds a complete pipeline for **sign language recognition**, including:

- Hand detection using computer vision  
- Dataset creation using webcam  
- Deep learning model training (MobileNetV2)  
- Real-time gesture recognition  

It helps bridge communication gaps and demonstrates how AI can assist accessibility.

---

## ✨ Features

### 🎥 Real-Time Hand Detection
- Uses webcam input  
- Detects hand region using cvzone hand tracking  
- Crops and normalizes hand images  

### 🗂️ Dataset Creation
- Capture images for each sign  
- Automatically stores images in class-wise folders  
- Generates training-ready dataset  

### 🧠 Deep Learning Model
- MobileNetV2 transfer learning  
- Trained on custom ISL dataset  
- Multi-class sign classification  

### 📊 Training Visualization
- Accuracy vs Epoch graph  
- Loss vs Epoch graph  
- Class mapping generation  

### 💾 Model Export
- Saves trained model as `keras_model.h5`  
- Saves class labels in `labels.txt`  

---

## 🛠️ Tech Stack

- Python  
- TensorFlow / Keras  
- OpenCV  
- cvzone  
- NumPy  
- Matplotlib  

---

## 📂 Project Structure

ISL-Recognition/
│
├── datacollection.py
├── train_model.py
├── test.py
├── keras_model.h5
├── labels.txt
│
└── Data/
    ├── Hello/
    ├── Thanks/
    ├── Please/
    └── ...

---

## ▶️ How It Works

Step 1 — Collect Dataset

python datacollection.py
Press S → capture image

Press Q → quit

Images saved class-wise inside Data/.

Step 2 — Train Model
python train_model.py

This will:

Load dataset

Train MobileNetV2 model

Plot accuracy & loss graphs

Save model + labels

Step 3 — Use Model for Recognition

Load keras_model.h5 and labels.txt in a prediction script to perform real-time classification.

🧠 Model Details

Architecture: MobileNetV2 (Transfer Learning)

Input Size: 224 × 224

Output: Softmax multi-class classification

Optimizer: Adam

Loss: Categorical Crossentropy

📊 Training Output

Accuracy vs Epoch graph

Loss vs Epoch graph

Class index mapping

🎯 Use Cases

Assistive communication tools

Accessibility research

Computer vision learning

Academic projects

AI + healthcare innovations

🚀 Future Improvements

Real-time sign prediction script

Sentence formation from signs

LSTM for continuous gestures

Deploy as web app

Add more ISL gestures

👨‍💻 Author

Pratham Patil
BMS College of Engineering
ISE | AI | Computer Vision | Deep Learning


