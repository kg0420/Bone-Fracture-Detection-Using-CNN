# 🦴 Fracture Detection System using Deep Learning

An AI-powered web application that detects and classifies fracture regions from X-ray images using a Convolutional Neural Network (CNN).  
The system is designed with safety checks to avoid unreliable predictions and is deployed as a user-friendly web interface.

---

## 🚀 Project Overview

Fracture detection from X-ray images is a critical task in medical diagnosis.  
This project uses **Deep Learning (CNN + MobileNetV2)** to classify X-ray images into different fracture-related categories with high confidence.

To improve reliability:
- Predictions are shown **only when confidence ≥ 95%**
- Invalid or non–X-ray images are safely rejected
- Dataset integrity issues are handled during training

---

## 🧠 Model Details

- **Architecture:** Transfer Learning with MobileNetV2  
- **Input Size:** 256 × 256 × 3  
- **Classes (8):**
  - XR_ELBOW  
  - XR_FINGER  
  - XR_FOREARM  
  - XR_HAND  
  - XR_HUMERUS  
  - XR_SHOULDER  
  - XR_WRIST  
  - NOT_XRAY (for rejecting non-X-ray images)

- **Loss Function:** Categorical Crossentropy  
- **Optimizer:** Adam  

---

## 🖥️ Web Application Features

- 📤 Upload X-ray image
- 🧠 AI-based fracture classification
- 📊 Confidence score display
- ⚠️ Low-confidence rejection (< 95%)
- 🎨 Modern 3D UI using Tailwind CSS
- 🔐 Safe handling of invalid images

---

## 🛠️ Tech Stack

### Backend / ML
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Pillow

### Frontend
- HTML
- Tailwind CSS (CDN)
- Glassmorphism & 3D UI effects

### Deployment
- Hugging Face Spaces (for ML demo)
- GitHub Pages (for portfolio)

---
## 📁 Project Structure


├── app.py

├── Fracture_Detection_Model.h5

├── requirements.txt

├── templates/

│ └── index.html

├── static/

│ └── uploads/

└── README.md




