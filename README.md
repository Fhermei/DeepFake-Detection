# 🎬 DeepFake Detection App

An AI-powered mobile application that detects whether a video is **REAL** or **DEEPFAKE** using a trained deep learning model.

Built with **React Native (Expo)** for the mobile app and **Flask + TensorFlow** for the backend.

---

## 📱 How It Works

1. User opens the app on their phone
2. Selects a video from their gallery
3. The app sends the video to the AI backend
4. The model analyzes 30 frames from the video
5. Returns a verdict: ✅ **REAL** or ⚠️ **DEEPFAKE** with confidence score

---

## 🗂️ Project Structure

```
DeepFake-Detection/
│
├── DeepfakeDetector/          # React Native mobile app
│   ├── App.js                 # Main app file
│   └── package.json
│
├── backend_api.py             # Flask API server
├── requirements.txt           # Python dependencies
├── deepfake_detector_final.h5 # Trained AI model
└── README.md
```

---

## 🚀 Deployment

### Backend (Render.com)
The Flask API is deployed on Render.com and runs 24/7 in the cloud.

**Live API URL:** `https://your-app-name.onrender.com`

### Mobile App (Expo)
The React Native app is published on Expo and can be opened by anyone with the Expo Go app.

**Expo Link:** `exp://exp.host/@your-username/DeepfakeDetector`

---

## 🧠 Model Details

- **Architecture:** Custom CNN trained on video frames
- **Input:** 30 frames extracted from video, resized to 224×224
- **Output:** Probability score (< 0.5 = Deepfake, > 0.5 = Real)
- **Label Mapping:** 0 = DEEPFAKE, 1 = REAL
- **Framework:** TensorFlow / Keras

---

## ⚙️ Local Development

### Run the Backend
```bash
pip install -r requirements.txt
python backend_api.py
```

### Run the Mobile App
```bash
cd DeepfakeDetector
npx expo start
```

---

## 📊 Prediction Output

The app shows:
- **Verdict:** REAL or DEEPFAKE
- **Confidence:** Percentage score
- **Probability breakdown:** Deepfake % vs Real %
- **Video info:** Duration, frames, FPS, resolution

---

## 👩‍💻 Built By

Oyewole Oluwafemi 
Nigerian Political Deepfake Detection System  
FERA Research Group

---

## 📄 License

This project is for academic research purposes.