# 🚗 AI Drowsiness Detection System

An AI-powered real-time Driver Drowsiness Detection System that monitors a driver's eye movements, yawning, and head position using Computer Vision and MediaPipe. The system provides both voice and visual alerts when signs of drowsiness or distraction are detected, helping improve road safety.

---

## 📌 Features

- 👁️ Real-time Eye Aspect Ratio (EAR) based drowsiness detection
- 😮 Yawning detection using Mouth Aspect Ratio (MAR)
- 🧑 Head pose estimation for distraction detection
- 🔊 Voice alert using Text-to-Speech (pyttsx3)
- 🚨 Visual warning messages on the screen
- 📷 Live webcam monitoring
- ⚡ Fast and lightweight real-time processing

---

## 🛠️ Technologies Used

- Python
- OpenCV
- MediaPipe Face Landmarker
- NumPy
- pyttsx3
- TensorFlow (MediaPipe backend)

---

## 📂 Project Structure

```
AI-Drowsiness-Detection/
│
├── drowsiness_detector.py
├── face_landmarker.task
```

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/Sritharan2005/AI-Drowsiness-Detection.git
```

### Navigate to the project

```bash
cd AI-Drowsiness-Detection
```

### Install required packages

```bash
pip install mediapipe opencv-python numpy pyttsx3
```

---

## ▶️ Run the Project

```bash
python drowsiness_detector.py
```

---

## 🔄 Workflow

1. Capture live video from webcam
2. Detect face using MediaPipe Face Landmarker
3. Detect facial landmarks
4. Calculate Eye Aspect Ratio (EAR)
5. Calculate Mouth Aspect Ratio (MAR)
6. Estimate head direction
7. Detect:
   - Drowsiness
   - Yawning
   - Driver distraction
8. Generate voice and visual alerts
9. Continue monitoring in real time

---

## 📸 Output

The system displays:

- EAR value
- MAR value
- Head direction
- Drowsiness warning
- Yawning warning
- Driver distraction warning
- Voice alerts

---

## 🚀 Future Improvements

- Blink rate analysis
- Mobile application integration
- Driver identity recognition
- Night vision support
- SMS alert to emergency contact
- Cloud monitoring dashboard
- Fatigue prediction using Deep Learning

---

## 👨‍💻 Author

**Sritharan R**

🎓 B.E. Computer Science and Engineering (AI & ML)

📧 Email: srirogu@gmail.com

🔗 GitHub: https://github.com/Sritharan2005

🔗 LinkedIn: https://www.linkedin.com/in/sritharan-ravi-a31d2005

---

## ⭐ Support

If you found this project helpful, consider giving it a ⭐ on GitHub.

---

## 📜 License

This project is intended for educational and research purposes.
