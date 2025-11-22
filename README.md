# 🔍 AI Video Person Search (YOLO12 & DeepFace)

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![YOLO12](https://img.shields.io/badge/YOLO-12-00FFFF?style=for-the-badge)
![DeepFace](https://img.shields.io/badge/DeepFace-Face%20Recognition-FF6F00?style=for-the-badge)
![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-5-C51A4A?style=for-the-badge&logo=Raspberry%20Pi&logoColor=white)

> **Smart Video Analysis System for Embedded Devices**
>
> Automatically detects specific people in YouTube videos using **YOLO12** for detection and **DeepFace** for recognition.

---

## 📖 Overview
This project is an intelligent surveillance solution optimized for the **Raspberry Pi 5**. It efficiently processes video streams to identify when a target person appears. By combining the speed of **YOLO12** with the accuracy of **DeepFace**, it achieves robust performance even in resource-constrained environments.

### 🎯 Key Features
- **🚀 High Efficiency**: Implements adaptive frame skipping (`checks_per_sec`) to maintain FPS on Raspberry Pi.
- **🧠 Hybrid AI Engine**: 
  - **Detection**: YOLO12 (Fast & Lightweight)
  - **Identification**: DeepFace (ArcFace/SFace models)
- **📹 Auto-Processing**: Downloads YouTube videos via `yt-dlp` and analyzes them instantly.
- **📊 Smart Logging**: Merges consecutive timestamps into readable intervals (e.g., `00:10 ~ 00:15`).

---

## 🛠️ Tech Stack

| Category | Technology | Description |
| :--- | :--- | :--- |
| **Hardware** | Raspberry Pi 5 | 8GB RAM Recommended |
| **Language** | Python 3.x | Core Logic |
| **Detection** | **YOLO12** | Ultralytics SOTA Object Detection |
| **Recognition** | **DeepFace** | Facial Analysis Framework |
| **Utils** | OpenCV, yt-dlp | Video Processing & Downloading |

---

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/0-vin/YOLOv12-DeepFace-Video-Analyzer.git
cd YOLOv12-DeepFace-Video-Analyzer
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Usage
Run the script with your target video and face image.
```bash
python main.py --url "YOUTUBE_URL" --face "target_person.jpg" --cps 2
```

---

## 🔧 Arguments

| Argument | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| **--url** | str | (Required) YouTube Video URL | -
| **--face** | str | (Required) Target face image path | -
| **--cps** | float | Checks per second (Efficiency control) | 2.0
| **--yolo** | str | Path to YOLO model file | yolov12n-face.pt
| **--deepface** | str | "DeepFace model name (ArcFace, SFace)" | ArcFace

---

## 📊 [RESULT REPORT]
The system generates a clean report file automatically.

```Plaintext
Video: https://youtu.be/example_video
Target: my_photo.jpg
Model: YOLO12 + ArcFace
Process Time: 45.20s
----------------------------------------
Found Intervals:
✅ 00:12.50 ~ 00:15.00 (Distance: 0.32)
✅ 01:23.10 ~ 01:28.40 (Distance: 0.41)
✅ 03:45.00 ~ 03:45.50 (Distance: 0.29)
```

---

## 👨‍💻 Author
**(0-vin)**
### 📧 Contact: https://github.com/0-vin

"The more you explore, the more you grow."





