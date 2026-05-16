# SafeSight AI 👁️🤖

An industrial safety monitor utilizing computer vision and real-time hand-gesture tracking to provide automated machine control and emergency-stop functionality.

---

## 🚀 Core Features
* **Real-Time Hand Tracking:** Integrates high-fidelity tracking frameworks to identify worker positioning.
* **Dynamic Machine Control:** Processes distinct hand gestures to issue instantaneous commands (e.g., Start, Stop, and Slow).
* **Industrial Safety Overrides:** Designed to cut machine power immediately upon detecting emergency or hazardous posture cues.

---

## 🛠️ System Architecture & Tech Stack
* **Language:** Python
* **Frameworks & Libraries:** OpenCV, MediaPipe, TensorFlow / Keras
* **Interface & Processing:** Custom multi-threaded dashboard scripts (`main_dashboard.py`) for low-latency visual data streams.

---

## 📁 File Structure
* `main.py`: The core application coordinator initializing video capture and model inference loops.
* `main_dashboard.py`: Handles the user interface layout, camera preview windows, and status logs.
* `data_collection.py`: Custom pipeline script for capturing image frames and gathering dataset arrays.
* `requirements.txt`: Lists all package dependencies needed to deploy the project locally.

---

## ⚙️ Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/taibaabid/SafeSight-AI.git](https://github.com/taibaabid/SafeSight-AI.git)
   cd SafeSight-AI
