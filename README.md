 Dash Robot Controller (Test1.py)

A PyQt5 GUI app to control a Dash robot with:
- Camera preview, face detection/tracking (Haar + optional face_recognition)
- Offline speech recognition (Vosk) + mic gain
- Optional Ollama NLU (local LLM) and smart replies
- Piper TTS with pyttsx3 fallback
- Tablet detection (ArUco/Color), medication reminders, BLE, choreography

This repository contains one main script: `Test1.py`.

Quick start

Commands to run on your computer:

```bash
# 1) Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2) Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3) (Optional) set environment variables
# Copy the example and edit values (Ollama, Vosk model path, Piper)
# macOS/Linux:
cp .env.example .env
# Windows PowerShell:
copy .env.example .env

# 4) Run the app
python Test1.py
```

If you see missing system libraries on Linux, install:
```bash
# Debian/Ubuntu
sudo apt-get update
sudo apt-get install -y libqt5multimedia5 libqt5multimedia5-plugins libportaudio2
```

Optional features

Offline ASR (Vosk)
- Download a Vosk model (e.g., vosk-model-small-en-us-0.15) and set `VOSK_MODEL_PATH` to that folder in `.env`.

Google SpeechRecognition backend
- Install PyAudio (may require OS libs):
  - Windows: `pip install PyAudio`
  - macOS: `brew install portaudio && pip install PyAudio`
  - Ubuntu/Debian: `sudo apt-get install -y portaudio19-dev && pip install PyAudio`

Piper TTS
- Install the `piper` binary and a voice model.
- Set `PIPER_BIN` (if not in PATH) and `PIPER_MODEL` to the `.onnx` voice file.

Ollama (local LLM)
- Install Ollama: https://ollama.com
- Pull a model: `ollama pull qwen2.5:1.5b-instruct`
- Set `OLLAMA_URL` and `OLLAMA_MODEL` in `.env`.

ArUco tablet detection
- `opencv-contrib-python` provides `cv2.aruco`. In the app UI, select Tablet Mode: ArUco.

### Face recognition (optional)
- `face-recognition` (dlib) is optional and may need a compiler toolchain to install.
- If not installed, the app falls back to Haar detection.

Install the full optional set:
```bash
pip install -r requirements-all.txt
```

Common issues
Head Tracking is not working 
