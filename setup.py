from setuptools import setup

setup(
    name="dash-robot-controller",
    version="0.1.0",
    description="Dash robot controller GUI (PyQt5) with Vosk ASR, Ollama NLU (optional), Piper TTS, BLE",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    install_requires=[
        "PyQt5>=5.15.9",
        "numpy>=1.24",
        "Pillow>=9.5",
        "opencv-contrib-python>=4.7.0",
        "sounddevice>=0.4.6",
        "vosk>=0.3.45",
        "SpeechRecognition>=3.10.0",
        "pyttsx3>=2.90",
        "bleak>=0.22.2",
        "aubio>=0.4.9",
        "simpleaudio>=1.0.4",
    ],
    extras_require={
        "face": ["face-recognition>=1.3.0"],
        "google-asr": ["PyAudio>=0.2.14"],
        "all": ["face-recognition>=1.3.0", "PyAudio>=0.2.14"],
    },
)
