# Gesture-to-Speech (G2S) System

Real-time hand-gesture recognition and speech pipeline with two modes:
- Camera gesture mode (sign -> text -> speech)
- STT mode (mic speech -> live English transcript -> convert to selected language on Stop)

## Current Features

- Real-time hand landmark detection using MediaPipe
- Gesture classification using trained SVM model (`models/model.pkl`)
- In-window OpenCV control panel (no separate web UI needed)
- Text-to-speech output with language-specific neural voices
- Translation support for 6 output languages:
  - English
  - Telugu
  - Hindi
  - French
  - Spanish
  - German
- STT screen with microphone controls
- Live runtime logs written to one file with timestamps

## Application Modes

## 1) Camera Gesture Mode

Main camera view + right control panel.

Buttons:
- Language buttons (6)
- `STT` (switch to Speech-to-Text screen)
- `Run` (speak current buffer)
- `Clear` (clear gesture buffer)
- `Exit` (clean shutdown)

Behavior:
- Detected gestures are buffered.
- `CONFIRM` gesture or `Run` triggers speech output.

## 2) STT Mode

Flow is now fixed to this behavior:
1. Press `Mic: Start`
2. Speak in **English input only**
3. App shows live recognized text in English while listening
4. Select target language anytime (English/Telugu/Hindi/French/Spanish/German)
5. Press `Mic: Stop`
6. Captured English text is converted to selected target language and displayed

Notes:
- Input recognition language is always English.
- Conversion happens on Stop (not continuously while speaking).

## Installation

## 1. Create and activate venv

```bash
python3 -m venv g2s_env
source g2s_env/bin/activate
```

## 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

## 3. Install required system packages (Linux/Raspberry Pi)

```bash
sudo apt-get update
sudo apt-get install -y \
  ffmpeg flac \
  portaudio19-dev libportaudio2 \
  fonts-noto-core python3-pil
```

Why these matter:
- `flac`: required by SpeechRecognition backend
- `portaudio*`: required for microphone capture via `sounddevice`
- `fonts-noto-core` + Pillow: proper rendering for non-Latin scripts (Telugu/Hindi)

## Run

```bash
python codes/main.py
```

## Logs

Runtime logs are written to:
- `codes/logs/main_latest.txt`

Current logging behavior:
- Single file only
- Overwritten on each app restart
- Each line has timestamp prefix

Example:
```text
[2026-04-20 20:02:40] Runtime log file: /home/.../codes/logs/main_latest.txt
```

## Project Structure

```text
g2s_project/
├── codes/
│   ├── main.py
│   ├── ui_app.py
│   ├── backend.py
│   ├── collect.py
│   ├── train.py
│   ├── logs/
│   │   └── main_latest.txt
│   └── .tts_cache/
├── models/
│   └── model.pkl
├── requirements.txt
└── README.md
```

## Important Runtime Notes

- If you see Raspberry Pi "Low voltage warning", performance and detection reliability will drop.
- STT and translation require internet connectivity.
- If language text appears as `????`, verify:
  - Pillow is installed in the same Python interpreter used to run app
  - Noto fonts are installed (`fonts-noto-core`)

## Training / Retraining Gestures

Collect data:
```bash
python codes/collect.py
```

Train model:
```bash
python codes/train.py
```

When to retrain:
- New gesture classes added
- Existing classes not recognized consistently
- Different lighting/background/hand style from previous dataset

## Current Known Limits

- STT input language is intentionally fixed to English.
- Gesture model quality depends on dataset diversity and camera conditions.
- Low power on Raspberry Pi can cause lag and missed detections.

