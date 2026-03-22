# Gesture-to-Speech (G2S) System

A real-time hand gesture recognition system that converts sign language gestures into spoken text with multi-language support.

## 📋 Project Overview

This project uses computer vision (MediaPipe) to detect hand landmarks from a camera feed, recognizes specific sign language gestures through a trained machine learning model, and converts the recognized signs to natural speech using text-to-speech (TTS) technology.

### Key Features

- **Real-time Hand Detection**: Uses MediaPipe to detect hand landmarks from camera input
- **Gesture Recognition**: SVM-based classification with voting mechanism for robust recognition
- **Multi-language Support**: English, Telugu, Hindi, French, and Spanish
- **Real Language Translation**: Google Translate integration for accurate language conversion (not just accent changes)
- **Text-to-Speech**: Edge-TTS with streaming audio playback via edge-playback
- **In-Camera UI**: Interactive button overlay directly in the camera window
- **Buffer Management**: Auto-sync detected signs, manual Reload/Clear controls
- **Flexible Output Modes**: CONFIRM gesture-based or manual Run button for immediate output
- **Performance Optimized**: 
  - TTS file caching in `.tts_cache/` directory
  - Translation caching in-memory
  - Fast gesture voting (VOTE_LEN=2)
  - Responsive UI with mouse click handling
- **Robust Camera Handling**: Automatic camera discovery, sanity checks, V4L2 fallback support

## 🛠️ System Requirements

- **Python**: 3.8 or higher
- **Camera**: USB or integrated camera with V4L2 support
- **OS**: Linux recommended (tested on Raspberry Pi and desktop Linux)
- **Display**: X11 or framebuffer support (can run headless with streaming)
- **Libraries**: See `requirements.txt`

### Minimum Hardware
- Raspberry Pi 4 (2GB RAM) or equivalent
- USB camera or Raspberry Pi Camera Module

## 📦 Installation

### 1. Clone Repository
```bash
cd /home/rsaisrujan/g2s_project
```

### 2. Create Virtual Environment
```bash
python3 -m venv g2s_env
source g2s_env/bin/activate  # On Linux/Mac
# or on Windows:
# g2s_env\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### Key Dependencies
- **opencv-python**: Camera input and display
- **mediapipe**: Hand landmark detection
- **scikit-learn**: SVM gesture classification
- **edge-tts**: Text-to-speech synthesis
- **edge-playback**: Streaming audio playback
- **deep-translator**: Real-time language translation
- **numpy**: Numerical computations

## 🚀 Quick Start

### Run Main Application
```bash
python codes/main.py
```

This launches the **in-camera UI** mode where:
- Camera feed displays in the left portion (800×600)
- Interactive UI panel appears on the right (260px wide)
- Real-time hand gesture detection and recognition
- Click buttons to control behavior

### (Optional) Standalone Tkinter UI
```bash
python codes/ui_app.py
```

Provides a separate Tkinter window for testing UI independently (without camera).

## 🎮 How to Use

### 1. Starting the Application
```bash
python codes/main.py
```

**Wait for startup**:
- Camera will initialize (may take 3-5 seconds)
- "Idle" status is displayed at the bottom

### 2. Camera Position & Lighting
- Position hand 20-50cm from camera
- Ensure adequate lighting (avoid backlighting)
- Face palm toward camera for best detection
- Full hand in frame preferred

### 3. Performing Gestures
- Perform sign language gestures in front of the camera
- Each recognized gesture appears as text in the **buffer display** (top of UI panel)
- Buffer auto-updates as new signs are detected (no button needed)

### 4. Control Interface (In-Camera UI Panel)

**Layout** (left to right, top to bottom):
```
┌─────────────────┐
│  Buffer Display │  ← Shows collected sign text
├─────────────────┤
│ EN   TE   HI    │  ← Language selection (English, Telugu, Hindi)
│ FR   ES         │  ← More languages (French, Spanish)
├─────────────────┤
│ Reload    Run   │  ← Reload previous output / Run without CONFIRM
├─────────────────┤
│ Clear     Exit  │  ← Clear buffer / Exit application
├─────────────────┤
│ Selected: EN    │  ← Status indicator
└─────────────────┘
```

### 5. Output Modes

#### Option A: CONFIRM Gesture (Automatic)
1. Perform sign gestures to build buffer
2. Show **CONFIRM gesture** (defined in model training)
3. System speaks the buffered text in selected language
4. Buffer automatically clears for next sequence

#### Option B: Run Button (Manual)
1. Perform sign gestures to build buffer
2. **Click the Run button** on UI panel
3. System speaks the buffered text in selected language
4. Buffer automatically clears for next sequence

### 6. Control Buttons

| Button | Action |
|--------|--------|
| **EN / TE / HI / FR / ES** | Select output language (highlights when active) |
| **Reload** | Restores the previous output back to buffer (allows re-speak) |
| **Run** | Immediately output current buffer as speech (without CONFIRM gesture) |
| **Clear** | Empty the current buffer (discard unsent signs) |
| **Exit** | Gracefully close application (cleans up camera and threads) |

### 7. Workflow Example

```
1. Start app → "Idle" status
2. Perform signs: "HELLO" "WORLD" → Buffer shows: "HELLO WORLD"
3. Click "Run" button → Speaks: "Hello world" (in English)
4. Buffer clears → "Idle" status
5. Perform new signs: "THANKS" → Buffer shows: "THANKS"
6. Perform CONFIRM gesture → Speaks: "Thanks"
7. Buffer clears → "Idle" status
8. Click "Reload" → Buffer restores previous: "THANKS"
9. Click "Clear" → Buffer erased → "Idle" status
10. Click "Exit" → App closes with cleanup
```

## 📁 Project Structure

```
g2s_project/
├── codes/
│   ├── main.py           # Main app: camera capture, detection, TTS, UI state
│   ├── ui_app.py         # UI rendering (OpenCV & Tkinter classes)
│   ├── backend.py        # Gesture recognition: SVM voting, cooldown logic
│   ├── collect.py        # Data collection harness for training
│   ├── train.py          # Model training pipeline
│   └── __pycache__/      # Python cache
├── models/               # Trained SVM model directory
├── g2s_env/              # Virtual environment
├── .tts_cache/           # TTS audio file cache (auto-created)
└── README.md             # This file
```

## 🔧 Configuration & Tuning

### Gesture Recognition Parameters (backend.py)

```python
VOTE_LEN = 2                      # Vote window for stability (lower = faster)
CONF_THRESHOLD = 0.5              # Confidence threshold (0.0-1.0)
MIN_GAP_ANY = 0.0                 # Min gap between different signs
COOLDOWN_SAME_LABEL = 0.1         # Cooldown for repeated same sign (seconds)
```

**To adjust**: Edit `codes/backend.py` and restart application

### Camera Selection

The system auto-discovers cameras:
1. Probes `/dev/video0` through `/dev/video3`
2. Performs sanity checks (can read frames)
3. Uses first working camera
4. Falls back to V4L2 backend if default fails

**To force specific camera**: Edit `codes/main.py` line with `camera_index` in `discover_camera_indices()`

### Language & Voice

- **Selected Language**: Click button in UI (highlights active)
- **Output Voice**: Matches language (auto-selected by edge-tts)
- **Translation**: Real-time Google Translate (stored in memory cache)

## 📊 Troubleshooting

### Camera Not Opening
```
Error: Camera not accessible
```
**Solution**:
- Check `/dev/video*` permissions: `ls -la /dev/video*`
- Grant permissions: `sudo usermod -a -G video $USER`
- Restart application
- Try different camera index in `main.py`

### No Audio Output
```
No sound from speaker
```
**Solution**:
1. Check speaker/headphone connection
2. Test audio device: `aplay -l` (on Linux)
3. Verify TTS is generating: Check `.tts_cache/` for .mp3 files
4. Fallback chain: ffplay → cvlc → mpg123 → playsound
5. Install missing player: `sudo apt-get install ffmpeg vlc mpg123`

### Gestures Not Recognized
```
Buffer not updating with signs
```
**Solution**:
- Ensure hand is fully visible in camera frame
- Check lighting (avoid backlight)
- Position hand 20-50cm from camera
- Verify hand landmarks visible in console output
- Retrain model with current gesture data (see **Training** section)

### TTS Latency / Slow Speech Output
```
Delay from button click to audio
```
**Solution**:
- First output after app start may be slower (TTS library loading)
- Subsequent outputs use `.tts_cache/` (no delay expected)
- Clear cache if corrupted: `rm -rf .tts_cache/`
- Check system resources: `top`, `free -h`

### UI Buttons Not Clickable
```
Clicking on panel has no effect
```
**Solution**:
- Ensure left mouse button is used
- Click within button rectangle bounds
- Try different language button first to verify
- Check console for "UI action" logs

## 🎓 Training Custom Gestures

### Collect Training Data
```bash
python codes/collect.py
```
**Interactive steps**:
1. Select gesture name (e.g., "HELLO")
2. Show gesture to camera ~50 times
3. System captures hand landmarks
4. Repeat for other signs

**Output**: Training data saved to `data/` directory

### Train Model
```bash
python codes/train.py
```
**Output**: Trained SVM model saved to `models/` directory

### Reload in Main App
- Model auto-loads on `main.py` startup
- No additional steps needed

## 📝 Logs & Debugging

### Enable Verbose Logging
Edit `codes/main.py` and change logging levels (search for `logging.basicConfig`).

### View Detection Console Output
When running `python codes/main.py`:
- "Detected sign:" logs show recognized gestures
- "Speaking:" logs show TTS output
- "UI action:" logs show button clicks
- "Reloaded:" shows buffer reload

### Common Log Messages
```
[INFO] Camera initialized: /dev/video0
[INFO] Detected sign: HELLO (confidence: 0.85)
[INFO] Speaking: "Hello world" in language: en
[INFO] UI action: run (triggered by button)
[INFO] Reloaded last buffer: THANKS
```

## 🌍 Multi-Language Notes

### Supported Languages
- **English** (en) - Default, clear pronunciation
- **Telugu** (te) - Local Indian language, phonetic accuracy
- **Hindi** (hi) - Indian language, diverse voice options
- **French** (fr) - European language, accent handling
- **Spanish** (es) - Spanish language, natural flow

### How Translation Works
1. User selects language via UI button (highlights selected)
2. When output triggered (CONFIRM or Run):
   - Buffer text (typically English sign labels) is translated to selected language
   - Google Translate API handles conversion
   - Translation cached in-memory for repeated signs
3. Edge-TTS generates speech in selected language
4. Audio played via speaker

### Accent vs Translation
- **Old behavior**: Phonetic spelling changes (e.g., "THANK YOU" → "THENK YOO") — accent only
- **Current behavior**: Real language translation (e.g., "THANK YOU" → Spanish: "Gracias") — full translation

## 🔌 Dependencies & Licenses

| Package | Purpose | License |
|---------|---------|---------|
| opencv-python | Camera input & display | Apache 2.0 |
| mediapipe | Hand detection | Apache 2.0 |
| scikit-learn | SVM model | BSD |
| edge-tts | Text-to-speech | GPL-3.0 |
| edge-playback | Audio streaming | MIT |
| deep-translator | Language translation | MIT |
| numpy | Numerical ops | BSD |

## 🚀 Performance Tips

- **Lower Latency**: Click "Run" instead of waiting for CONFIRM gesture
- **Faster Startup**: Keep TTS cache intact (don't delete `.tts_cache/`)
- **Smooth Camera**: Ensure adequate lighting and frame rate (30+ FPS ideal)
- **Reduced CPU**: Lower video resolution in `main.py` if needed

## 📞 Known Limitations

1. **Camera**: Requires USB or integrated camera with V4L2 support
2. **Hand Pose**: Works best with open palm, full hand visible
3. **Translation**: Relies on Google Translate (may fail without internet)
4. **Audio**: Requires speaker output or headphones
5. **Performance**: Raspberry Pi Zero may struggle; Pi 4+ recommended

## 🔄 Future Enhancements

- [ ] Word-by-word custom dictionary for consistent sign labels
- [ ] Headless SSH mode (no X11 display needed)
- [ ] Performance optimization for slower hardware
- [ ] Support for continuous gesture streaming
- [ ] Gesture confidence feedback UI
- [ ] Customizable button layout
- [ ] Data logging and statistics

## 📧 Troubleshooting Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment activated
- [ ] All dependencies installed (`pip list | grep -E "opencv|mediapipe|scikit"`)
- [ ] Camera connected and working (`ls /dev/video*`)
- [ ] Speaker/audio functional
- [ ] Network connection available (for TTS/Translation)
- [ ] Model file exists in `models/` directory
- [ ] Sufficient disk space for `.tts_cache/`

## 📜 Version History

**Current Version**: 1.0
- Real-time hand gesture recognition
- Multi-language speech output
- In-camera UI with button controls
- TTS and translation caching
- Robust camera initialization
- Manual Run button (alternative to CONFIRM gesture)

---

**Last Updated**: March 2026  
**Maintained by**: G2S Development Team
