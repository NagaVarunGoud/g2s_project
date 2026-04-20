import os

# Optional: suppress noisy native stderr logs from MediaPipe/TFLite on Linux ARM.
# Default is ON to keep terminal focused on important app-level logs.
if os.environ.get("G2S_SUPPRESS_NATIVE_LOGS", "1") == "1":
    try:
        _devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(_devnull_fd, 2)
        os.close(_devnull_fd)
    except OSError:
        pass

# Detect display availability BEFORE importing cv2/Qt-linked libraries.
# Hard-coding "xcb" crashes when there is no X11 server (e.g. plain SSH session).
if os.environ.get("DISPLAY"):
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
else:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Suppress noisy runtime logs from MediaPipe / TensorFlow / OpenCV.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")

import cv2
import mediapipe as mp
import numpy as np
import backend
import time
import threading
import asyncio
import edge_tts
from playsound import playsound
from deep_translator import GoogleTranslator
import signal
import sys
import shutil
import subprocess
import glob
import hashlib
import re
from datetime import datetime
from collections import deque
from ui_app import OpenCVCameraUI, OpenCVSTTUI

try:
    import sounddevice as sd
except Exception:
    sd = None

try:
    import speech_recognition as sr
except Exception:
    sr = None

if hasattr(cv2, "setLogLevel"):
    try:
        cv2.setLogLevel(0)  # silent
    except Exception:
        pass


LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOG_DIR, "main_latest.txt")


class TeeStream:
    """Mirror writes to terminal and a log file with per-line timestamps."""

    def __init__(self, *streams):
        self.streams = streams
        self._at_line_start = True

    def _prefix_with_timestamp(self, text):
        if not text:
            return text

        out = []
        for ch in text:
            if self._at_line_start:
                out.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ")
                self._at_line_start = False
            out.append(ch)
            if ch == "\n":
                self._at_line_start = True
        return "".join(out)

    def write(self, data):
        text = str(data)
        stamped = self._prefix_with_timestamp(text)
        for stream in self.streams:
            try:
                stream.write(stamped)
            except Exception:
                pass
        return len(text)

    def flush(self):
        for stream in self.streams:
            try:
                stream.flush()
            except Exception:
                pass


def setup_runtime_log_capture():
    log_file = open(LOG_FILE_PATH, "w", buffering=1, encoding="utf-8")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, log_file)
    sys.stderr = TeeStream(original_stderr, log_file)
    print(f"Runtime log file: {LOG_FILE_PATH}")


setup_runtime_log_capture()

# ---------------------------------------------------------------------------
# Camera connection note
# ---------------------------------------------------------------------------
# The camera must be plugged into the Raspberry Pi (the machine running this
# script), NOT into the local computer that is SSH-ing into the Pi.
# If you want a live preview window over SSH, use X11 forwarding:
#   ssh -X <user>@<raspberrypi-ip>
# ---------------------------------------------------------------------------

has_display = bool(os.environ.get("DISPLAY"))
if not has_display:
    print("=" * 60)
    print("G2S SYSTEM - headless mode (no display detected)")
    print("  Detected gestures will be printed to the terminal.")
    print("  CAMERA: connect it to the Raspberry Pi, not to your")
    print("          local computer running VS Code SSH.")
    print("  For a live preview, use:  ssh -X <user>@<pi-ip>")
    print("=" * 60)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

CAMERA_CAPTURE_W = 640
CAMERA_CAPTURE_H = 480
CAMERA_VIEW_W = 960
CAMERA_VIEW_H = 720
DETECTION_W = 320
DETECTION_H = 240
UI_PANEL_W = 248
WINDOW_W = CAMERA_VIEW_W + UI_PANEL_W
FALLBACK_SCREEN_W = 480
FALLBACK_SCREEN_H = 320


def detect_display_size():
    if not os.environ.get("DISPLAY"):
        return None

    try:
        proc = subprocess.run(
            ["xrandr", "--current"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        match = re.search(r"current\s+(\d+)\s+x\s+(\d+)", proc.stdout)
        if not match:
            return None
        return int(match.group(1)), int(match.group(2))
    except Exception:
        return None

CAMERA_DEVICE_PATH = "/dev/video0"

hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.35,
    min_tracking_confidence=0.35
)


def kill_camera_users(device_path, include_self=False):
    if not os.path.exists(device_path):
        return

    fuser_path = shutil.which("fuser")
    if not fuser_path:
        return

    proc = subprocess.run(
        [fuser_path, device_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )

    current_pid = os.getpid()
    for token in proc.stdout.split():
        if not token.isdigit():
            continue
        pid = int(token)
        if pid == current_pid and not include_self:
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            continue


def discover_camera_indices(max_index=3):
    indices = []
    for path in sorted(glob.glob("/dev/video*")):
        suffix = path.replace("/dev/video", "")
        if suffix.isdigit():
            idx = int(suffix)
            if 0 <= idx <= max_index:
                indices.append(idx)

    unique = sorted(set(indices))
    if not unique:
        return [0, 1, 2, 3]
    return unique


def open_camera_with_sanity(index):
    # Try V4L2 first, then generic backend fallback.
    for backend_flag in (cv2.CAP_V4L2, None):
        cap_local = cv2.VideoCapture(index, backend_flag) if backend_flag is not None else cv2.VideoCapture(index)
        if not cap_local.isOpened():
            cap_local.release()
            continue

        ok, _ = cap_local.read()
        if ok:
            return cap_local

        cap_local.release()

    return None


# Release stale holders before opening the camera.
kill_camera_users(CAMERA_DEVICE_PATH)

cap = None
active_camera_path = CAMERA_DEVICE_PATH
for cam_index in discover_camera_indices():
    cam_path = f"/dev/video{cam_index}"
    kill_camera_users(cam_path)
    candidate = open_camera_with_sanity(cam_index)
    if candidate is not None:
        cap = candidate
        active_camera_path = cam_path
        print(f"Using camera index {cam_index} ({cam_path})")
        break
    else:
        print(f"Camera index {cam_index} not usable, trying next...")

if cap is None:
    print("ERROR: Camera not accessible")
    print("Check camera connection, permissions, and whether /dev/video* exists.")
    sys.exit(1)

# Faster camera
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_CAPTURE_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_CAPTURE_H)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 30)

actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
if actual_w and actual_h:
    print(f"Camera stream resolution: {actual_w}x{actual_h}")

if has_display:
    screen_size = detect_display_size() or (FALLBACK_SCREEN_W, FALLBACK_SCREEN_H)
    screen_w, screen_h = screen_size
    # Keep panel readable while guaranteeing the composed frame fits display.
    UI_PANEL_W = max(150, int(screen_w * 0.30))
    UI_PANEL_W = min(UI_PANEL_W, max(150, screen_w - 170))
    CAMERA_VIEW_W = max(160, screen_w - UI_PANEL_W)
    CAMERA_VIEW_H = max(240, screen_h)
    WINDOW_W = CAMERA_VIEW_W + UI_PANEL_W
    print(f"Display size: {screen_w}x{screen_h} -> camera {CAMERA_VIEW_W}x{CAMERA_VIEW_H}, panel {UI_PANEL_W}")

if has_display:
    cv2.namedWindow("G2S SYSTEM", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("G2S SYSTEM", WINDOW_W, CAMERA_VIEW_H)
    cv2.setWindowProperty("G2S SYSTEM", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow("G2S SYSTEM", 0, 0)
camera_ui = OpenCVCameraUI(CAMERA_VIEW_W, CAMERA_VIEW_H, UI_PANEL_W)
stt_ui = OpenCVSTTUI(WINDOW_W, CAMERA_VIEW_H)

prev_time = time.perf_counter()
fps_history = deque(maxlen=10)

def normalize(landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    coords = coords - coords[0]
    scale = np.max(np.linalg.norm(coords, axis=1))
    coords = coords / (scale if scale else 1)
    return coords.flatten()

VOICE_FILE = "voice.mp3"
VOICE_NAME = "en-US-JennyNeural"
VERBOSE_DETECTIONS = True
SENTENCE_GAP_SECONDS = 0.0
AUDIO_BOOST_PERCENT = 306
AUDIO_BOOST_MULTIPLIER = 1.0 + (AUDIO_BOOST_PERCENT / 100.0)
TTS_RATE_PERCENT = -10


def tts_percent_str(value):
    return f"{value:+d}%"

LANGUAGE_TO_VOICE = {
    "English": "en-US-JennyNeural",
    "Telugu": "te-IN-MohanNeural",
    "Hindi": "hi-IN-SwaraNeural",
    "French": "fr-FR-DeniseNeural",
    "Spanish": "es-ES-ElviraNeural",
    "German": "de-DE-KatjaNeural",
}
LANGUAGE_TO_CODE = {
    "English": "en",
    "Telugu": "te",
    "Hindi": "hi",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
}
STT_LANGUAGE_TO_CODE = {
    "English": "en-US",
    "Telugu": "te-IN",
    "Hindi": "hi-IN",
    "French": "fr-FR",
    "Spanish": "es-ES",
    "German": "de-DE",
}
STT_INPUT_LANGUAGE = "English"
STT_CHUNK_SECONDS = 1.2
selected_language = "English"
ui_status = "Idle"
ui_mode = "camera"
stt_converted_text = ""
stt_raw_text = ""
stt_listening = False
stt_thread = None
stt_stop_event = threading.Event()
stt_device_index = None
stt_device_name = "Mic: Detecting..."
stt_capture_language = STT_INPUT_LANGUAGE
stt_display_language = "English"
stored_buffer_lines = []
last_output_buffer_lines = []
TTS_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".tts_cache")
os.makedirs(TTS_CACHE_DIR, exist_ok=True)
translation_cache = {}
exit_requested = False


def set_status(value):
    global ui_status
    ui_status = value


def set_language(language):
    global selected_language
    selected_language = language
    print(f"Selected language: {language}")
    if ui_mode == "stt" and stt_listening:
        set_status(f"Listening ({STT_INPUT_LANGUAGE}) | Target: {language}")
    else:
        set_status(f"Language: {language}")


def _preferred_stt_device():
    global stt_device_name

    if sd is None:
        stt_device_name = "Mic: PortAudio missing"
        return None, None

    try:
        devices = sd.query_devices()
    except Exception as exc:
        print(f"Microphone enumeration failed: {exc}")
        stt_device_name = "Mic: Unavailable"
        return None, None

    first_input = None
    logitech_input = None
    for idx, dev in enumerate(devices):
        max_inputs = int(dev.get("max_input_channels", 0))
        if max_inputs <= 0:
            continue

        if first_input is None:
            first_input = (idx, dev)

        name = str(dev.get("name", ""))
        if "logitech" in name.lower():
            logitech_input = (idx, dev)
            break

    chosen = logitech_input or first_input
    if chosen is None:
        stt_device_name = "Mic: No input device"
        return None, None

    idx, dev = chosen
    stt_device_name = f"Mic: {dev.get('name', 'Input')}"

    default_rate = int(dev.get("default_samplerate") or 16000)
    sample_rate = default_rate if default_rate > 0 else 16000
    return idx, sample_rate


def _stt_loop():
    global stt_converted_text, stt_raw_text, stt_listening, stt_device_index, stt_display_language

    if sr is None:
        set_status("Install SpeechRecognition")
        stt_listening = False
        return

    if sd is None:
        set_status("Install PortAudio for mic")
        stt_listening = False
        return

    if shutil.which("flac") is None:
        print("STT blocked: FLAC binary not found in PATH.")
        set_status("Install FLAC for STT")
        stt_listening = False
        return

    stt_device_index, sample_rate = _preferred_stt_device()
    if stt_device_index is None:
        set_status("No microphone found")
        stt_listening = False
        return

    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True

    print(f"STT using input device index {stt_device_index}")
    if ui_mode == "stt":
        set_status(f"Listening ({stt_capture_language})...")

    lang_code = STT_LANGUAGE_TO_CODE.get(STT_INPUT_LANGUAGE, "en-US")

    while not stt_stop_event.is_set() and ui_mode == "stt":
        try:
            frames = int(sample_rate * STT_CHUNK_SECONDS)
            audio = sd.rec(
                frames,
                samplerate=sample_rate,
                channels=1,
                dtype="int16",
                device=stt_device_index,
                blocking=True,
            )
        except Exception as exc:
            print(f"Microphone read failed: {exc}")
            if ui_mode == "stt":
                set_status("Mic read error")
            time.sleep(0.5)
            continue

        if stt_stop_event.is_set() or ui_mode != "stt":
            break

        try:
            audio_data = sr.AudioData(audio.tobytes(), sample_rate, 2)
            text = recognizer.recognize_google(audio_data, language=lang_code)
            text = text.strip()
            if text:
                if not stt_raw_text:
                    stt_raw_text = text
                elif not stt_raw_text.endswith(text):
                    stt_raw_text = f"{stt_raw_text} {text}"

                stt_converted_text = stt_raw_text
                stt_display_language = STT_INPUT_LANGUAGE
                if ui_mode == "stt":
                    set_status(f"Recognizing ({STT_INPUT_LANGUAGE})...")
            else:
                if ui_mode == "stt":
                    set_status(f"Listening ({STT_INPUT_LANGUAGE})...")
        except sr.UnknownValueError:
            if ui_mode == "stt":
                set_status(f"Listening ({STT_INPUT_LANGUAGE})...")
        except sr.RequestError as exc:
            print(f"STT request failed: {exc}")
            if ui_mode == "stt":
                set_status("STT network error")
            time.sleep(0.6)
        except Exception as exc:
            print(f"STT failed: {exc}")
            if ui_mode == "stt":
                set_status("STT processing error")
            time.sleep(0.4)

    stt_listening = False


def start_stt_listening():
    global stt_thread, stt_listening, stt_raw_text, stt_converted_text, stt_capture_language, stt_display_language

    if stt_listening:
        return

    stt_capture_language = STT_INPUT_LANGUAGE
    stt_display_language = stt_capture_language
    stt_raw_text = ""
    stt_converted_text = ""
    stt_stop_event.clear()
    stt_listening = True
    set_status(f"Starting microphone ({STT_INPUT_LANGUAGE})...")
    stt_thread = threading.Thread(target=_stt_loop, daemon=True)
    stt_thread.start()


def stop_stt_listening(convert_on_stop=False):
    global stt_listening, stt_thread
    stt_stop_event.set()
    stt_listening = False

    if stt_thread is not None and stt_thread.is_alive():
        stt_thread.join(timeout=STT_CHUNK_SECONDS + 0.6)

    if convert_on_stop:
        finalize_stt_conversion()


def translate_text_between_languages(text, source_language, target_language):
    if not text:
        return ""

    source_code = LANGUAGE_TO_CODE.get(source_language, "en")
    target_code = LANGUAGE_TO_CODE.get(target_language, "en")
    if source_code == target_code:
        return text

    cache_key = (text, source_code, target_code)
    if cache_key in translation_cache:
        return translation_cache[cache_key]

    try:
        translated = GoogleTranslator(source=source_code, target=target_code).translate(text)
        translation_cache[cache_key] = translated
        return translated
    except Exception as exc:
        print(f"Translate failed ({source_language}->{target_language}): {exc}")
        return text


def finalize_stt_conversion():
    global stt_converted_text, stt_display_language

    raw_text = stt_raw_text.strip()
    if not raw_text:
        set_status("No speech captured")
        return

    target_language = selected_language
    source_language = STT_INPUT_LANGUAGE

    converted = translate_text_between_languages(raw_text, source_language, target_language)
    stt_converted_text = converted.strip() if converted.strip() else raw_text
    stt_display_language = target_language
    set_status(f"Converted English -> {target_language}")


def toggle_stt_listening():
    if stt_listening:
        stop_stt_listening(convert_on_stop=True)
    else:
        start_stt_listening()


def enter_stt_mode():
    global ui_mode
    ui_mode = "stt"
    print("Opening STT screen")
    set_status("STT ready - press Mic Start")


def enter_camera_mode():
    global ui_mode
    ui_mode = "camera"
    stop_stt_listening(convert_on_stop=False)
    print("Returned to camera screen")
    set_status("Idle")


def reload_last_buffer():
    global stored_buffer_lines
    if not last_output_buffer_lines:
        print("Reload requested, but no previous output buffer available.")
        set_status("Idle")
        return
    stored_buffer_lines = list(last_output_buffer_lines)
    print("Reloaded last buffer:", " ".join(stored_buffer_lines))
    set_status("Idle")


def clear_buffer():
    global stored_buffer_lines, detected_signs, last_buffered_sign
    stored_buffer_lines = []
    detected_signs = []
    last_buffered_sign = None
    print("Buffer cleared")
    set_status("Idle")


def request_exit():
    global exit_requested
    exit_requested = True
    set_status("Exiting")


def build_sentence_from_signs(sign_lines):
    # Convert dataset-style tokens like HELLO_WORLD into natural spoken text.
    normalized = [token.replace("_", " ").strip() for token in sign_lines if token and token.strip()]
    if len(normalized) > 1:
        sentence = ". ".join(normalized)
    elif normalized:
        sentence = normalized[0]
    else:
        sentence = ""
    sentence = re.sub(r"\s+", " ", sentence).strip()
    return sentence


def split_text_into_sentences(text):
    # Fast path: single line (most gesture outputs)
    if "." not in text and "!" not in text and "?" not in text:
        return [text.strip()] if text.strip() else []
    # Multi-sentence case
    parts = [p.strip() for p in re.split(r"[.!?]+", text) if p.strip()]
    return parts if parts else ([text.strip()] if text.strip() else [])


def collapse_spelled_acronyms(text):
    # Example: "H A SH" -> "HASH", keeps model/team names from sounding broken.
    # Only process if spaces are present (fast check)
    if " " not in text:
        return text
    return re.sub(r"\b(?:[A-Z]\s+){2,}[A-Z]\b", lambda m: m.group(0).replace(" ", ""), text)


def normalize_for_translation(text):
    cleaned = re.sub(r"\s+", " ", text).strip()
    
    # Fast path: already normalized
    if not cleaned or cleaned[0].islower():
        return cleaned
    
    cleaned = collapse_spelled_acronyms(cleaned)

    # Sign labels are often ALL CAPS; translating lowercase yields better quality.
    letters = re.sub(r"[^A-Za-z]", "", cleaned)
    if letters and letters.isupper() and len(letters) > 1:
        cleaned = cleaned.lower()
        cleaned = re.sub(r"\bi\b", "I", cleaned)
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:]

    return cleaned


def run_now_without_confirm():
    set_status("Processing")

    output_lines = []
    if stored_buffer_lines:
        output_lines = stored_buffer_lines[-8:]
    elif detected_signs:
        output_lines = detected_signs[-8:]

    if not output_lines:
        print("Run requested, but no signs in buffer.")
        set_status("Idle")
        return

    sentence = build_sentence_from_signs(output_lines)
    print("Run output sentence:", sentence)
    last_output_buffer_lines[:] = output_lines
    speak(sentence)
    set_status("Idle")


def translate_text_for_language(text, language):
    return translate_text_between_languages(text, "English", language)


def handle_ui_action(action, value=None):
    if action == "lang":
        set_language(value)
    elif action == "stt":
        enter_stt_mode()
    elif action == "toggle_mic":
        toggle_stt_listening()
    elif action == "back_camera":
        enter_camera_mode()
    elif action == "reload":
        reload_last_buffer()
    elif action == "run":
        run_now_without_confirm()
    elif action == "clear":
        clear_buffer()
    elif action == "exit":
        request_exit()


def play_audio_file(path):
    # Prefer ffplay first to avoid playsound's gi backend dependency on Linux.
    ffplay = shutil.which("ffplay")
    if ffplay:
        try:
            proc = subprocess.run(
                [
                    ffplay,
                    "-nodisp",
                    "-autoexit",
                    "-loglevel",
                    "quiet",
                    "-af",
                    f"volume={AUDIO_BOOST_MULTIPLIER}",
                    path,
                ],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if proc.returncode == 0:
                print("TTS: playback via ffplay")
                return True
        except Exception:
            pass

    cvlc = shutil.which("cvlc")
    if cvlc:
        try:
            proc = subprocess.run(
                [
                    cvlc,
                    "--play-and-exit",
                    "--intf",
                    "dummy",
                    "--gain",
                    str(AUDIO_BOOST_MULTIPLIER),
                    path,
                ],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if proc.returncode == 0:
                print("TTS: playback via cvlc")
                return True
        except Exception:
            pass

    mpg123 = shutil.which("mpg123")
    if mpg123:
        try:
            proc = subprocess.run(
                [
                    mpg123,
                    "-q",
                    "-f",
                    str(int(32768 * AUDIO_BOOST_MULTIPLIER)),
                    path,
                ],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if proc.returncode == 0:
                print("TTS: playback via mpg123")
                return True
        except Exception:
            pass

    try:
        playsound(path)
        print("TTS: playback via playsound")
        return True
    except Exception as exc:
        print(f"Audio playback failed with playsound: {exc}")

    return False


def play_with_edge_playback(text):
    edge_playback = shutil.which("edge-playback")
    if not edge_playback:
        return False

    try:
        print("TTS: using edge-playback (streaming)...")
        proc = subprocess.run(
            [
                edge_playback,
                "--text",
                text,
                "--voice",
                LANGUAGE_TO_VOICE.get(selected_language, VOICE_NAME),
                "--rate",
                tts_percent_str(TTS_RATE_PERCENT),
                "--volume",
                f"+{AUDIO_BOOST_PERCENT}%",
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if proc.returncode == 0:
            print("TTS: edge-playback finished")
            return True
        print(f"TTS: edge-playback exited with code {proc.returncode}, falling back")
        return False
    except Exception as exc:
        print(f"TTS: edge-playback failed ({exc}), falling back to mp3 path")
        return False


def cache_file_path(text, voice):
    key = hashlib.sha1(
        f"{voice}|{AUDIO_BOOST_PERCENT}|{TTS_RATE_PERCENT}|{text}".encode("utf-8")
    ).hexdigest()
    return os.path.join(TTS_CACHE_DIR, f"{key}.mp3")


async def speak_async(text):
    if not text:
        return
    voice = LANGUAGE_TO_VOICE.get(selected_language, VOICE_NAME)

    # Split first, then translate each sentence for better language quality.
    source_chunks = split_text_into_sentences(text)
    if not source_chunks:
        return

    if selected_language == "English":
        translated_chunks = [normalize_for_translation(c) for c in source_chunks]
    else:
        # Faster path: one translation call for the whole text.
        prepared_chunks = [normalize_for_translation(c) for c in source_chunks]
        delimiter = " ||| "
        prepared_full = delimiter.join(prepared_chunks)
        translated_full = translate_text_for_language(prepared_full, selected_language)

        if "|||" in translated_full:
            translated_chunks = [c.strip() for c in translated_full.split("|||") if c.strip()]
        else:
            translated_chunks = split_text_into_sentences(translated_full)

        if not translated_chunks:
            translated_chunks = [translated_full]

        print(f"Translated ({selected_language}): {' '.join(translated_chunks)}")

    for idx, chunk in enumerate(translated_chunks):
        if idx > 0:
            await asyncio.sleep(SENTENCE_GAP_SECONDS)

        cached_path = cache_file_path(chunk, voice)
        print(f"TTS: start -> {chunk}")

        # Lowest-latency FIRST attempt: stream directly to player without waiting for file.
        if play_with_edge_playback(chunk):
            print("TTS: done")
            continue

        # Fast path: if phrase was spoken before, play cached audio instantly.
        if os.path.exists(cached_path):
            print("TTS: cache hit")
            if play_audio_file(cached_path):
                print("TTS: done")
                continue

        communicate = edge_tts.Communicate(
            text=chunk,
            voice=voice,
            rate=tts_percent_str(TTS_RATE_PERCENT),
            volume=f"+{AUDIO_BOOST_PERCENT}%",
        )
        temp_path = f"{cached_path}.tmp"
        try:
            print("TTS: generating mp3 with edge-tts...")
            await communicate.save(temp_path)
            os.replace(temp_path, cached_path)
            if not play_audio_file(cached_path):
                print("Audio playback failed: no usable player found.")
            else:
                print("TTS: playback done")
        except Exception as exc:
            print(f"TTS failed: {exc}")
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            print("TTS: done")


def speak(text):
    def _runner():
        try:
            asyncio.run(speak_async(text))
        except Exception:
            pass

    threading.Thread(target=_runner, daemon=True).start()

# ?? CLEAN EXIT HANDLER
def cleanup(sig=None, frame=None):
    print("\nClosing safely...")
    stop_stt_listening()
    try:
        cap.release()
    except:
        pass
    try:
        hands.close()
    except:
        pass
    kill_camera_users(active_camera_path)
    cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)   # Ctrl+C safe
signal.signal(signal.SIGTERM, cleanup)

if has_display:
    cv2.setMouseCallback(
        "G2S SYSTEM",
        lambda event, x, y, flags, param: (
            camera_ui.handle_mouse(event, x, y, lambda action, value=None: handle_ui_action(action, value))
            if ui_mode == "camera"
            else stt_ui.handle_mouse(event, x, y, lambda action, value=None: handle_ui_action(action, value))
        ),
    )

last_result = None
last_hand_landmarks = None
last_detection = False
last_detection_ts = 0.0
fullscreen_applied = False

# Process more often for smoother detection.
PROCESS_EVERY_N_FRAMES = 1
LANDMARK_HOLD_SECONDS = 1.2
frame_count = 0

buffer = []
detected_signs = []
last_buffered_sign = None
last_confirm_empty_ts = 0.0
last_logged_gesture = None

while True:
    if exit_requested:
        cleanup()

    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    frame_count += 1

    if has_display:
        camera_frame = cv2.resize(frame, (CAMERA_VIEW_W, CAMERA_VIEW_H))
        display_frame = np.zeros((CAMERA_VIEW_H, WINDOW_W, 3), dtype=np.uint8)
        display_frame[:, :CAMERA_VIEW_W] = camera_frame

    if frame_count % PROCESS_EVERY_N_FRAMES == 0:

        detection_frame = cv2.resize(frame, (DETECTION_W, DETECTION_H), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)

        res = hands.process(rgb)

        if res.multi_hand_landmarks:
            last_detection = True
            last_detection_ts = time.time()
            for hl in res.multi_hand_landmarks:
                last_hand_landmarks = hl
                norm = normalize(hl.landmark)

                gesture, buffer = backend.process(norm)

                if gesture:
                    last_result = gesture
                    if VERBOSE_DETECTIONS and gesture != last_logged_gesture:
                        print("Detected:", gesture)
                        last_logged_gesture = gesture
                    if gesture == "CONFIRM":
                        output_lines = []
                        if detected_signs:
                            output_lines = detected_signs[-8:]
                        elif stored_buffer_lines:
                            output_lines = stored_buffer_lines[-8:]

                        if output_lines:
                            sentence = build_sentence_from_signs(output_lines)
                            print("Output sentence:", sentence)
                            last_output_buffer_lines[:] = output_lines
                            speak(sentence)

                            # Clear current working buffer after CONFIRM output.
                            detected_signs.clear()
                            stored_buffer_lines.clear()
                            last_buffered_sign = None
                        else:
                            now_ts = time.time()
                            if now_ts - last_confirm_empty_ts > 1.0:
                                print("CONFIRM detected, but no signs to output.")
                                last_confirm_empty_ts = now_ts
                    else:
                        if gesture != last_buffered_sign:
                            detected_signs.append(gesture)
                            last_buffered_sign = gesture
                            stored_buffer_lines[:] = detected_signs[-8:]
                break
        else:
            if last_detection and (time.time() - last_detection_ts) > LANDMARK_HOLD_SECONDS:
                last_detection = False
                last_hand_landmarks = None

        # Auto-sync buffer box with current collected signs when signs exist.
        # Keep last output visible after CONFIRM clears runtime detections.
        if detected_signs:
            stored_buffer_lines[:] = detected_signs[-8:]

    if has_display:
        if last_hand_landmarks is not None:
            mp_drawing.draw_landmarks(
                camera_frame,
                last_hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

        if last_result:
            cv2.putText(camera_frame, f"Gesture: {last_result}",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 0), 2)

        now = time.perf_counter()
        dt = max(now - prev_time, 1e-6)
        prev_time = now
        fps_history.append(1.0 / dt)
        fps = sum(fps_history) / len(fps_history)

        cv2.putText(camera_frame, f"FPS: {fps:.1f}",
                    (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 2)

        status = "Hand: Detected" if last_detection else "Hand: Searching..."
        status_color = (0, 255, 0) if last_detection else (0, 165, 255)
        cv2.putText(camera_frame, status,
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, status_color, 2)

        if detected_signs:
            text = "Signs: " + " ".join(detected_signs[-6:])
            cv2.putText(camera_frame, text,
                        (20, 155),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)

        if ui_mode == "camera":
            # Copy updated camera area and draw separate UI panel.
            display_frame[:, :CAMERA_VIEW_W] = camera_frame
            camera_ui.draw(display_frame, stored_buffer_lines, selected_language, ui_status)
            cv2.imshow("G2S SYSTEM", display_frame)
        else:
            stt_frame = np.zeros((CAMERA_VIEW_H, WINDOW_W, 3), dtype=np.uint8)
            stt_ui.draw(
                stt_frame,
                selected_language,
                stt_converted_text,
                ui_status,
                mic_listening=stt_listening,
                mic_label=stt_device_name,
                text_language=stt_display_language,
            )
            cv2.imshow("G2S SYSTEM", stt_frame)

        # Apply fullscreen after first frame is shown; then re-assert periodically.
        if not fullscreen_applied:
            cv2.resizeWindow("G2S SYSTEM", WINDOW_W, CAMERA_VIEW_H)
            cv2.setWindowProperty("G2S SYSTEM", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.moveWindow("G2S SYSTEM", 0, 0)
            cv2.waitKey(1)
            fullscreen_applied = True
        elif frame_count % 60 == 0:
            cv2.setWindowProperty("G2S SYSTEM", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.moveWindow("G2S SYSTEM", 0, 0)

        # ESC = clean exit
        if cv2.waitKey(1) == 27:
            cleanup()

# fallback (never reached normally)
cleanup()
