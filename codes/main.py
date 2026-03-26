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
from collections import deque
from ui_app import OpenCVCameraUI

if hasattr(cv2, "setLogLevel"):
    try:
        cv2.setLogLevel(0)  # silent
    except Exception:
        pass

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
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
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

LANGUAGE_TO_VOICE = {
    "English": "en-US-JennyNeural",
    "Telugu": "te-IN-MohanNeural",
    "Hindi": "hi-IN-SwaraNeural",
    "French": "fr-FR-DeniseNeural",
    "Spanish": "es-ES-ElviraNeural",
}
LANGUAGE_TO_CODE = {
    "English": "en",
    "Telugu": "te",
    "Hindi": "hi",
    "French": "fr",
    "Spanish": "es",
}
selected_language = "English"
ui_status = "Idle"
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
    set_status(f"Language: {language}")


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

    sentence = " ".join(output_lines)
    print("Run output sentence:", sentence)
    last_output_buffer_lines[:] = output_lines
    speak(sentence)
    set_status("Idle")


def translate_text_for_language(text, language):
    if not text:
        return ""

    code = LANGUAGE_TO_CODE.get(language, "en")
    if code == "en":
        return text

    cache_key = (text, code)
    if cache_key in translation_cache:
        return translation_cache[cache_key]

    try:
        translated = GoogleTranslator(source="en", target=code).translate(text)
        translation_cache[cache_key] = translated
        return translated
    except Exception as exc:
        print(f"Translate failed ({language}): {exc}")
        return text


def handle_ui_action(action, value=None):
    if action == "lang":
        set_language(value)
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
                [ffplay, "-nodisp", "-autoexit", "-loglevel", "quiet", path],
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
                [cvlc, "--play-and-exit", "--intf", "dummy", path],
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
                [mpg123, "-q", path],
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
                "+20%",
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
    key = hashlib.sha1(f"{voice}|{text}".encode("utf-8")).hexdigest()
    return os.path.join(TTS_CACHE_DIR, f"{key}.mp3")


async def speak_async(text):
    if not text:
        return

    translated_text = translate_text_for_language(text, selected_language)
    if translated_text != text:
        print(f"Translated ({selected_language}): {translated_text}")

    voice = LANGUAGE_TO_VOICE.get(selected_language, VOICE_NAME)
    cached_path = cache_file_path(translated_text, voice)

    print(f"TTS: start -> {translated_text}")

    # Fast path: if phrase was spoken before, play cached audio instantly.
    if os.path.exists(cached_path):
        print("TTS: cache hit")
        if play_audio_file(cached_path):
            print("TTS: done")
            return

    # Lowest-latency first attempt: stream directly to player.
    if play_with_edge_playback(translated_text):
        print("TTS: done")
        return

    communicate = edge_tts.Communicate(text=translated_text, voice=voice)
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
        lambda event, x, y, flags, param: camera_ui.handle_mouse(
            event, x, y, lambda action, value=None: handle_ui_action(action, value)
        ),
    )

last_result = None
last_hand_landmarks = None
last_detection = False
fullscreen_applied = False

# Process more often for smoother detection.
PROCESS_EVERY_N_FRAMES = 2
MAX_MISSED_DETECTIONS = 3
missed_detections = 0
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

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        res = hands.process(rgb)

        if res.multi_hand_landmarks:
            last_detection = True
            missed_detections = 0
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
                            sentence = " ".join(output_lines)
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
            missed_detections += 1
            if missed_detections > MAX_MISSED_DETECTIONS:
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

        # Copy updated camera area and draw separate UI panel.
        display_frame[:, :CAMERA_VIEW_W] = camera_frame
        camera_ui.draw(display_frame, stored_buffer_lines, selected_language, ui_status)

        cv2.imshow("G2S SYSTEM", display_frame)

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
