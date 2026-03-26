import os
import sys
import time
import shutil
import signal
import subprocess
import re

# Detect display availability BEFORE importing cv2/Qt-linked libraries.
# Without this, Qt will try to connect to X11 and crash when running via SSH
# without X11 forwarding (e.g. plain `ssh user@raspberrypi`).
headless_forced = os.environ.get("G2S_HEADLESS", "0") == "1"
has_display_env = bool(os.environ.get("DISPLAY"))
if not has_display_env or headless_forced:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Reduce noisy backend logs from OpenCV / TF / MediaPipe runtime.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")

import cv2
import mediapipe as mp
import numpy as np
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
SAVE_PATH = os.path.join(BASE_DIR, "dataset.pkl")

SAMPLES = 10
REQUIRED_DETECTIONS = 3
AUTO_CAPTURE_EVERY_N_FRAMES = 4
CAMERA_DEVICE_INDEX = 0
CAMERA_DEVICE_PATH = "/dev/video0"
FALLBACK_SCREEN_W = 480
FALLBACK_SCREEN_H = 320

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,   # faster
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

if hasattr(cv2, "setLogLevel"):
    try:
        cv2.setLogLevel(2)  # errors only
    except Exception:
        pass


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


def kill_camera_users(device_path, include_self=False):
    """Best-effort: terminate processes currently holding the camera device."""
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
    pids = []
    for token in proc.stdout.split():
        if token.isdigit():
            pids.append(int(token))

    current_pid = os.getpid()
    for pid in sorted(set(pids)):
        if pid == current_pid and not include_self:
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            continue


def open_camera_with_sanity(index):
    cap_local = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap_local.isOpened():
        cap_local = cv2.VideoCapture(index)

    if not cap_local.isOpened():
        return None

    # Try reading one frame to verify the camera is truly usable.
    ok, _ = cap_local.read()
    if not ok:
        cap_local.release()
        return None

    return cap_local


def normalize_landmarks(landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    coords = coords - coords[0]
    scale = np.max(np.linalg.norm(coords, axis=1))
    coords = coords / (scale if scale else 1)
    return coords.flatten()

def augment(data):
    data = data.reshape(-1, 3)
    noise = data + np.random.normal(0, 0.01, data.shape)
    return noise.flatten()

dataset, labels = [], []

print("=" * 60)
print("G2S Data Collection")
print("=" * 60)
print("CAMERA SETUP: Connect the camera to the Raspberry Pi")
print("  (not to the local computer running VS Code SSH).")
print("  The script runs ON the Pi, so the Pi must see the camera.")
print()
if not has_display_env or headless_forced:
    print("TIP: To see a live camera preview with hand landmarks,")
    print("  connect a monitor to the Pi, OR enable X11 forwarding:")
    print("    ssh -X <user>@<raspberrypi-ip>")
    print("  then re-run this script.")
    print()

sentences = input("Enter sentences (comma separated): ").split(",")

has_display = has_display_env and not headless_forced
if not has_display:
    print("No DISPLAY detected, so camera preview cannot open.")
    print("Run with a GUI display (Pi monitor) or use X11 forwarding:")
    print("  ssh -X <user>@<raspberrypi-ip>")
    sys.exit(1)

# Release stale camera holders first so we can open a fresh capture session.
kill_camera_users(CAMERA_DEVICE_PATH)
cap = open_camera_with_sanity(CAMERA_DEVICE_INDEX)
if cap is None:
    print("Camera not available.")
    print("Check cable, permissions, and that no other app is using /dev/video0.")
    sys.exit(1)

# Camera tuning for smoother live preview and lower latency.
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

# Low processing resolution (fast)
cap.set(3, 320)
cap.set(4, 240)

view_w, view_h = 800, 600
if has_display:
    screen_size = detect_display_size() or (FALLBACK_SCREEN_W, FALLBACK_SCREEN_H)
    screen_w, screen_h = screen_size
    view_w = max(320, screen_w)
    view_h = max(240, screen_h)
    print(f"Collect display size: {screen_w}x{screen_h} -> window {view_w}x{view_h}")

    cv2.namedWindow("Collect", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Collect", view_w, view_h)
    cv2.setWindowProperty("Collect", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow("Collect", 0, 0)

fullscreen_applied = False

stop_requested = False
clean_sentences = [s.strip() for s in sentences if s.strip()]

for idx, sentence in enumerate(clean_sentences):
    label = sentence.strip().upper().replace(" ", "_")
    print(f"Collecting for: {label}")

    count = 0

    frame_idx = 0
    detected_streak = 0
    fps = 0.0
    prev_ts = time.perf_counter()

    while count < SAMPLES and not stop_requested:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_idx += 1

        now_ts = time.perf_counter()
        dt = max(now_ts - prev_ts, 1e-6)
        prev_ts = now_ts
        instant_fps = 1.0 / dt
        fps = instant_fps if fps == 0.0 else (0.9 * fps + 0.1 * instant_fps)

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        norm = None
        if res.multi_hand_landmarks:
            for hl in res.multi_hand_landmarks:
                norm = normalize_landmarks(hl.landmark)
                break

        if norm is not None:
            detected_streak += 1
        else:
            detected_streak = 0

        # Render preview at actual display size for consistent UI.
        display_frame = cv2.resize(frame, (view_w, view_h))

        text_scale_main = 0.55 if view_h <= 360 else 0.7
        text_scale_fps = 0.50 if view_h <= 360 else 0.65
        text_thickness = 1 if view_h <= 360 else 2
        text_y1 = 24 if view_h <= 360 else 30
        text_y2 = 48 if view_h <= 360 else 60
        # Draw hand landmarks on the display frame
        if res.multi_hand_landmarks:
            for hl in res.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    display_frame,
                    hl,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        # UI overlay
        status_text = f"{label}: {count}/{SAMPLES}"
        if count < SAMPLES:
            if norm is None:
                status_text += " - No landmarks (paused)"
            elif detected_streak >= REQUIRED_DETECTIONS:
                status_text += " - Auto capturing..."
            else:
                status_text += " - Hold hand steady"

        cv2.putText(display_frame, status_text,
                    (10, text_y1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    text_scale_main, (0, 255, 0), text_thickness)
        cv2.putText(display_frame,
                f"FPS: {fps:.1f}",
                (10, text_y2),
                cv2.FONT_HERSHEY_SIMPLEX,
                text_scale_fps, (255, 255, 0), text_thickness)

        cv2.imshow("Collect", display_frame)

        if has_display and not fullscreen_applied:
            cv2.resizeWindow("Collect", view_w, view_h)
            cv2.setWindowProperty("Collect", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.moveWindow("Collect", 0, 0)
            cv2.waitKey(1)
            fullscreen_applied = True
        elif has_display and frame_idx % 60 == 0:
            cv2.setWindowProperty("Collect", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.moveWindow("Collect", 0, 0)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to quit
            stop_requested = True
            break

        # Auto-capture only when landmarks are stable for a few frames.
        if (
            norm is not None
            and detected_streak >= REQUIRED_DETECTIONS
            and frame_idx % AUTO_CAPTURE_EVERY_N_FRAMES == 0
        ):
            dataset.append(norm)
            labels.append(label)
            dataset.append(augment(norm))
            labels.append(label)
            count += 1
            print(f"{label}: {count}/{SAMPLES}")

    if stop_requested:
        break

    if idx < len(clean_sentences) - 1:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            pause_frame = cv2.resize(frame, (640, 480))
            cv2.putText(
                pause_frame,
                "Phase complete: Press ENTER for next phrase",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                pause_frame,
                "Press ESC to quit",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 200, 255),
                2,
            )
            cv2.imshow("Collect", pause_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                stop_requested = True
                break
            if key == 13:
                break

        if stop_requested:
            break

if has_display:
    cv2.destroyAllWindows()
cap.release()
# Best-effort cleanup in case another process grabbed the camera unexpectedly.
kill_camera_users(CAMERA_DEVICE_PATH)

joblib.dump((dataset, labels), SAVE_PATH)
print("Dataset saved!")
