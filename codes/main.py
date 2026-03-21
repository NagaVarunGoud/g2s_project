import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import mediapipe as mp
import numpy as np
import backend
import auth
import time
import threading
from playsound import playsound
import signal
import sys

# ---------------------------------------------------------------------------
# GitHub sign-in
# ---------------------------------------------------------------------------

# State shared between the sign-in polling thread and the display loop.
_sign_in_code: str = ""
_sign_in_url: str = ""
_sign_in_error: str = ""

def _draw_sign_in_screen(frame: np.ndarray, user: dict | None) -> None:
    """Render sign-in instructions (or welcome message) onto *frame* in-place."""
    h, w = frame.shape[:2]

    if user:
        # Already signed in — show a brief welcome overlay.
        cv2.putText(frame, f"Signed in as: {user.get('login', '')}",
                    (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return

    # Dark overlay so text is readable regardless of background.
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cx = w // 2

    def center(text, y, scale=0.7, color=(255, 255, 255), thickness=1):
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        cv2.putText(frame, text, (cx - tw // 2, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

    center("G2S — GitHub Sign In", 80, scale=1.1, color=(100, 200, 255), thickness=2)

    if _sign_in_error:
        center(_sign_in_error, 160, color=(0, 80, 220))
        center("Press ESC to quit.", h - 40, color=(180, 180, 180))
        return

    if _sign_in_code:
        center("1. Open the URL below in your browser:", 160)
        center(_sign_in_url, 210, color=(80, 200, 255), thickness=2)
        center("2. Enter the code:", 270)
        center(_sign_in_code, 330, scale=1.4, color=(0, 220, 100), thickness=3)
        center("Waiting for GitHub authorization...", 420, color=(200, 200, 200))
    else:
        center("Connecting to GitHub...", 260, color=(200, 200, 200))

    center("Press ESC to quit.", h - 40, color=(180, 180, 180))


def _run_sign_in() -> dict | None:
    """
    Drive the sign-in flow in a background thread and keep the display loop
    informed via the module-level state variables.
    """
    global _sign_in_code, _sign_in_url, _sign_in_error

    def _on_code(user_code: str, verification_uri: str) -> None:
        global _sign_in_code, _sign_in_url
        _sign_in_code = user_code
        _sign_in_url = verification_uri

    try:
        return auth.sign_in(on_code=_on_code)
    except ValueError as exc:
        _sign_in_error = str(exc)
        return None
    except Exception as exc:  # network errors, etc.
        _sign_in_error = f"Sign-in error: {exc}"
        return None


def show_sign_in_screen() -> dict | None:
    """
    Display an OpenCV window with sign-in instructions and block until the
    user has authenticated (or presses ESC to quit).

    Returns the GitHub user profile dict on success, or None if the user
    cancelled / sign-in failed.
    """
    win = "G2S — Sign In"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 900, 500)

    result: list[dict | None] = [None]
    done = threading.Event()

    def _worker():
        result[0] = _run_sign_in()
        done.set()

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    canvas = np.zeros((500, 900, 3), dtype=np.uint8)

    while not done.is_set():
        frame = canvas.copy()
        _draw_sign_in_screen(frame, None)
        cv2.imshow(win, frame)
        if cv2.waitKey(200) == 27:   # ESC
            cv2.destroyWindow(win)
            sys.exit(0)

    cv2.destroyWindow(win)
    return result[0]


# Run sign-in if no session exists.
if not auth.is_signed_in():
    github_user = show_sign_in_screen()
    if github_user is None:
        print("Sign-in failed or was cancelled. Exiting.")
        sys.exit(1)
    print(f"Signed in as: {github_user.get('login', 'unknown')}")
else:
    github_user = auth.get_current_user()
    print(f"Welcome back, {github_user.get('login', 'unknown')}!")

# ---------------------------------------------------------------------------
# Camera + hand tracking
# ---------------------------------------------------------------------------

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ?? FORCE V4L2 backend (IMPORTANT FIX)
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# Try alternate index if needed
if not cap.isOpened():
    print("Trying camera index 1...")
    cap = cv2.VideoCapture(1, cv2.CAP_V4L2)

if not cap.isOpened():
    print("ERROR: Camera not accessible")
    exit()

# Faster camera
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(3, 160)
cap.set(4, 120)

cv2.namedWindow("G2S SYSTEM", cv2.WINDOW_NORMAL)
cv2.resizeWindow("G2S SYSTEM", 900, 700)

prev_time = time.time()

def normalize(landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    coords = coords - coords[0]
    scale = np.max(np.linalg.norm(coords, axis=1))
    coords = coords / (scale if scale else 1)
    return coords.flatten()

# ?? Sound mapping
SOUND_MAP = {
    "HI": "/usr/share/sounds/alsa/Front_Center.wav",
    "GOOD": "/usr/share/sounds/alsa/Front_Center.wav",
    "EXCELLENT": "/usr/share/sounds/alsa/Front_Center.wav"
}

def play_audio(label):
    try:
        if label in SOUND_MAP:
            playsound(SOUND_MAP[label])
    except:
        pass

# ?? CLEAN EXIT HANDLER
def cleanup(sig=None, frame=None):
    print("\nClosing safely...")
    try:
        cap.release()
    except:
        pass
    cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)   # Ctrl+C safe
signal.signal(signal.SIGTERM, cleanup)

last_result = None

PROCESS_EVERY_N_FRAMES = 4
frame_count = 0

buffer = []

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    frame_count += 1

    display_frame = cv2.resize(frame, (900, 700))

    if frame_count % PROCESS_EVERY_N_FRAMES == 0:

        small = cv2.resize(frame, (160, 120))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        res = hands.process(rgb)

        if res.multi_hand_landmarks:
            for hl in res.multi_hand_landmarks:
                norm = normalize(hl.landmark)

                gesture, buffer = backend.process(norm)

                if gesture:
                    last_result = gesture
                    print("Detected:", gesture)

                    threading.Thread(
                        target=play_audio,
                        args=(gesture,),
                        daemon=True
                    ).start()

    if last_result:
        cv2.putText(display_frame, last_result,
                    (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.0, (0, 255, 0), 4)

    fps = int(1 / (time.time() - prev_time + 1e-6))
    prev_time = time.time()

    cv2.putText(display_frame, f"FPS: {fps}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 0, 0), 2)

    # Show signed-in GitHub username in the bottom-left corner.
    if github_user:
        login = github_user.get("login", "")
        cv2.putText(display_frame, f"GitHub: {login}",
                    (20, display_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 200, 100), 1)

    cv2.imshow("G2S SYSTEM", display_frame)

    # ESC = clean exit
    if cv2.waitKey(1) == 27:
        cleanup()

# fallback (never reached normally)
cleanup()
