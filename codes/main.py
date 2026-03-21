import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import mediapipe as mp
import numpy as np
import backend
import time
import threading
from playsound import playsound
import signal
import sys

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

    cv2.imshow("G2S SYSTEM", display_frame)

    # ESC = clean exit
    if cv2.waitKey(1) == 27:
        cleanup()

# fallback (never reached normally)
cleanup()
