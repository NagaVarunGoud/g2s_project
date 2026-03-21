import os

import cv2
import mediapipe as mp
import numpy as np
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
SAVE_PATH = os.path.join(BASE_DIR, "dataset.pkl")

SAMPLES = 60

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,   # faster
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

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

sentences = input("Enter sentences (comma separated): ").split(",")

has_display = bool(os.environ.get("DISPLAY"))
if not has_display:
    print("No DISPLAY detected. Running in headless mode (no preview window).")
elif os.environ.get("G2S_HEADLESS", "0") == "1":
    has_display = False
    print("Running in headless mode (no preview window).")

cap = cv2.VideoCapture(0)

# Low processing resolution (fast)
cap.set(3, 160)
cap.set(4, 120)

# Bigger display window (GUI only)
if has_display:
    cv2.namedWindow("Collect", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Collect", 800, 600)

stop_requested = False

for sentence in sentences:
    label = sentence.strip().upper().replace(" ", "_")
    print(f"Collecting for: {label}")

    count = 0

    while count < SAMPLES and not stop_requested:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        # Resize for smoother display (upscale only for UI)
        display_frame = cv2.resize(frame, (640, 480))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        norm = None
        if res.multi_hand_landmarks:
            for hl in res.multi_hand_landmarks:
                norm = normalize_landmarks(hl.landmark)
                break

        if has_display:
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
                if norm is not None:
                    status_text += " - Press ENTER to capture"
                else:
                    status_text += " - Show hand"

            cv2.putText(display_frame, status_text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

            cv2.imshow("Collect", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC to quit
                stop_requested = True
                break
            if key == 13 and norm is not None:  # ENTER to capture this sample
                dataset.append(norm)
                labels.append(label)
                dataset.append(augment(norm))
                labels.append(label)
                count += 1
                print(f"{label}: {count}/{SAMPLES}")
        else:
            if norm is not None:
                dataset.append(norm)
                labels.append(label)
                dataset.append(augment(norm))
                labels.append(label)
                count += 1
            # Keep headless progress visible in terminal.
            if count % 10 == 0 and count != 0:
                print(f"{label}: {count}/{SAMPLES}")

if has_display:
    cv2.destroyAllWindows()
cap.release()

joblib.dump((dataset, labels), SAVE_PATH)
print("Dataset saved!")
