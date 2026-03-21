import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import mediapipe as mp
import numpy as np
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
SAVE_PATH = os.path.join(BASE_DIR, "dataset.pkl")

SAMPLES = 60

mp_hands = mp.solutions.hands
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

cap = cv2.VideoCapture(0)

# Low processing resolution (fast)
cap.set(3, 160)
cap.set(4, 120)

# Bigger display window
cv2.namedWindow("Collect", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Collect", 800, 600)

for sentence in sentences:
    label = sentence.strip().upper().replace(" ", "_")
    print(f"Collecting for: {label}")

    count = 0

    while count < SAMPLES:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        # Resize for smoother display (upscale only for UI)
        display_frame = cv2.resize(frame, (640, 480))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        ready_to_capture = False
        if res.multi_hand_landmarks:
            for hl in res.multi_hand_landmarks:
                norm = normalize_landmarks(hl.landmark)

                dataset.append(norm)
                labels.append(label)

                dataset.append(augment(norm))
                labels.append(label)

                count += 1
                ready_to_capture = True

        # UI overlay
        status_text = f"{label}: {count}/{SAMPLES}"
        if ready_to_capture and count < SAMPLES:
            status_text += " - Press ENTER for next sample"
        
        cv2.putText(display_frame, status_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

        cv2.imshow("Collect", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to quit
            break
        elif key == 13:  # ENTER to next sample (gap between sessions)
            if ready_to_capture:
                continue  # Go to next iteration for next sample

cv2.destroyAllWindows()
cap.release()

joblib.dump((dataset, labels), SAVE_PATH)
print("Dataset saved!")
