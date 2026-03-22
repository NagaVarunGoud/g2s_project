import os
import joblib
import numpy as np
import time
from collections import deque
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train.py first.")

MODEL = joblib.load(MODEL_PATH)

VOTE_LEN = 2
CONF_THRESHOLD = 0.5
MIN_GAP_ANY = 0.0
COOLDOWN_SAME_LABEL = 0.1

pred_queue = deque(maxlen=VOTE_LEN)
sentence_buffer = []
last_time = 0
last_label = None
last_raw_label = None


def extract_features(landmarks):
    pts = landmarks.reshape(-1, 3)

    dists = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dists.append(np.linalg.norm(pts[i] - pts[j]))

    angles = []
    for i in range(1, len(pts) - 1):
        v1 = pts[i] - pts[i - 1]
        v2 = pts[i + 1] - pts[i]

        cos_angle = np.dot(v1, v2) / (
            np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6
        )
        angles.append(cos_angle)

    return np.concatenate([landmarks, dists, angles])


def process(landmarks):
    global last_time, sentence_buffer, last_label, last_raw_label

    feat = extract_features(landmarks).reshape(1, -1)
    probs = MODEL.predict_proba(feat)[0]

    idx = np.argmax(probs)
    label = MODEL.classes_[idx]
    conf = probs[idx]

    if conf < CONF_THRESHOLD:
        pred_queue.clear()
        return None, sentence_buffer

    # When classifier output changes, drop old vote history for faster switching.
    if last_raw_label is not None and label != last_raw_label:
        pred_queue.clear()
    last_raw_label = label

    pred_queue.append(label)

    if len(pred_queue) == VOTE_LEN:
        majority = Counter(pred_queue).most_common(1)[0][0]
        pred_queue.clear()

        now = time.time()

        # Small global gap avoids bursts; longer cooldown only for identical repeats.
        if now - last_time < MIN_GAP_ANY:
            return None, sentence_buffer
        if majority == last_label and now - last_time < COOLDOWN_SAME_LABEL:
            return None, sentence_buffer

        last_time = now
        last_label = majority
        sentence_buffer.append(majority)
        return majority, sentence_buffer

    return None, sentence_buffer
