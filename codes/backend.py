import os
import joblib
import numpy as np
import time
from collections import deque

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")

MODEL = joblib.load(MODEL_PATH)

VOTE_LEN = 6
CONF_THRESHOLD = 0.7
COOLDOWN = 1.5

pred_queue = deque(maxlen=VOTE_LEN)
sentence_buffer = []
last_time = 0


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
    global last_time, sentence_buffer

    feat = extract_features(landmarks).reshape(1, -1)
    probs = MODEL.predict_proba(feat)[0]

    idx = np.argmax(probs)
    label = MODEL.classes_[idx]
    conf = probs[idx]

    if conf < CONF_THRESHOLD:
        return None, sentence_buffer

    pred_queue.append(label)

    if len(pred_queue) == VOTE_LEN:
        majority = max(set(pred_queue), key=pred_queue.count)
        pred_queue.clear()

        if time.time() - last_time > COOLDOWN:
            last_time = time.time()
            sentence_buffer.append(majority)
            return majority, sentence_buffer

    return None, sentence_buffer
