import os
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")

# Load dataset
dataset, labels = joblib.load(DATA_PATH)


def extract_features(landmarks):
    pts = landmarks.reshape(-1, 3)

    # Distance features
    dists = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dists.append(np.linalg.norm(pts[i] - pts[j]))

    # Angle features
    angles = []
    for i in range(1, len(pts) - 1):
        v1 = pts[i] - pts[i - 1]
        v2 = pts[i + 1] - pts[i]

        cos_angle = np.dot(v1, v2) / (
            np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6
        )
        angles.append(cos_angle)

    return np.concatenate([landmarks, dists, angles])


# Prepare data
X = np.array([extract_features(np.array(d)) for d in dataset])
y = np.array(labels)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = SVC(kernel='rbf', probability=True, C=10)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save model
joblib.dump(model, MODEL_PATH)
print("Model saved!")
