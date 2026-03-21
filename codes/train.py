import os
import sys
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report

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

if len(X) == 0:
    raise ValueError("Dataset is empty. Run collect.py first.")

classes, counts = np.unique(y, return_counts=True)
if len(classes) < 2:
    raise ValueError("Need at least 2 gesture classes to train.")
if np.min(counts) < 2:
    raise ValueError("Each class needs at least 2 samples for train/test split.")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build pipeline: scaling + SVM
pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("svc", SVC(probability=True, random_state=42)),
    ]
)

# Use CV folds that fit the smallest class.
cv_folds = int(min(5, np.min(np.bincount(np.unique(y_train, return_inverse=True)[1]))))
if cv_folds < 2:
    cv_folds = 2

param_grid = [
    {
        "svc__kernel": ["rbf"],
        "svc__C": [1, 10, 30, 100],
        "svc__gamma": ["scale", 0.01, 0.005, 0.001],
    },
    {
        "svc__kernel": ["poly"],
        "svc__C": [1, 10, 30],
        "svc__gamma": ["scale", 0.01],
        "svc__degree": [2, 3],
    },
]

cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="accuracy",
    cv=cv,
    n_jobs=1,
    verbose=0,
)

print("Training model... please wait")
search.fit(X_train, y_train)
model = search.best_estimator_

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Best params: {search.best_params_}")
print(f"CV accuracy: {search.best_score_ * 100:.2f}%")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification report:")
print(classification_report(y_test, model.predict(X_test), zero_division=0))

# Save model
joblib.dump(model, MODEL_PATH)
print("Model saved!")
print("Training complete. Exiting.")
sys.exit(0)
