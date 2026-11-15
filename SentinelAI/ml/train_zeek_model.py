from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "ddos_cleaned.csv"
MODEL_PATH = ROOT / "models" / "zeek_ddos_model.pkl"
RESULT_PATH = ROOT / "models" / "zeek_evaluation_results.json"


FEATURES = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Average Packet Size",
    "Down/Up Ratio",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
]


# -----------------------------
# 1. Load dataset
# -----------------------------
print("Loading dataset...")

df = pd.read_csv(
    DATA_PATH,
    usecols=FEATURES + ["Label"],
)

print(f"Rows: {len(df)}")
print(f"Features: {len(FEATURES)}")


# -----------------------------
# 2. Clean invalid values
# -----------------------------
X = df[FEATURES].copy()
y = df["Label"].astype(int)

X = X.replace([np.inf, -np.inf], np.nan)

# Network quantities cannot be negative.
NON_NEGATIVE_FEATURES = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Average Packet Size",
    "Down/Up Ratio",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
]

for column in NON_NEGATIVE_FEATURES:
    X.loc[X[column] < 0, column] = np.nan

X = X.fillna(0)


# -----------------------------
# 3. Chronological 80/20 split
# -----------------------------
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
y_train = y.iloc[:split_index]

X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]

print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")


# -----------------------------
# 4. Train Random Forest
# -----------------------------
print("\nTraining Zeek-compatible model...")

model = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
)

model.fit(X_train, y_train)


# -----------------------------
# 5. Evaluate
# -----------------------------
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)
f1 = f1_score(y_test, predictions, zero_division=0)
cm = confusion_matrix(y_test, predictions)


print("\nZeek-Compatible Model Evaluation")
print("--------------------------------")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)


# -----------------------------
# 6. Feature importance
# -----------------------------
importance = sorted(
    zip(FEATURES, model.feature_importances_),
    key=lambda x: x[1],
    reverse=True,
)

print("\nFeature Importance:")
for feature, score in importance:
    print(f"{feature}: {score:.6f}")


# -----------------------------
# 7. Save model
# -----------------------------
model_bundle = {
    "model": model,
    "features": FEATURES,
    "model_type": "Zeek-compatible Random Forest",
    "training_features": 16,
    "split_method": "chronological_80_20",
}

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(model_bundle, MODEL_PATH)


# -----------------------------
# 8. Save evaluation results
# -----------------------------
results = {
    "model_type": "Zeek-compatible Random Forest",
    "features": FEATURES,
    "training_samples": int(len(y_train)),
    "test_samples": int(len(y_test)),
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "confusion_matrix": cm.tolist(),
    "split_method": "chronological_80_20",
    "n_estimators": 150,
    "random_state": 42,
}

with open(RESULT_PATH, "w") as f:
    json.dump(results, f, indent=4)


print("\nModel saved!")
print(f"Model:   {MODEL_PATH}")
print(f"Results: {RESULT_PATH}")
