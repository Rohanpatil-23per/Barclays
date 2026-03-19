import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import json
import time

XTEST_PATH = "master_dataset/X_test.csv"
XTRAIN_PATH = "master_dataset/X_train.csv"
YTRAIN_PATH = "master_dataset/y_train.csv"
YTEST_PATH = "master_dataset/y_test.csv"
SCALER_PATH = "models/layer1_scaler.pkl"
MODEL_PATH = "models/isolation_forest.pkl"

print("Loading data...")
X_train = pd.read_csv(XTRAIN_PATH)
X_test = pd.read_csv(XTEST_PATH)
y_test = pd.read_csv(YTEST_PATH)['is_attack']

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")

# Load scaler built by rebuild_dataset.py
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Train only on benign traffic — classic unsupervised anomaly detection
y_train = pd.read_csv(YTRAIN_PATH)['is_attack']
benign_mask = y_train == 0
X_train_benign = X_train_scaled[benign_mask]
print(f"Training on {X_train_benign.shape[0]} benign samples only...")

start = time.time()
iso = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    max_samples=512,
    random_state=42,
    n_jobs=-1,
    verbose=1
)
iso.fit(X_train_benign)
print(f"Training done in {time.time()-start:.1f}s")

# Evaluate
print("\nEvaluating...")
preds = iso.predict(X_test_scaled)          # 1 = normal, -1 = anomaly
scores = iso.decision_function(X_test_scaled)  # higher = more normal

# Convert to binary: anomaly=1, benign=0
y_pred_binary = (preds == -1).astype(int)
y_true_binary = y_test.astype(int)

print("\nClassification Report:")
print(classification_report(y_true_binary, y_pred_binary,
      target_names=["Benign", "Attack"]))

# Save model
with open(MODEL_PATH, "wb") as f:
    pickle.dump(iso, f)
print(f"\nModel saved to {MODEL_PATH}")

# Save anomaly score stats for inference threshold tuning
stats = {
    "score_mean": float(np.mean(scores)),
    "score_std":  float(np.std(scores)),
    "threshold":  float(np.percentile(scores, 5))
}
with open("models/iso_forest_stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print(f"Score stats saved: {stats}")
