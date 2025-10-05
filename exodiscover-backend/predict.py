import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# --- Load CSV ---
df = pd.read_csv(r"model\lighkurve_KOI_dataset.csv")

# --- Features ---
feature_cols = ['koi_period', 'koi_period_err1', 'koi_period_err2',
                'koi_time0bk', 'koi_time0bk_err1', 'koi_time0bk_err2',
                'koi_duration', 'koi_duration_err1', 'koi_duration_err2',
                'koi_quarters']

X = df[feature_cols].apply(pd.to_numeric, errors='coerce')
X = X.fillna(X.mean())

# --- Labels (for evaluation) ---
y = df['koi_disposition']

# --- Split 70/30 (same as training) ---
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# --- Load model, scaler, and label encoder ---
model = load_model("lightcurve_model.h5")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# --- Scale test features ---
X_test_scaled = scaler.transform(X_test)

# --- Predict ---
probs = model.predict(X_test_scaled)
pred_indices = np.argmax(probs, axis=1)
pred_labels = le.inverse_transform(pred_indices)

# --- Create a DataFrame with predictions ---
results = pd.DataFrame({
    "KOI_index": X_test.index,
    "Actual": y_test.values,
    "Predicted": pred_labels
})

# Add probability columns for each class
for i, class_name in enumerate(le.classes_):
    results[f"Prob_{class_name}"] = probs[:, i]

# --- Save predictions to CSV ---
results.to_csv("predictions_test_set.csv", index=False)
print("Predictions saved to predictions_test_set.csv")

# --- Save predictions to pickle ---
with open("predictions_test_set.pkl", "wb") as f:
    pickle.dump(results, f)
print("Predictions saved to predictions_test_set.pkl")

# --- Compute accuracy ---
accuracy = accuracy_score(y_test, pred_labels)
print(f"\nTest Set Accuracy: {accuracy:.3f}")

# --- Classification report ---
print("\nClassification Report:")
print(classification_report(y_test, pred_labels, target_names=le.classes_))

# --- Show first few predictions ---
print("\nSample predictions:")
print(results.head())
