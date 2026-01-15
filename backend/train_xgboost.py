import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
from scipy import signal, stats
import warnings
warnings.filterwarnings('ignore')

def extract_features(lightcurve):
    """Extract astronomical features from a light curve"""
    features = {}

    # Basic statistics
    features['mean'] = np.mean(lightcurve)
    features['std'] = np.std(lightcurve)
    features['median'] = np.median(lightcurve)
    features['max'] = np.max(lightcurve)
    features['min'] = np.min(lightcurve)
    features['range'] = features['max'] - features['min']
    features['coefficient_variation'] = features['std'] / features['mean'] if features['mean'] != 0 else 0

    # Transit detection features
    threshold = features['mean'] - 2 * features['std']
    dips = lightcurve < threshold
    features['num_dips'] = np.sum(np.diff(np.concatenate(([0], dips.astype(int), [0]))) == 1)
    features['dip_fraction'] = np.sum(dips) / len(lightcurve)
    features['max_dip_depth'] = features['mean'] - features['min']

    # Statistical moments
    features['skewness'] = stats.skew(lightcurve)
    features['kurtosis'] = stats.kurtosis(lightcurve)

    # Frequency domain features
    fft = np.fft.fft(lightcurve)
    power = np.abs(fft[:len(fft)//2])**2
    freqs = np.fft.fftfreq(len(lightcurve))[:len(fft)//2]

    # Find peaks in power spectrum
    peaks, _ = signal.find_peaks(power, height=np.mean(power))
    features['num_periods'] = len(peaks)

    if len(peaks) > 0:
        primary_peak = peaks[np.argmax(power[peaks])]
        features['primary_period'] = 1.0 / freqs[primary_peak] if freqs[primary_peak] != 0 else 0
    else:
        features['primary_period'] = 0

    # Signal-to-noise ratio
    signal_power = np.var(lightcurve)
    noise_estimate = np.var(np.diff(lightcurve)) / 2
    features['snr'] = signal_power / noise_estimate if noise_estimate != 0 else 0

    # Smoothness
    features['smoothness'] = np.mean(np.abs(np.diff(lightcurve)))

    # Transit characteristics
    below_mean = lightcurve < features['mean']
    transit_groups = []
    current_transit = 0
    for is_below in below_mean:
        if is_below:
            current_transit += 1
        else:
            if current_transit > 0:
                transit_groups.append(current_transit)
            current_transit = 0
    if current_transit > 0:
        transit_groups.append(current_transit)

    features['transit_count'] = len(transit_groups)
    features['avg_transit_length'] = np.mean(transit_groups) if transit_groups else 0
    features['max_transit_length'] = max(transit_groups) if transit_groups else 0

    return features

print("=" * 60)
print("TRAINING MODEL WITH DIVERSE KOI DATA")
print("=" * 60)

# Load the diverse KOI training data (500 each of Confirmed, Candidate, False Positive)
print("\nLoading KOI training data...")
koi_windows = np.load('training_data/windows.npy')
koi_labels = np.load('training_data/labels.npy')

# Check if this already includes Kepler-10
print(f"Loaded {len(koi_windows)} total windows")
unique, counts = np.unique(koi_labels, return_counts=True)
for label, count in zip(unique, counts):
    label_name = {0: 'False Positive', 1: 'Candidate', 2: 'Confirmed'}[label]
    print(f"  {label_name}: {count}")

# If the training data has more than 1500 windows, it includes Kepler-10
# Let's use only the original 1500 KOI windows (500 each)
if len(koi_windows) > 1500:
    print("\nDetected Kepler-10 in training data. Using only diverse KOI data...")
    # Take balanced subset - 500 from each class
    false_pos_idx = np.where(koi_labels == 0)[0][:500]
    candidate_idx = np.where(koi_labels == 1)[0][:500]
    confirmed_idx = np.where(koi_labels == 2)[0][:500]

    selected_idx = np.concatenate([false_pos_idx, candidate_idx, confirmed_idx])
    koi_windows = koi_windows[selected_idx]
    koi_labels = koi_labels[selected_idx]

    print(f"Using {len(koi_windows)} balanced KOI windows (500 each class)")

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    koi_windows, koi_labels, test_size=0.2, random_state=42, stratify=koi_labels
)

print(f"\nTraining set: {len(X_train)} windows")
print(f"Test set: {len(X_test)} windows")

# Extract features
print("\nExtracting features from training data...")
feature_names = ['mean', 'std', 'median', 'max', 'min', 'range', 'coefficient_variation',
                 'num_dips', 'dip_fraction', 'max_dip_depth', 'skewness', 'kurtosis',
                 'num_periods', 'primary_period', 'snr', 'smoothness', 'transit_count',
                 'avg_transit_length', 'max_transit_length']

train_features = []
for i, window in enumerate(X_train):
    if (i + 1) % 200 == 0:
        print(f"  Processed {i + 1}/{len(X_train)} windows...")
    features = extract_features(window)
    train_features.append([features[name] for name in feature_names])

train_features = np.array(train_features)
print(f"Extracted features shape: {train_features.shape}")

# Extract features from test set
print("\nExtracting features from test data...")
test_features = []
for window in X_test:
    features = extract_features(window)
    test_features.append([features[name] for name in feature_names])
test_features = np.array(test_features)
print(f"Test features shape: {test_features.shape}")

# Train XGBoost model
print("\nTraining XGBoost model...")
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,  # Reduced from 8 to prevent overfitting
    learning_rate=0.1,
    random_state=42,
    eval_metric='mlogloss',
    min_child_weight=3,  # Added to prevent overfitting
    subsample=0.8,  # Use 80% of data for each tree
    colsample_bytree=0.8  # Use 80% of features for each tree
)

model.fit(train_features, y_train)
print("Training complete!")

# Evaluate on test set
print("\n" + "=" * 60)
print("TESTING ON HELD-OUT KOI DATA")
print("=" * 60)

predictions = model.predict(test_features)
accuracy = accuracy_score(y_test, predictions)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
print(f"Correct predictions: {np.sum(predictions == y_test)}/{len(y_test)}")

# Confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, predictions)
print("                  Predicted")
print("                  FP   Cand  Conf")
print(f"Actual FP       {cm[0]}")
print(f"       Cand     {cm[1]}")
print(f"       Conf     {cm[2]}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, predictions,
                          target_names=['False Positive', 'Candidate', 'Confirmed']))

# Now test on Kepler-10 (completely unseen data)
print("\n" + "=" * 60)
print("TESTING ON KEPLER-10 (COMPLETELY UNSEEN)")
print("=" * 60)

kepler10_windows = np.load('lightcurve_data/all_windows.npy')
print(f"Loaded {len(kepler10_windows)} Kepler-10 windows")

# Extract features
print("Extracting features...")
kepler10_features = []
for window in kepler10_windows[:20]:  # Test first 20
    features = extract_features(window)
    kepler10_features.append([features[name] for name in feature_names])
kepler10_features = np.array(kepler10_features)

# Predict
kepler10_predictions = model.predict(kepler10_features)
kepler10_probabilities = model.predict_proba(kepler10_features)

label_map = {0: 'False Positive', 1: 'Candidate', 2: 'Confirmed'}

print("\nFirst 20 Kepler-10 predictions:")
confirmed_count = 0
for i in range(len(kepler10_predictions)):
    pred_label = label_map[kepler10_predictions[i]]
    confidence = kepler10_probabilities[i][kepler10_predictions[i]] * 100
    print(f"Window {i:3d}: {pred_label:15s} ({confidence:5.1f}%)")
    if kepler10_predictions[i] == 2:  # Confirmed
        confirmed_count += 1

print(f"\nKepler-10 predictions:")
print(f"  Confirmed: {confirmed_count}/20 ({confirmed_count/20*100:.1f}%)")
print(f"  Candidate: {np.sum(kepler10_predictions == 1)}/20")
print(f"  False Positive: {np.sum(kepler10_predictions == 0)}/20")

if confirmed_count >= 15:
    print("\nResult: GOOD - Model correctly identifies most Kepler-10 as Confirmed")
elif confirmed_count >= 10:
    print("\nResult: FAIR - Model identifies some Kepler-10 as Confirmed")
else:
    print("\nResult: POOR - Model does not identify Kepler-10 correctly")

# Save model
print("\nSaving model...")
joblib.dump(model, 'model/lightcurve_xgb_model.pkl')
print("Model saved to model/lightcurve_xgb_model.pkl")

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print(f"Model trained on: {len(X_train)} diverse KOI windows")
print(f"Test accuracy on KOI: {accuracy * 100:.2f}%")
print(f"Kepler-10 predictions: {confirmed_count}/20 Confirmed")
