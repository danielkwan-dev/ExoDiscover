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
print("TRAINING MODEL WITH 70/30 KEPLER-10 SPLIT + KOI DATA")
print("=" * 60)

# Load Kepler-10 data
print("\nLoading Kepler-10 data...")
kepler10_windows = np.load('lightcurve_data/all_windows.npy')
kepler10_labels = np.full(len(kepler10_windows), 2)  # All Confirmed
print(f"Loaded {len(kepler10_windows)} Kepler-10 windows (all Confirmed)")

# Split Kepler-10 into 70% train, 30% test
kepler10_train_windows, kepler10_test_windows, kepler10_train_labels, kepler10_test_labels = train_test_split(
    kepler10_windows, kepler10_labels, test_size=0.3, random_state=42
)
print(f"  Training set: {len(kepler10_train_windows)} windows")
print(f"  Test set:     {len(kepler10_test_windows)} windows")

# Load KOI training data from previous download
print("\nLoading KOI training data...")
koi_windows = np.load('training_data/windows.npy')
koi_labels = np.load('training_data/labels.npy')
print(f"Loaded {len(koi_windows)} KOI windows")
unique, counts = np.unique(koi_labels, return_counts=True)
for label, count in zip(unique, counts):
    label_name = {0: 'False Positive', 1: 'Candidate', 2: 'Confirmed'}[label]
    print(f"  {label_name}: {count}")

# Combine Kepler-10 training portion with KOI data
print("\nCombining datasets for training...")
train_windows = np.concatenate([kepler10_train_windows, koi_windows])
train_labels = np.concatenate([kepler10_train_labels, koi_labels])
print(f"Total training windows: {len(train_windows)}")
unique, counts = np.unique(train_labels, return_counts=True)
for label, count in zip(unique, counts):
    label_name = {0: 'False Positive', 1: 'Candidate', 2: 'Confirmed'}[label]
    print(f"  {label_name}: {count}")

# Extract features
print("\nExtracting features from training data...")
feature_names = ['mean', 'std', 'median', 'max', 'min', 'range', 'coefficient_variation',
                 'num_dips', 'dip_fraction', 'max_dip_depth', 'skewness', 'kurtosis',
                 'num_periods', 'primary_period', 'snr', 'smoothness', 'transit_count',
                 'avg_transit_length', 'max_transit_length']

train_features = []
for i, window in enumerate(train_windows):
    if (i + 1) % 200 == 0:
        print(f"  Processed {i + 1}/{len(train_windows)} windows...")
    features = extract_features(window)
    train_features.append([features[name] for name in feature_names])

train_features = np.array(train_features)
print(f"Extracted features shape: {train_features.shape}")

# Extract features from Kepler-10 test set
print("\nExtracting features from Kepler-10 test set...")
test_features = []
for window in kepler10_test_windows:
    features = extract_features(window)
    test_features.append([features[name] for name in feature_names])
test_features = np.array(test_features)
print(f"Test features shape: {test_features.shape}")

# Train XGBoost model
print("\nTraining XGBoost model...")
# Calculate class weights for imbalance
unique_classes, class_counts = np.unique(train_labels, return_counts=True)
total_samples = len(train_labels)
class_weights = {int(cls): total_samples / (len(unique_classes) * count)
                 for cls, count in zip(unique_classes, class_counts)}

sample_weights = np.array([class_weights[int(label)] for label in train_labels])

model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    random_state=42,
    eval_metric='mlogloss'
)

model.fit(train_features, train_labels, sample_weight=sample_weights)
print("Training complete!")

# Evaluate on Kepler-10 test set
print("\n" + "=" * 60)
print("TESTING ON HELD-OUT KEPLER-10 DATA (30%)")
print("=" * 60)

predictions = model.predict(test_features)
accuracy = accuracy_score(kepler10_test_labels, predictions)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
print(f"Correct predictions: {np.sum(predictions == kepler10_test_labels)}/{len(kepler10_test_labels)}")

# Show first 20 predictions
print("\nFirst 20 predictions:")
label_map = {0: 'False Positive', 1: 'Candidate', 2: 'Confirmed'}
probabilities = model.predict_proba(test_features)

for i in range(min(20, len(predictions))):
    pred_label = label_map[predictions[i]]
    true_label = label_map[kepler10_test_labels[i]]
    confidence = probabilities[i][predictions[i]] * 100
    status = "CORRECT" if predictions[i] == kepler10_test_labels[i] else "WRONG"
    print(f"Window {i:3d}: {pred_label:15s} ({confidence:5.1f}%) [True: {true_label}] {status}")

# Confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(kepler10_test_labels, predictions)
print(cm)

# Classification report
print("\nClassification Report:")
print(classification_report(kepler10_test_labels, predictions,
                          target_names=['False Positive', 'Candidate', 'Confirmed'],
                          zero_division=0))

# Save model
print("\nSaving model...")
joblib.dump(model, 'model/lightcurve_xgb_model.pkl')
print("Model saved to model/lightcurve_xgb_model.pkl")

# Update training data to include only 70% of Kepler-10
print("\nUpdating training_data files...")
np.save('training_data/windows.npy', train_windows)
np.save('training_data/labels.npy', train_labels)
print(f"Saved {len(train_windows)} training windows and labels")

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print(f"Model trained on: {len(train_windows)} windows")
print(f"  - Kepler-10 (Confirmed): {len(kepler10_train_windows)}")
print(f"  - KOI data: {len(koi_windows)}")
print(f"Tested on: {len(kepler10_test_windows)} held-out Kepler-10 windows")
print(f"Accuracy on unseen Kepler-10: {accuracy * 100:.2f}%")
