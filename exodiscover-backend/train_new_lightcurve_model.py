import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
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
print("TRAINING NEW LIGHTCURVE MODEL (ANTI-OVERFITTING)")
print("=" * 60)

# Load KOI dataset
print("\nLoading lightkurve_KOI_dataset.csv...")
koi_df = pd.read_csv('model/lighkurve_KOI_dataset.csv')
print(f"Total KOI records: {len(koi_df)}")

# Map dispositions to labels
disposition_map = {
    'CONFIRMED': 2,
    'CANDIDATE': 1,
    'FALSE POSITIVE': 0
}

koi_df['label'] = koi_df['koi_disposition'].map(disposition_map)

# Count by disposition
print("\nDisposition distribution:")
for disp, label in disposition_map.items():
    count = len(koi_df[koi_df['label'] == label])
    print(f"  {disp}: {count}")

# Select unique KEPIDs to avoid data leakage
print("\nSelecting unique KEPIDs for train/test split...")
unique_kepids = koi_df.groupby('label')['kepid'].apply(lambda x: x.unique()).to_dict()

print("Unique KEPIDs per class:")
for label, kepids in unique_kepids.items():
    label_name = {0: 'FALSE POSITIVE', 1: 'CANDIDATE', 2: 'CONFIRMED'}[label]
    print(f"  {label_name}: {len(kepids)} unique stars")

# Sample KEPIDs for training (not individual windows)
# This ensures train and test come from different stars
train_kepids = {}
test_kepids = {}

for label, kepids in unique_kepids.items():
    # Use 70% of stars for training, 30% for testing
    n_train = int(len(kepids) * 0.7)
    np.random.seed(42)
    shuffled = np.random.permutation(kepids)
    train_kepids[label] = shuffled[:n_train]
    test_kepids[label] = shuffled[n_train:]

print("\nTrain/Test split by unique stars:")
for label in [0, 1, 2]:
    label_name = {0: 'FALSE POSITIVE', 1: 'CANDIDATE', 2: 'CONFIRMED'}[label]
    print(f"  {label_name}: {len(train_kepids[label])} train stars, {len(test_kepids[label])} test stars")

print("\nThis prevents data leakage - train and test use different stars!")

# Download light curves for training and test sets
import lightkurve as lk

def download_light_curves(kepids_dict, max_per_class=300):
    """Download light curves for given KEPIDs"""
    all_windows = []
    all_labels = []

    for label, kepids in kepids_dict.items():
        label_name = {0: 'FALSE POSITIVE', 1: 'CANDIDATE', 2: 'CONFIRMED'}[label]
        print(f"\nDownloading {label_name} light curves...")

        count = 0
        for kepid in kepids:
            if count >= max_per_class:
                break

            try:
                search_result = lk.search_lightcurve(f'KIC {kepid}', mission='Kepler', cadence='long')
                if len(search_result) == 0:
                    continue

                lc_collection = search_result.download_all()
                if lc_collection is None or len(lc_collection) == 0:
                    continue

                lc = lc_collection.stitch()
                flux = lc.flux.value
                flux = flux[~np.isnan(flux)]

                if len(flux) < 256:
                    continue

                # Extract windows
                num_windows = min(3, len(flux) // 256)  # Max 3 windows per star
                for i in range(num_windows):
                    if count >= max_per_class:
                        break

                    start_idx = i * (len(flux) // (num_windows + 1))
                    window = flux[start_idx:start_idx + 256]

                    if len(window) == 256:
                        # Normalize
                        window = (window - np.mean(window)) / (np.std(window) + 1e-8)
                        all_windows.append(window)
                        all_labels.append(label)
                        count += 1

                if (count % 50) == 0:
                    print(f"  Downloaded {count}/{max_per_class} windows...")

            except Exception as e:
                continue

        print(f"  Total {label_name}: {count} windows")

    return np.array(all_windows), np.array(all_labels)

print("\n" + "=" * 60)
print("DOWNLOADING TRAINING DATA")
print("=" * 60)
X_train_windows, y_train = download_light_curves(train_kepids, max_per_class=300)

print("\n" + "=" * 60)
print("DOWNLOADING TEST DATA")
print("=" * 60)
X_test_windows, y_test = download_light_curves(test_kepids, max_per_class=100)

# Extract features
print("\n" + "=" * 60)
print("EXTRACTING FEATURES")
print("=" * 60)

feature_names = ['mean', 'std', 'median', 'max', 'min', 'range', 'coefficient_variation',
                 'num_dips', 'dip_fraction', 'max_dip_depth', 'skewness', 'kurtosis',
                 'num_periods', 'primary_period', 'snr', 'smoothness', 'transit_count',
                 'avg_transit_length', 'max_transit_length']

print("\nExtracting training features...")
X_train_features = []
for i, window in enumerate(X_train_windows):
    if (i + 1) % 100 == 0:
        print(f"  {i + 1}/{len(X_train_windows)} windows...")
    features = extract_features(window)
    X_train_features.append([features[name] for name in feature_names])
X_train_features = np.array(X_train_features)

print("\nExtracting test features...")
X_test_features = []
for window in X_test_windows:
    features = extract_features(window)
    X_test_features.append([features[name] for name in feature_names])
X_test_features = np.array(X_test_features)

print(f"\nTraining features: {X_train_features.shape}")
print(f"Test features: {X_test_features.shape}")

# Train with strong regularization to prevent overfitting
print("\n" + "=" * 60)
print("TRAINING MODEL")
print("=" * 60)

model = xgb.XGBClassifier(
    n_estimators=100,  # Reduced from 200
    max_depth=4,  # Shallow trees to prevent overfitting
    learning_rate=0.05,  # Slower learning
    random_state=42,
    eval_metric='mlogloss',
    min_child_weight=5,  # Stronger regularization
    subsample=0.7,  # Use only 70% of data per tree
    colsample_bytree=0.7,  # Use only 70% of features per tree
    reg_alpha=0.1,  # L1 regularization
    reg_lambda=1.0  # L2 regularization
)

# Cross-validation on training set
print("\nPerforming 5-fold cross-validation...")
cv_scores = cross_val_score(model, X_train_features, y_train, cv=5, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Train on full training set
print("\nTraining on full training set...")
model.fit(X_train_features, y_train)

# Evaluate
print("\n" + "=" * 60)
print("EVALUATION")
print("=" * 60)

train_pred = model.predict(X_train_features)
train_acc = accuracy_score(y_train, train_pred)

test_pred = model.predict(X_test_features)
test_acc = accuracy_score(y_test, test_pred)

print(f"\nTraining Accuracy: {train_acc * 100:.2f}%")
print(f"Test Accuracy: {test_acc * 100:.2f}%")
print(f"Difference: {(train_acc - test_acc) * 100:.2f}%")

if train_acc - test_acc > 0.10:
    print("\nWARNING: Large gap indicates overfitting!")
elif train_acc - test_acc > 0.05:
    print("\nMild overfitting detected")
else:
    print("\nGood generalization!")

print("\nTest Set Confusion Matrix:")
cm = confusion_matrix(y_test, test_pred)
print("                  Predicted")
print("                  FP   Cand  Conf")
for i, row in enumerate(cm):
    label_name = ['FP', 'Cand', 'Conf'][i]
    print(f"Actual {label_name:4s}    {row}")

print("\nTest Set Classification Report:")
print(classification_report(y_test, test_pred,
                          target_names=['False Positive', 'Candidate', 'Confirmed']))

# Save model
print("\nSaving model...")
joblib.dump(model, 'model/lightcurve_xgb_model.pkl')
np.save('training_data/windows.npy', X_train_windows)
np.save('training_data/labels.npy', y_train)
print("Model saved!")

print("\n" + "=" * 60)
print("COMPLETE!")
print("=" * 60)
